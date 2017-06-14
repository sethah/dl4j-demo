package com.cloudera.datascience.dl4j.cnn

import java.io.File

import scala.collection.JavaConverters._
import com.cloudera.datascience.dl4j.cnn.spark.BroadcastComputationGraph
import org.apache.spark.api.java.function.PairFunction
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SQLContext, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.functions._
import org.datavec.image.loader.NativeImageLoader
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.Layer
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.layers.feedforward.dense.DenseLayer
import org.deeplearning4j.nn.transferlearning.{FineTuneConfiguration, TransferLearning}
import org.deeplearning4j.spark.api.RDDTrainingApproach
import org.deeplearning4j.spark.impl.graph.SparkComputationGraph
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j

import scala.collection.mutable

object Utils {

  /**
   * This method computes the forward pass for a computation graph using RDD operations.
   */
  def getPredictions(
      input: RDD[DataSet],
      model: ComputationGraph,
      sc: SparkContext): RDD[DataSet] = {
    // the training parameters don't matter since we are just computing the forward pass
    val tm = new ParameterAveragingTrainingMaster.Builder(1)
      .build()
    val sparkModel = new SparkComputationGraph(sc, model, tm)
    val pairFunc = new PairFunction[DataSet, INDArray, INDArray] {
      override def call(t: DataSet): (INDArray, INDArray) = {
        (t.getLabels, t.getFeatureMatrix)
      }
    }
    sparkModel
      .feedForwardWithKeySingle(input.toJavaRDD().mapToPair(pairFunc), 1)
      .rdd.map { case (label, features) =>
      new DataSet(features, label)
    }
  }

  /**
   * This is a work-around to remove some layers of a model.
   */
  def removeLayer(
      model: ComputationGraph,
      outputLayer: String,
      removeLayers: Iterable[Layer]): ComputationGraph = {
    val builder = new TransferLearning.GraphBuilder(model)
      .setFeatureExtractor(outputLayer)
    // remove all the unfrozen layers, leaving just the un-trainable part of the model
    removeLayers.foreach { layer =>
      builder.removeVertexAndConnections(layer.conf().getLayer.getLayerName)
    }
    builder.setOutputs(outputLayer)
    builder.build()
  }

  def prettyPrintEvaluationStats(eval: Evaluation): String = {
    val result = new StringBuilder()
    result ++= s"Accuracy: ${eval.accuracy()}\n"
    result ++= s"Precision: ${eval.precision()}\n"
    result ++= s"Recall: ${eval.recall()}\n"
    result.result()
  }

  /**
   * SparkComputationGraph does not seem to have an evaluate method, this is to fill in that gap.
   */
  def evaluate(model: ComputationGraph, data: RDD[DataSet], evalBatchSize: Int): Evaluation = {
    val bcGraph = BroadcastComputationGraph.fromGraph(model)
    data.mapPartitions { it =>
      val eval = new Evaluation()
      val graph = bcGraph.toGraph
      val collect = new mutable.ArrayBuffer[DataSet]()
      var totalCount = 0
      while (it.hasNext) {
        collect.clear()
        var nExamples = 0
        while (it.hasNext && nExamples < evalBatchSize) {
          val next = it.next()
          nExamples += next.numExamples()
          collect += next
        }
        totalCount += nExamples
        val data = DataSet.merge(collect.toList.asJava)
        val output = graph.output(data.getFeatureMatrix())
        eval.eval(data.getLabels(), output.head)
      }
      Iterator.single(eval)
    }.reduce { case (e1, e2) =>
      e1.merge(e2)
      e1
    }
  }

  /**
   * Load the serialized byte arrays as [[org.apache.spark.ml.feature.LabeledPoint]]s.
   */
  def loadMllib(spark: SparkSession, path: String): DataFrame = {
    val df = spark.read.parquet(path)
    val toLabel = (l: Array[Byte]) => {
      val lArray = Nd4j.fromByteArray(l)
      val labelData = lArray.data().asFloat()
      var j = 0
      while (labelData(j) == 0F) { j += 1}
      j.toDouble
    }
    val toFeatures = (f: Array[Byte]) => {
      val fArray = Nd4j.fromByteArray(f)
      val features = Vectors.dense(fArray.data().asFloat().map(_.toDouble))
      features
    }
    val featuresUDF = udf(toFeatures)
    val labelUDF = udf(toLabel)
    df.select(labelUDF(col("_2")).as("label"), featuresUDF(col("_1")).as("features"))
  }

}
