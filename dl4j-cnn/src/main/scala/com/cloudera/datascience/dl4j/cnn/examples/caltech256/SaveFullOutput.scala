package com.cloudera.datascience.dl4j.cnn.examples.caltech256

import java.io.File

import com.cloudera.datascience.dl4j.cnn.Utils
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.transferlearning.{FineTuneConfiguration, TransferLearning, TransferLearningHelper}
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import scopt.OptionParser

/**
 * This program takes the featurized output from the first dense layer of the VGG16
 * model and further featurizes it through one or more of the last two dense layers of VGG16.
 */
object SaveFullOutput {

  private[this] case class Params(
    dataPath: String = null,
    savePath: String = null,
    modelPath: String = null,
    lastLayer: String = null)

  private[this] object Params {
    def parseArgs(args: Array[String]): Params = {
      val params = new OptionParser[Params]("train an existing model") {
        opt[String]("data")
          .text("the path of data")
          .action((x, c) => c.copy(dataPath = x))
        opt[String]("save")
          .text("save the output at this location")
          .action((x, c) => c.copy(savePath = x))
        opt[String]("model")
          .text("the path for the full model")
          .action((x, c) => c.copy(modelPath = x))
        opt[String]("lastLayer")
          .text("the last layer in the desired model")
          .action((x, c) => c.copy(lastLayer = x))
      }.parse(args, Params()).get
      require(params.dataPath != null && params.savePath != null && params.modelPath != null
        && params.lastLayer != null,
        "You must supply the data path and a path to save the output and the model path and" +
          "the last layer in the model")
      params
    }
  }

  def main(args: Array[String]): Unit = {
    val param = Params.parseArgs(args)
    val sparkConf = new SparkConf().setAppName("Featurize VGG dense layers")
    val sc = new SparkContext(sparkConf)
    val spark = SparkSession.builder().getOrCreate()
    import spark.sqlContext.implicits._

    val modelFile = new File(param.modelPath)
    val vgg16 = ModelSerializer.restoreComputationGraph(modelFile)
    val data = spark.read.parquet(param.dataPath)
      .rdd
      .map { case Row(f: Array[Byte], l: Array[Byte]) =>
        new DataSet(Nd4j.fromByteArray(f), Nd4j.fromByteArray(l))
      }
    val model = param.lastLayer match {
      case "predictions" => convToPredictions(vgg16)
      case "fc2" => convToFC2(vgg16)
    }

    val finalOutput = Utils.getPredictions(data, model, sc)
    val df = finalOutput.map { ds =>
      (Nd4j.toByteArray(ds.getFeatureMatrix), Nd4j.toByteArray(ds.getLabels))
    }.toDF()
    df.write.parquet(param.savePath)
  }

  private def convToFC2(vgg: ComputationGraph): ComputationGraph = {
    val helper = new TransferLearningHelper(vgg, "fc1")
    val unfrozen = helper.unfrozenGraph()

    // cut off the predictions layer
    val fineTuneConf = new FineTuneConfiguration.Builder()
      .build()
    new TransferLearning.GraphBuilder(unfrozen)
      .fineTuneConfiguration(fineTuneConf)
      .removeVertexAndConnections("predictions")
      .setOutputs("fc2")
      .build()
  }

  private def convToPredictions(vgg: ComputationGraph): ComputationGraph = {
    val helper = new TransferLearningHelper(vgg, "fc2")
    helper.unfrozenGraph()
  }
}
