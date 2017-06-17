package com.cloudera.datascience.dl4j.cnn.examples.mnist

import java.io.{DataInputStream, FileInputStream}
import java.nio.file.{Path, Paths}

import org.apache.spark.sql.SparkSession
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.{ConvolutionLayer, DenseLayer, OutputLayer, SubsamplingLayer}
import org.deeplearning4j.nn.conf.{LearningRatePolicy, MultiLayerConfiguration, NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.spark.api.RDDTrainingApproach
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions

import scala.collection.JavaConverters._
import scala.collection.mutable

/**
  * Simple example demonstrating how to train a neural network from scratch on the classic MNIST
  * dataset.
  */
object SparkMNISTExample {

  private val rate = 0.05
  private val seed = 42
  private val outputNum = 10
  private val batchSizePerWorker = 32
  private val numRows = 28
  private val numCols = 28
  private val numEpochs = 3
  private val trainingIterations = 5
  private val numChannels = 1
  private val testFeaturesPath = "mnist/t10k-images-idx3-ubyte"
  private val testLabelsPath = "mnist/t10k-labels-idx1-ubyte"
  private val trainFeaturesPath = "mnist/train-images-idx3-ubyte"
  private val trainLabelsPath = "mnist/train-labels-idx1-ubyte"

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder
      .appName("DL4J and Spark on MNIST")
      .master("local[*]")
      .getOrCreate()
    try {
      val sc = spark.sparkContext

      val trainImgs = load(Paths.get(trainFeaturesPath), Paths.get(trainLabelsPath))
      val testImgs = load(Paths.get(testFeaturesPath), Paths.get(testLabelsPath))
      val trainRDD = sc.parallelize(trainImgs)
      val testRDD = sc.parallelize(testImgs)

      val nnConf = feedforward
      val tm = new ParameterAveragingTrainingMaster.Builder(1)
        .averagingFrequency(10)
        .workerPrefetchNumBatches(2)
        .batchSizePerWorker(batchSizePerWorker)
        .rddTrainingApproach(RDDTrainingApproach.Export)
        .build()
      val model = new SparkDl4jMultiLayer(sc, nnConf, tm)
      model.setListeners(new ScoreIterationListener(5))
      (0 until numEpochs).foreach { _ =>
        model.fit(trainRDD)
      }

      val evaluation = model.evaluate(testRDD)

      val stats = evaluation.stats()
      println(stats)
    } finally {
      spark.stop()
    }
  }

  /**
    * Load MNIST train or test set, given the provided paths for images and labels.
    */
  def load(
      featureFile: Path,
      labelFile: Path,
      numTrainExamples: Option[Int] = None): Array[DataSet] = {
    val featureDataStream = new DataInputStream(new FileInputStream(featureFile.toString))
    val labelDataStream = new DataInputStream(new FileInputStream(labelFile.toString))

    assert(featureDataStream.readInt() == 2051)
    assert(labelDataStream.readInt() == 2049)
    val numExamples = featureDataStream.readInt()
    val labelExamples = labelDataStream.readInt()
    assert(numExamples == labelExamples)

    val n = numTrainExamples.getOrElse(numExamples)
    val data = new Array[DataSet](n)
    var i = 0
    while (i < n) {
      val arr = new Array[Float](numRows * numCols)
      var j = 0
      while (j < numRows * numCols) {
        arr(j) = featureDataStream.readUnsignedByte() / 255F
        j += 1
      }
      val features = Nd4j.create(arr)
      val label = Nd4j.zeros(10)
      label.putScalar(labelDataStream.readUnsignedByte(), 1F)
      data(i) = new DataSet(features, label)
      i += 1
    }
    featureDataStream.close()
    labelDataStream.close()
    data
  }

  def feedforward: MultiLayerConfiguration = {
    new NeuralNetConfiguration.Builder()
      .seed(seed)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .iterations(1)
      .activation(Activation.RELU)
      .weightInit(WeightInit.RELU)
      .learningRate(rate)
      .updater(Updater.NESTEROVS).momentum(0.98)
      .regularization(true)
      .l2(rate * 0.005)
      .list()
      .layer(0, new DenseLayer.Builder()
        .nIn(numRows * numCols)
        .weightInit(WeightInit.UNIFORM)
        .nOut(500)
        .build())
      .layer(1, new DenseLayer.Builder()
        .nIn(500)
        .nOut(100)
        .build())
      .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
        .activation(Activation.SOFTMAX)
        .nIn(100)
        .nOut(outputNum)
        .build())
      .pretrain(false)
      .backprop(true)
      .build()
  }

  def lenet: MultiLayerConfiguration = {
    val learningRateSchedule = new mutable.HashMap[java.lang.Integer, java.lang.Double]()
    learningRateSchedule.put(0, 0.01)
    learningRateSchedule.put(1000, 0.005)
    new NeuralNetConfiguration.Builder()
      .seed(seed)
      .iterations(trainingIterations) // Training iterations as above
      .regularization(true)
      .learningRate(.01)
      .learningRateDecayPolicy(LearningRatePolicy.Schedule)
      .learningRateSchedule(learningRateSchedule.asJava)
      .weightInit(WeightInit.XAVIER)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .updater(Updater.NESTEROVS).momentum(0.9)
      .list()
      .layer(0, new ConvolutionLayer.Builder(5, 5)
        // nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
        .nIn(numChannels)
        .stride(1, 1)
        .nOut(20)
        .activation(Activation.IDENTITY)
        .build())
      .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
        .kernelSize(2, 2)
        .stride(2, 2)
        .build())
      .layer(2, new ConvolutionLayer.Builder(5, 5)
        // Note that nIn need not be specified in later layers
        .stride(1, 1)
        .nOut(50)
        .activation(Activation.IDENTITY)
        .build())
      .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
        .kernelSize(2, 2)
        .stride(2, 2)
        .build())
      .layer(4, new DenseLayer.Builder().activation(Activation.RELU)
        .nOut(500).build())
      .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
        .nOut(outputNum)
        .activation(Activation.SOFTMAX)
        .build())
      .setInputType(InputType.convolutionalFlat(28,28,1))
      .backprop(true)
      .pretrain(false)
      .build()
  }
}

