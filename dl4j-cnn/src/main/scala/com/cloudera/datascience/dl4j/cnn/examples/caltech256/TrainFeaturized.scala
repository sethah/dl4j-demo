package com.cloudera.datascience.dl4j.cnn.examples.caltech256

import java.io.File

import scala.collection.JavaConverters._
import scala.util.Random
import com.cloudera.datascience.dl4j.cnn.Utils
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.storage.StorageLevel
import org.deeplearning4j.api.storage.impl.RemoteUIStatsStorageRouter
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.{BatchNormalization, DenseLayer, DropoutLayer, OutputLayer}
import org.deeplearning4j.nn.conf.{ComputationGraphConfiguration, NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.spark.api.RDDTrainingApproach
import org.deeplearning4j.spark.impl.graph.SparkComputationGraph
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster
import org.deeplearning4j.ui.stats.StatsListener
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import scopt.OptionParser

/**
 * Trains a single layer dense neural network on input data that has been featurized by the
 * VGG16 convolutional neural network. Allows for different architectures depending on the layers
 * used to featurize the data. Currently supports:
 *   last vgg layer -> model shape -> output dim
 *   fc2 (4096) -> (4096, 257) -> 257
 *   predictions (1000) -> (1000, 257) -> 257
 */
object TrainFeaturized {

  private[this] case class Params(
      dataPath: String = null,
      validPath: Option[String] = None,
      inputLayer: String = null,
      savePath: Option[String] = None,
      ui: Option[String] = None,
      seed: Long = 42,
      numEpochs: Int = 1,
      batchSizePerWorker: Int = 8,
      averagingFrequency: Int = 5,
      learningRate: Double = 0.001,
      momentum: Double = 0.9,
      regularization: Double = 0.0,
      dropout: Double = 0.0,
      validationInterval: Int = 4,
      updater: Updater = Updater.NESTEROVS) {
  }

  private[this] object Params {
    def parseArgs(args: Array[String]): Params = {
      val params = new OptionParser[Params]("train an existing model") {
        opt[String]("train")
          .text("the path of the training data")
          .action((x, c) => c.copy(dataPath = x))
        opt[String]("valid")
          .text("the path of the validation data")
          .action((x, c) => c.copy(validPath = Some(x)))
        opt[String]("save")
          .text("save the model at this location")
          .action((x, c) => c.copy(savePath = Some(x)))
        opt[String]("inputLayer")
          .text("the output of this layer is used as input to the trained model")
          .action((x, c) => c.copy(inputLayer = x))
        opt[Long]("seed")
          .text("random seed for training")
          .action((x, c) => c.copy(seed = x))
        opt[Int]("epochs")
          .text("number of training epochs")
          .action((x, c) => c.copy(numEpochs = x))
        opt[Int]("validationInterval")
          .text("how often to evaluate on the validation set")
          .action((x, c) => c.copy(validationInterval = x))
        opt[String]("ui")
          .text("remote ui address")
          .action((x, c) => c.copy(ui = Some(x)))
        opt[Double]("rate")
          .text("learning rate")
          .action((x, c) => c.copy(learningRate = x))
        opt[Int]("batchSize")
          .text("batch size per worker")
          .action((x, c) => c.copy(batchSizePerWorker = x))
        opt[Int]("averagingFrequency")
          .text("frequency for paramter averaging")
          .action((x, c) => c.copy(averagingFrequency = x))
        opt[Double]("momentum")
          .text("sgd momentum")
          .action((x, c) => c.copy(momentum = x))
        opt[Double]("reg")
          .text("regularization")
          .action((x, c) => c.copy(regularization = x))
        opt[Double]("dropout")
          .text("regularization")
          .action((x, c) => c.copy(dropout = x))
        opt[String]("updater")
          .text("sgd updater")
          .action((x, c) => {
            val updater = x.toLowerCase() match {
              case "nesterovs" => Updater.NESTEROVS
              case "adam" => Updater.ADAM
              case "adadelta" => Updater.ADADELTA
              case "rmsprop" => Updater.RMSPROP
              case u => throw new IllegalArgumentException(s"Invalid updater: $u")
            }
            c.copy(updater = updater)
          })
      }.parse(args, Params()).get
      require(params.dataPath != null && params.inputLayer != null,
        "You must supply the training data path and the input layer.")
      params
    }
  }

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().appName("Train a saved model").getOrCreate()
    try {
      val sc = spark.sparkContext
      val rng = new Random()
      org.apache.log4j.Logger.getRootLogger.setLevel(org.apache.log4j.Level.ERROR)
      val logger4j = org.apache.log4j.LogManager.getLogger(this.getClass)
      logger4j.setLevel(org.apache.log4j.Level.INFO)
      val params = Params.parseArgs(args)

      val confBuilder = new NeuralNetConfiguration.Builder()
        .seed(params.seed)
        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
        .iterations(1)
        .activation(Activation.SOFTMAX)
        .weightInit(WeightInit.XAVIER)
        .learningRate(params.learningRate)
        .updater(params.updater)
        .momentum(params.momentum)
        .regularization(true)
        .l2(params.regularization)
        .dropOut(params.dropout)
        .graphBuilder()
      val graph = getModel(params.inputLayer, confBuilder)

      // need to randomize the data to make sure minibatches aren't uniformly labeled
      val trainRDD = spark.read.parquet(params.dataPath)
        .rdd
        .map(x => (rng.nextDouble, x))
        .sortByKey()
        .values
        .map { case Row(f: Array[Byte], l: Array[Byte]) =>
          new DataSet(Nd4j.fromByteArray(f), Nd4j.fromByteArray(l))
        }
      val validRDD = params.validPath.map { path =>
        spark.read.parquet(path)
          .rdd
          .map { case Row(f: Array[Byte], l: Array[Byte]) =>
            new DataSet(Nd4j.fromByteArray(f), Nd4j.fromByteArray(l))
          }
      }
      trainRDD.persist(StorageLevel.MEMORY_AND_DISK_SER)

      val tm = new ParameterAveragingTrainingMaster.Builder(1)
        .averagingFrequency(params.averagingFrequency)
        .workerPrefetchNumBatches(2)
        .batchSizePerWorker(params.batchSizePerWorker)
        .rddTrainingApproach(RDDTrainingApproach.Export)
        .build()

      val model = new SparkComputationGraph(sc, graph, tm)

      params.ui.foreach { address =>
        val remoteUIRouter = new RemoteUIStatsStorageRouter(address)
        model.setListeners(remoteUIRouter, List(new StatsListener(null)).asJava)
      }
      logger4j.info(graph.summary())
      (0 until params.numEpochs).foreach { i =>
        logger4j.info(s"epoch $i starting")
        model.fit(trainRDD)
        if ((i + 1) % params.validationInterval == 0) {
          logger4j.info(s"Train score: ${model.calculateScore(trainRDD, true)}")
          val trainEval = Utils.evaluate(model.getNetwork, trainRDD, 16)
          logger4j.info(s"Train stats:\n${Utils.prettyPrintEvaluationStats(trainEval)}")
          validRDD.foreach { validData =>
            val eval = Utils.evaluate(model.getNetwork, validData, 16)
            logger4j.info(s"Validation stats:\n${Utils.prettyPrintEvaluationStats(eval)}")
            logger4j.info(s"Validation score: ${model.calculateScore(validData, true)}")
          }
        }
      }

      params.savePath.foreach { path =>
        val locationToSave = new File(path)
        ModelSerializer.writeModel(model.getNetwork, locationToSave, true)
      }

    } finally {
      spark.stop()
    }
}

  /**
   * This method returns a simple fully connected layer with appropriate shape, depending on the
   * which layer in the VGG model this model will be stacked on top of.
   * @param inputLayer Stack the new model after this layer in the original VGG
   * @param confBuilder A graph builder that has the base configuration already set.
   */
  def getModel(
      inputLayer: String,
      confBuilder: ComputationGraphConfiguration.GraphBuilder): ComputationGraph = {
    inputLayer.toLowerCase() match {
      case "predictions" =>
        confBuilder
          .addLayer("batchNorm0",
            new BatchNormalization.Builder()
              .nIn(1000)
              .nOut(1000)
              .build(),
            "in")
          .addLayer("layer0",
            new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
              .activation(Activation.SOFTMAX)
              .nIn(1000)
              .nOut(257)
              .build(),
            "batchNorm0")
          .setOutputs("layer0")
      case "fc2" =>
        confBuilder
          .addLayer("dropout0",
            new DropoutLayer.Builder().build(), "in")
          .addLayer("batchNorm0",
            new BatchNormalization.Builder()
              .nIn(4096)
              .nOut(4096)
              .build(),
            "dropout0")
          .addLayer("predictions",
            new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
              .activation(Activation.SOFTMAX)
              .nIn(4096)
              .nOut(257)
              .build(),
            "batchNorm0")
          .setOutputs("predictions")
      case "fc1" =>
        confBuilder
          .addLayer("dropout0",
            new DropoutLayer.Builder().build(), "in")
          .addLayer("batchNorm0",
            new BatchNormalization.Builder()
              .nIn(4096)
              .nOut(4096)
              .build(),
            "dropout0")
          .addLayer("fc2",
            new DenseLayer.Builder()
              .activation(Activation.RELU)
              .nIn(4096)
              .nOut(4096)
              .build(),
            "batchNorm0")
          .addLayer("dropout1",
            new DropoutLayer.Builder().build(), "fc2")
          .addLayer("batchNorm1",
            new BatchNormalization.Builder()
              .nIn(4096)
              .nOut(4096)
              .build(),
            "dropout1")
          .addLayer("predictions",
            new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
              .activation(Activation.SOFTMAX)
              .nIn(4096)
              .nOut(257)
              .build(),
            "batchNorm1")
          .setOutputs("predictions")
      case _ =>
        throw new IllegalArgumentException(s"layer $inputLayer not supported right now")
    }
    val conf = confBuilder
      .addInputs("in")
      .backprop(true)
      .pretrain(false)
      .build()
    new ComputationGraph(conf)
  }
}
