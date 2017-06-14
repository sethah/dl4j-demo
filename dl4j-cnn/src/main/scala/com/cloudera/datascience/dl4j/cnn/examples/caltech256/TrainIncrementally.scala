package com.cloudera.datascience.dl4j.cnn.examples.caltech256

import java.io.{File, PrintWriter}

import scala.collection.JavaConverters._
import com.cloudera.datascience.dl4j.cnn.Utils
import org.apache.log4j.FileAppender
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext}
import org.deeplearning4j.api.storage.impl.RemoteUIStatsStorageRouter
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.{LearningRatePolicy, Updater}
import org.deeplearning4j.nn.transferlearning.{FineTuneConfiguration, TransferLearning}
import org.deeplearning4j.spark.api.RDDTrainingApproach
import org.deeplearning4j.spark.impl.graph.SparkComputationGraph
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster
import org.deeplearning4j.ui.stats.StatsListener
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import scopt.OptionParser

/**
 * This program is intended to train models incrementally, i.e. start from a saved model state, run
 * a specified number of epochs, and save a new model.
 */
object TrainIncrementally {
  private[this] case class Params(
    dataPath: String = null,
    validPath: Option[String] = None,
    modelPath: String = null,
    savePath: String = null,
    ui: Option[String] = None,
    numEpochs: Int = 1,
    batchSizePerWorker: Int = 8,
    averagingFrequency: Int = 5,
    learningRate: Double = 0.001,
    learningRateSchedule: Option[String] = None,
    momentum: Double = 0.9,
    regularization: Double = 0.0,
    dropout: Double = 0.0,
    validationInterval: Int = 4,
    seed: Long = 42L,
    updater: Updater = Updater.NESTEROVS)

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
          .action((x, c) => c.copy(savePath = x))
        opt[String]("model")
          .text("restore the model from location")
          .action((x, c) => c.copy(modelPath = x))
        opt[String]("learningRateSchedule")
          .text("learning rate schedule in form of iter1,rate1|iter2,rate2|iter3,rate3")
          .action((x, c) => c.copy(learningRateSchedule = Some(x)))
        opt[String]("ui")
          .text("remote ui address, e.g. http://10.15.36.20:9007")
          .action((x, c) => c.copy(ui = Some(x)))
        opt[Int]("epochs")
          .text("number of training epochs")
          .action((x, c) => c.copy(numEpochs = x))
        opt[Int]("validationInterval")
          .text("how often to evaluate on the validation set")
          .action((x, c) => c.copy(validationInterval = x))
        opt[Double]("rate")
          .text("learning rate")
          .action((x, c) => c.copy(learningRate = x))
        opt[Int]("batchSize")
          .text("batch size per worker")
          .action((x, c) => c.copy(batchSizePerWorker = x))
        opt[Int]("averagingFrequency")
          .text("frequency for parameter averaging")
          .action((x, c) => c.copy(averagingFrequency = x))
        opt[Double]("momentum")
          .text("sgd momentum")
          .action((x, c) => c.copy(momentum = x))
        opt[Double]("reg")
          .text("regularization")
          .action((x, c) => c.copy(regularization = x))
        opt[Double]("dropout")
          .text("dropout")
          .action((x, c) => c.copy(dropout = x))
        opt[Long]("seed")
          .text("random seed")
          .action((x, c) => c.copy(seed = x))
        opt[String]("updater")
          .text("sgd updater")
          .action((x, c) => {
            val updater = x.toLowerCase() match {
              case "nesterovs" => Updater.NESTEROVS
              case "adam" => Updater.ADAM
              case "adadelta" => Updater.ADADELTA
              case "rmsprop" => Updater.RMSPROP
              case o => throw new IllegalArgumentException(s"Updater type $o not supported.")
            }
            c.copy(updater = updater)
          })
      }.parse(args, Params()).get
      require(params.dataPath != null && params.savePath != null && params.modelPath != null,
        "You must supply the data path and a path to save the model.")
      params
    }
  }

  def main(args: Array[String]): Unit = {
    val sparkConf = new SparkConf().setAppName("Train a saved model")
    val sc = new SparkContext(sparkConf)

    // quiet all the logs
    org.apache.log4j.Logger.getRootLogger.setLevel(org.apache.log4j.Level.ERROR)
    val logger4j = org.apache.log4j.LogManager.getLogger(this.getClass)
    logger4j.setLevel(org.apache.log4j.Level.INFO)
    val param = Params.parseArgs(args)
    try {
      val rng = new scala.util.Random()
      val spark = SparkSession.builder().getOrCreate()
      val restorePath = new File(param.modelPath)
      val restored = ModelSerializer.restoreComputationGraph(restorePath)
      val trainRDD = spark.read.parquet(param.dataPath)
        .rdd
        .map(x => (rng.nextDouble, x))
        .sortByKey()
        .values
        .map { case Row(f: Array[Byte], l: Array[Byte]) =>
          new DataSet(Nd4j.fromByteArray(f), Nd4j.fromByteArray(l))
        }
      val validRDD = param.validPath.map { path =>
        spark.read.parquet(path)
          .rdd
          .map { case Row(f: Array[Byte], l: Array[Byte]) =>
            new DataSet(Nd4j.fromByteArray(f), Nd4j.fromByteArray(l))
          }
      }
      trainRDD.persist(StorageLevel.MEMORY_AND_DISK_SER)

      val tm = new ParameterAveragingTrainingMaster.Builder(1)
        .averagingFrequency(param.averagingFrequency)
        .workerPrefetchNumBatches(2)
        .batchSizePerWorker(param.batchSizePerWorker)
        .rddTrainingApproach(RDDTrainingApproach.Export)
        .build()

      val lrSchedule = param.learningRateSchedule.map { sched =>
        val sch = new scala.collection.mutable.HashMap[java.lang.Integer, java.lang.Double]()
        sched.split("\\|").foreach { pair =>
          val Array(iter, rate) = pair.split(",")
          require(iter.toInt >= 0 && rate.toDouble > 0)
          sch.put(iter.toInt, rate.toDouble)
        }
        sch
      }

      val fineTuneConfBuilder = new FineTuneConfiguration.Builder()
        .learningRate(param.learningRate)
        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
        .updater(param.updater)
        .momentum(param.momentum)
        .regularization(true)
        .l2(param.regularization)
        .seed(param.seed)
      lrSchedule.foreach { sched =>
        fineTuneConfBuilder.learningRatePolicy(LearningRatePolicy.Schedule)
        fineTuneConfBuilder.learningRateSchedule(sched.asJava)
      }
      val fineTuneConf = fineTuneConfBuilder.build()

      val restoredNetwork = new TransferLearning.GraphBuilder(restored)
        .fineTuneConfiguration(fineTuneConf)
        .build()
      logger4j.info(s"Starting training for ${param.numEpochs} epochs")
      logger4j.info(s"Using model: \n ${restoredNetwork.summary()}")
      logger4j.info(s"Params: \n ${param.toString}")
      logger4j.info(s"Model configuration:\n ${restoredNetwork.getConfiguration.toString}")

      val model = new SparkComputationGraph(sc, restoredNetwork, tm)

      param.ui.foreach { address =>
        val remoteUIRouter = new RemoteUIStatsStorageRouter(address)
        model.setListeners(remoteUIRouter, List(new StatsListener(null)).asJava)
      }

      (0 until param.numEpochs).foreach { i =>
        logger4j.info(s"epoch $i starting")
        model.fit(trainRDD)
        if ((i + 1) % param.validationInterval == 0) {
          val trainEval = Utils.evaluate(model.getNetwork, trainRDD, 16)
          logger4j.info(s"Train score: ${model.calculateScore(trainRDD, true)}")
          logger4j.info(s"train stats:\n${Utils.prettyPrintEvaluationStats(trainEval)}")
          validRDD.foreach { validData =>
            val eval = Utils.evaluate(model.getNetwork, validData, 16)
            logger4j.info(s"Validation score: ${model.calculateScore(validData, true)}")
            logger4j.info(s"validation stats:\n${Utils.prettyPrintEvaluationStats(eval)}")
          }
        }
      }
      val locationToSave = new File(param.savePath)
      ModelSerializer.writeModel(model.getNetwork, locationToSave, true)
    } finally {
      sc.stop()
    }
  }

}

