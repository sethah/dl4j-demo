package com.cloudera.datascience.dl4j.cnn.examples.caltech256

import java.io.File

import com.cloudera.datascience.dl4j.cnn.Utils
import org.apache.spark.sql.{Row, SparkSession}
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import scopt.OptionParser

object ScoreSparkModel {

  private[this] case class Params (modelPath: String = null, dataPath: String = null)

  private[this] object Params {
    def parseArgs(args: Array[String]): Params = {
      val params = new OptionParser[Params]("train an existing model") {
        opt[String]("test")
          .text("the path of the test data")
          .action((x, c) => c.copy(dataPath = x))
        opt[String]("model")
          .text("location of the model")
          .action((x, c) => c.copy(modelPath = x))
      }.parse(args, Params()).get
      require(params.modelPath != null && params.dataPath != null,
        "You must supply the data path and a model path.")
      params
    }
  }

  def main(args: Array[String]): Unit = {
    val param = Params.parseArgs(args)
    val spark = SparkSession.builder().appName("score a model").getOrCreate()
    try {
      val sc = spark.sparkContext
      sc.hadoopConfiguration.set("mapreduce.input.fileinputformat.input.dir.recursive", "true")
      val restorePath = new File(param.modelPath)
      val restored = ModelSerializer.restoreComputationGraph(restorePath)
      val testRDD = spark.read.parquet(param.dataPath)
        .rdd
        .map { case Row(f: Array[Byte], l: Array[Byte]) =>
          new DataSet(Nd4j.fromByteArray(f), Nd4j.fromByteArray(l))
        }
      val eval = Utils.evaluate(restored, testRDD, 16)
      println(Utils.prettyPrintEvaluationStats(eval))

    } finally {
      spark.stop()
    }
  }
}
