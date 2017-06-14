package com.cloudera.datascience.dl4j.cnn.examples.caltech256

import java.io.File

import com.cloudera.datascience.dl4j.cnn.Utils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}
import org.datavec.image.loader.NativeImageLoader
import org.deeplearning4j.nn.api.Layer
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.modelimport.keras.trainedmodels.{TrainedModelHelper, TrainedModels}
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import scopt.OptionParser

/**
 * This program is used to featurize a directory of images, obtaining intermediate output
 * from a specified layer of the VGG16 convolutional network.
 */
object SaveFeaturizedData {
  private val imageHeight = 224
  private val imageWidth = 224
  private val numChannels = 3

  private[this] case class Params(
    numClasses: Int = 257,
    outputLayer: String = null,
    imagePath: String = null,
    savePath: String = null,
    modelPath: Option[String] = None)

  private[this] object Params {
    def parseArgs(args: Array[String]): Params = {
      val params = new OptionParser[Params]("Load and save jpegs") {
        opt[Int]("numClasses")
          .text("number of class labels")
          .action((x, c) => c.copy(numClasses = x))
        opt[String]("outputLayer")
          .text("the layer that serves as the output layer for the chopped model")
          .action((x, c) => c.copy(outputLayer = x))
        opt[String]("imagePath")
          .text("the path of jpeg data")
          .action((x, c) => c.copy(imagePath = x))
        opt[String]("savePath")
          .text("the path to save the featurized or byte[] version of jpeg data")
          .action((x, c) => c.copy(savePath = x))
        opt[String]("modelPath")
          .text("the path for the imported model")
          .action((x, c) => c.copy(modelPath = Some(x)))
      }.parse(args, Params()).get
      require(params.imagePath != null && params.savePath != null && params.modelPath != null,
        "You must supply the image data path and the path to save output.")
      params
    }
  }

  def main(args: Array[String]): Unit = {
    val param = Params.parseArgs(args)
    val numClasses = param.numClasses
    val imagePath = param.imagePath
    val savePath = param.savePath
    val modelPath = param.modelPath

    val sparkConf = new SparkConf().setAppName("Save output of convolutional layers").setMaster("local[*]")
    val sc = new SparkContext(sparkConf)
    sc.hadoopConfiguration.set("mapreduce.input.fileinputformat.input.dir.recursive", "true")
    val logger = org.apache.log4j.LogManager.getLogger(this.getClass)
    try {
      val spark = SparkSession.builder().getOrCreate()
      import spark.sqlContext.implicits._
      val vgg16 = modelPath.map { path =>
        val modelFile = new File(path)
        ModelSerializer.restoreComputationGraph(modelFile)
      }.getOrElse {
        logger.info("No model provided, fetching VGG...")
        val helper = new TrainedModelHelper(TrainedModels.VGG16)
        helper.loadModel
      }

      /*
        Split the layers where `param.outputLayer` is the last layer in the left section.
        What this would mean is that the frozen graph would have the NN layers up until
        `param.outputLayer` and the unfrozen graph will contain layers from the layer after the
        output layer, to the final "predictions" layer.

        Note: you can review the layer info within a graph using `unfrozen.summary()`.
      */
      val (_, unfrozen) = splitModelAt(param.outputLayer, vgg16)
      val choppedGraph = Utils.removeLayer(vgg16, param.outputLayer, unfrozen)

      Seq("valid").foreach { dir =>
        val predictions = featurizeJpegs(sc, s"$imagePath$dir/", numClasses, choppedGraph)
        val df = predictions.map { ds =>
          (Nd4j.toByteArray(ds.getFeatureMatrix), Nd4j.toByteArray(ds.getLabels))
        }.toDF()
        df.write.parquet(s"$savePath$dir/")
      }
    } finally {
      sc.stop()
    }
  }

  /**
   * Split a ComputationGraph.
   * @param splitLayer The name of the last layer in the left portion.
   */
  def splitModelAt(splitLayer: String, model: ComputationGraph): (Array[Layer], Array[Layer]) = {
    val layers = model.getLayers
    layers.splitAt(layers.map(_.conf().getLayer.getLayerName).indexOf(splitLayer) + 1)
  }

  /**
   * Load individual .jpg images and featurize them through a neural network graph.
   *
   * @param sc Active Spark context.
   * @param imagePath Path to load input jpgs from.
   * @param numClasses Length of the one-hot encoded label vectors.
   * @param choppedGraph The network to get predictions from.
   * @return RDD of DataSet, one DataSet per input image.
   */
  def featurizeJpegs(
      sc: SparkContext,
      imagePath: String,
      numClasses: Int,
      choppedGraph: ComputationGraph): RDD[DataSet] = {
    val jpegs = sc.binaryFiles(imagePath).mapPartitions { it =>
      val loader = new NativeImageLoader(imageHeight, imageWidth, numChannels)
      it.map { case (path, img) =>
        val inputStream = img.open()
        val mat = loader.asMatrix(inputStream)
        inputStream.close()
        val regex = ".+\\/(\\d{3})_\\d{4}\\.jpg".r
        val label = path match {
          case regex(l) => l.toInt - 1
          case _ =>
            throw new IllegalArgumentException(s"Could not parse label from path: $path")
        }
        val labelArray = Nd4j.zeros(numClasses)
        labelArray.putScalar(label, 1F)
        new DataSet(mat, labelArray)
      }
    }
    Utils.getPredictions(jpegs, choppedGraph, sc)
  }
}

