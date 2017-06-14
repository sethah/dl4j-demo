package com.cloudera.datascience.dl4j.cnn.examples.caltech256

import java.io.File

import org.deeplearning4j.nn.modelimport.keras.trainedmodels.{TrainedModelHelper, TrainedModels}
import org.deeplearning4j.util.ModelSerializer
import scopt.OptionParser

object ImportPreTrainedModel {
  private[this] case class Params(savePath: String = null)

  private[this] object Params {
    def parseArgs(args: Array[String]): Params = {
      val params = new OptionParser[Params]("Import pre-trained model") {
        opt[String]("save")
          .text("the path to save the imported model")
          .action((x, c) => c.copy(savePath = x))
      }.parse(args, Params()).get
      require(params.savePath != null,
        "You must supply the path to save the imported model.")
      params
    }
  }

  def main(args: Array[String]): Unit = {
    val modelImportHelper = new TrainedModelHelper(TrainedModels.VGG16)
    val vgg16 = modelImportHelper.loadModel()
    val param = Params.parseArgs(args)
    val locationToSave = new File(param.savePath)
    ModelSerializer.writeModel(vgg16, locationToSave, true)
  }
}