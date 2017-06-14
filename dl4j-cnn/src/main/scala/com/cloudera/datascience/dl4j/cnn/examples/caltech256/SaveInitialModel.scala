package com.cloudera.datascience.dl4j.cnn.examples.caltech256

import java.io.File

import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.{BatchNormalization, DenseLayer, DropoutLayer, OutputLayer}
import org.deeplearning4j.nn.conf.{NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import scopt.OptionParser

object SaveInitialModel {
  private[this] case class Params(savePath: String = null, predictAfter: String = null)

  private[this] object Params {
    def parseArgs(args: Array[String]): Params = {
      val params = new OptionParser[Params]("train an existing model") {
        opt[String]("save")
          .text("save the model at this location")
          .action((x, c) => c.copy(savePath = x))
        opt[String]("predictAfter")
          .text("save a model that will predict on the output of this layer")
          .action((x, c) => c.copy(predictAfter = x))
      }.parse(args, Params()).get
      require(params.savePath != null && params.predictAfter != null,
        "You must supply a path to save model and the last layer.")
      params
    }
  }

  def main(args: Array[String]): Unit = {
    val param = Params.parseArgs(args)
    val locationToSave = new File(param.savePath)
    val model = getModel(param.predictAfter)
    model.init()
    ModelSerializer.writeModel(model, locationToSave, true)
  }

  /**
   * This method returns a simple fully connected layer with appropriate shape, depending on the
   * which layer in the VGG model this model will be stacked on top of.
   * @param predictAfterLayer Stack the new model after this layer in the original VGG network.
   * @return
   */
  def getModel(predictAfterLayer: String): ComputationGraph = {
    val conf = predictAfterLayer match {
      case "predictions" =>
        new NeuralNetConfiguration.Builder()
          .seed(42)
          .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
          .iterations(1)
          .activation(Activation.SOFTMAX)
          .weightInit(WeightInit.XAVIER)
          .learningRate(0.001)
          .updater(Updater.NESTEROVS)
          .momentum(0.98)
          .regularization(true)
          .l2(0.00005)
          .graphBuilder()
          .addInputs("in")
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
          .backprop(true)
          .pretrain(false)
          .build()
      case "fc2" =>
        new NeuralNetConfiguration.Builder()
          .seed(42)
          .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
          .iterations(1)
          .activation(Activation.SOFTMAX)
          .weightInit(WeightInit.XAVIER)
          .learningRate(0.001)
          .updater(Updater.NESTEROVS)
          .momentum(0.98)
          .regularization(true)
          .l2(0.00005)
          .graphBuilder()
          .addInputs("in")
          .addLayer("layer0",
            new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
              .activation(Activation.SOFTMAX)
              .nIn(4096)
              .nOut(257)
              .build(),
            "in")
          .setOutputs("layer0")
          .backprop(true)
          .pretrain(false)
          .build()
      case "fc1" =>
        new NeuralNetConfiguration.Builder()
          .seed(42)
          .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
          .iterations(1)
          .activation(Activation.SOFTMAX)
          .weightInit(WeightInit.XAVIER)
          .learningRate(0.01)
          .updater(Updater.NESTEROVS)
          .momentum(0.9)
          .regularization(true)
          .l2(0.01)
          .dropOut(0.9)
          .graphBuilder()
          .addInputs("in")
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
          .build()
      case _ =>
        throw new IllegalArgumentException(s"layer $predictAfterLayer not supported right now")
    }
    new ComputationGraph(conf)
  }
}
