package com.cloudera.datascience.dl4j.cnn.spark

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration
import org.deeplearning4j.nn.graph.ComputationGraph
import org.nd4j.linalg.api.ndarray.INDArray

/**
 * This class is used to extract only the parts of a computation graph that are required to
 * reconstruct it. This is useful because we may want to ship some networks to use on RDDs, but
 * if the network has already been initialized then it will allocate a lot of extra space that is
 * only needed during training and does not need to be shipped. So, instead of:
 *    data.map { x => graph.output(x) }
 * We can do:
 *    val bcGraph = sc.broadcast(BroadcastComputationGraph.fromGraph(graph))
 *    data.mapPartitions { it =>
 *      val graph = bc.value.toGraph
 *      it.map { x => graph.output(x) }
 *    }
 */
case class BroadcastComputationGraph(jsonConfig: String, params: INDArray) {
  def toGraph: ComputationGraph = {
    val graph = new ComputationGraph(ComputationGraphConfiguration.fromJson(jsonConfig).clone())
    graph.init(params.unsafeDuplication(), false)
    graph
  }
}
object BroadcastComputationGraph {
  def fromGraph(graph: ComputationGraph): BroadcastComputationGraph = {
    BroadcastComputationGraph(graph.getConfiguration.toJson, graph.params())
  }
}
