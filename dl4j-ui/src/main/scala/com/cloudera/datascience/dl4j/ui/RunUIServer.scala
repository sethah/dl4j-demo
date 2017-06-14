package com.cloudera.datascience.dl4j.ui

import org.deeplearning4j.ui.play.PlayUIServer
import scopt.OptionParser

object RunUIServer {
  private[this] case class Params(port: String = "9000")

  private[this] object Params {
    def parseArgs(args: Array[String]): Params = {
      val params = new OptionParser[Params]("run the dl4j ui server") {
        opt[String]('p', "port")
          .text("port number for ui server")
          .action((x, c) => c.copy(port = x))
      }.parse(args, Params()).get
      params
    }
  }
  def main(args: Array[String]): Unit = {
    val params = Params.parseArgs(args)
    val ui = new PlayUIServer()
    ui.runMain(Array("--uiPort", params.port))
    ui.enableRemoteListener()
  }
}