package pl.akkomar.neuralnetworks

import breeze.linalg.DenseMatrix
import breeze.numerics._

/**
  * Neural network representation
  *
  * @param sizes list containing number of neurons in respective layers.
  *              For example, for network with 2 neurons in the first layer, 3 neurons in the second layer, and
  *              1 neuron in the final layer it'll be: List(2,3,1)
  */
class Network(sizes: Array[Int]) {
  val numberOfLayers: Int = sizes.length

  //skipping input layer as biases are only used in computing output from later layers
  val biases: Array[DenseMatrix[Double]] = sizes.tail.map { layerSize =>
    val normal01 = breeze.stats.distributions.Gaussian(0, 1)
    DenseMatrix.rand(layerSize, 1, normal01)
  }

  //w=weights(n) are weights connecting n+1 and n+2 layers of neurons
  //weights(n)(j,k) is weight for the connection between k-th neuron in the n+1 layer and j-th neuron in the n+2 layer
  val weights: Array[DenseMatrix[Double]] = sizes.dropRight(1).zip(sizes.tail).map { case (x, y) =>
    val normal01 = breeze.stats.distributions.Gaussian(0, 1)
    DenseMatrix.rand(x, y, normal01)
  }

  def feedForward(input: DenseMatrix[Double]): DenseMatrix[Double] = {
    var a = input
    biases.zip(weights).foreach { case (b, w) =>
      a = sigmoid((w * a) + b)
    }
    a
  }

  /**
    * Train network using mini-batch stochastic gradient descent
    */
  def sgd(): Unit = {

  }

  override def toString: String = {
    s"""number of layers: $numberOfLayers
       |biases: ${biases.mkString("[", ";\n", "]")}
       |weights: ${weights.mkString("[", ";\n", "]")}
     """.stripMargin
  }
}


object NetworkTest extends App {

  val network = new Network(Array(2, 3, 1))

  println(network)
}
