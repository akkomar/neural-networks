package pl.akkomar.neuralnetworks

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics._

import scala.collection.mutable

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

  def feedForward(input: DenseVector[Double]): DenseVector[Double] = {
    var a = input
    biases.zip(weights).foreach { case (b, w) =>
      a = sigmoid((w * a) + b)
    }
    a
  }

  /**
    * Returns a tuple (`nablaB`, `nablaW`) representing the gradient for the cost function.
    * `nablaB` and `nablaW` are layer-by-layer arrays of dense matrices similar to `biases` and `weights`
    * @param x
    * @param y
    * @return
    */
  private def backProp(x: DenseVector[Double], y: DenseVector[Double]):(Array[DenseMatrix[Double]],Array[DenseMatrix[Double]]) = {
    val nablaB: Array[DenseMatrix[Double]] = biases.map(b => DenseMatrix.zeros[Double](b.rows, b.cols))
    val nablaW: Array[DenseMatrix[Double]] = weights.map(w => DenseMatrix.zeros[Double](w.rows, w.cols))

    var activation = x

    //all activations, layer by layer
    val activations = new mutable.MutableList[DenseVector[Double]]()+=x

    //all z vectors, layer by layer
    val zs = new mutable.MutableList[DenseVector[Double]]()

    biases.zip(weights).foreach{case (b,w)=>
        val z:DenseVector[Double] = (w * activation)+b
        zs+=z
        activation = sigmoid(z)
        activations+=activation
    }

    //backward pass

    ???
  }

  /**
    * Updates the network weights and biases according to single iteration of gradient descent using backpropagation
    * to a single miniBatch of training data
    *
    * @param miniBatch
    * @param eta learning rate
    */
  private def updateMiniBatch(miniBatch: Array[TrainingExample], eta: Double): Unit = {
    val nablaB: Array[DenseMatrix[Double]] = biases.map(b => DenseMatrix.zeros[Double](b.rows, b.cols))
    val nablaW: Array[DenseMatrix[Double]] = weights.map(w => DenseMatrix.zeros[Double](w.rows, w.cols))
    miniBatch.foreach { case TrainingExample(x, y) =>
      val (deltaNablaB, deltaNablaW) = backProp(x, y)
      nablaB.indices.foreach{ n=>
        nablaB(n):+=deltaNablaB(n)
      }
      nablaW.indices.foreach{ n=>
        nablaW(n):+=deltaNablaW(n)
      }
    }
    //update weights
    weights.indices.foreach{n=>
      weights(n):-=(nablaW(n):*=(eta/miniBatch.length.toDouble))
    }
    //update biases
    biases.indices.foreach{n=>
      biases(n):-=(nablaB(n):*=(eta/miniBatch.length.toDouble))
    }
  }

  /**
    * Train network using mini-batch stochastic gradient descent
    */
  def sgd(trainingData: Array[TrainingExample], epochs: Int, miniBatchSize: Int, eta: Double): Unit = {

    val n = trainingData.size

    (0 until n) foreach { j =>
      import breeze.linalg.shuffle
      shuffle(trainingData)
      val miniBatches = trainingData.grouped(miniBatchSize)
      miniBatches.foreach { miniBatch =>
        updateMiniBatch(miniBatch, eta)
      }

      print(s"Epoch $j complete")
    }

  }

  override def toString: String = {
    s"""number of layers: $numberOfLayers
       |biases: ${biases.mkString("[", ";\n", "]")}
       |weights: ${weights.mkString("[", ";\n", "]")}
     """.stripMargin
  }
}

case class TrainingExample(input: DenseVector[Double], output: DenseVector[Double])

object NetworkTest extends App {

  val network = new Network(Array(2, 3, 1))

  println(network)
}
