package pl.akkomar.neuralnetworks

import breeze.linalg.{DenseMatrix, DenseVector, argmax, sum}
import breeze.numerics._
import breeze.linalg.shuffle

import scala.collection.mutable
import scala.util.Random

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
  val biases: Array[DenseVector[Double]] = sizes.tail.map { layerSize =>
    val normal01 = breeze.stats.distributions.Gaussian(0, 1)
    DenseVector.rand(layerSize, normal01)
  }

  //w=weights(n) are weights connecting n+1 and n+2 layers of neurons
  //weights(n)(j,k) is weight for the connection between k-th neuron in the n+1 layer and j-th neuron in the n+2 layer
  val weights: Array[DenseMatrix[Double]] = sizes.dropRight(1).zip(sizes.tail).map { case (x, y) =>
    val normal01 = breeze.stats.distributions.Gaussian(0, 1)
    DenseMatrix.rand(y, x, normal01)
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
    *
    * @param x
    * @param y
    * @return
    */
  private def backProp(x: DenseVector[Double], y: DenseVector[Double]): (Array[DenseVector[Double]], Array[DenseMatrix[Double]]) = {
    val nablaB: Array[DenseVector[Double]] = biases.map(b => DenseVector.zeros[Double](b.length))
    val nablaW: Array[DenseMatrix[Double]] = weights.map(w => DenseMatrix.zeros[Double](w.rows, w.cols))

    var activation = x

    //all activations, layer by layer
    val activations = new mutable.MutableList[DenseVector[Double]]() += x

    //all z vectors, layer by layer
    //z - weighted input to the neurons in layer `l`
    val zs = new mutable.MutableList[DenseVector[Double]]()

    biases.zip(weights).foreach { case (b, w) =>
      val z: DenseVector[Double] = (w * activation) + b
      zs += z
      activation = sigmoid(z)
      activations += activation
    }

    //backward pass
    var delta = costDerivative(activations.last, y) :* sigmoidPrime(zs.last)
    nablaB(nablaB.size - 1) = delta
    nablaW(nablaW.size - 1) = delta * activations.reverse(1).t

    ((numberOfLayers - 3) to 0 by -1).foreach { l =>
      val z = zs(l)
      val sp = sigmoidPrime(z)
      delta = (weights(l + 1).t * delta) :* sp
      nablaB(l) = delta
      nablaW(l) = delta * activations(l).t
    }
    (nablaB, nablaW)
  }

  /**
    *
    * @param outputActivations
    * @param y
    * @return vector of partial derivatives dCx/da for the output activations
    */
  private def costDerivative(outputActivations: DenseVector[Double], y: DenseVector[Double]): DenseVector[Double] = {
    outputActivations - y
  }

  /**
    * Derivative of the sigmoid function
    *
    * @param z
    * @return
    */
  private def sigmoidPrime(z: DenseVector[Double]): DenseVector[Double] = {
    sigmoid(z) :* (-sigmoid(z) + 1d)
  }

  /**
    * Updates the network weights and biases according to single iteration of gradient descent using backpropagation
    * to a single miniBatch of training data
    *
    * @param miniBatch
    * @param eta learning rate
    */
  private def updateMiniBatch(miniBatch: Array[TrainingExample], eta: Double): Unit = {
    val nablaB: Array[DenseVector[Double]] = biases.map(b => DenseVector.zeros[Double](b.length))
    val nablaW: Array[DenseMatrix[Double]] = weights.map(w => DenseMatrix.zeros[Double](w.rows, w.cols))
    miniBatch.foreach { case TrainingExample(x, y) =>
      val (deltaNablaB, deltaNablaW) = backProp(x, y)
      nablaB.indices.foreach { n =>
        nablaB(n) :+= deltaNablaB(n)
      }
      nablaW.indices.foreach { n =>
        nablaW(n) :+= deltaNablaW(n)
      }
    }
    //update weights
    weights.indices.foreach { n =>
      weights(n) :-= (nablaW(n) :*= (eta / miniBatch.length.toDouble))
    }
    //update biases
    biases.indices.foreach { n =>
      biases(n) :-= (nablaB(n) :*= (eta / miniBatch.length.toDouble))
    }
  }

  /**
    * Train network using mini-batch stochastic gradient descent
    */
  def train(trainingData: Array[TrainingExample], epochs: Int, miniBatchSize: Int, eta: Double, testData: Option[Array[TrainingExample]] = None): Unit = {
    (0 until epochs) foreach { j =>
      shuffle(trainingData)
      val miniBatches = trainingData.grouped(miniBatchSize)
      miniBatches.foreach { miniBatch =>
        updateMiniBatch(miniBatch, eta)
      }

      if (j % 1000 == 0) {
        testData match {
          case None =>
            println(s"Epoch $j complete")
          case Some(data) =>
            val error = evaluateError(data)
            println(s"Epoch $j, error: $error")
        }
      }
    }
  }

  /**
    * Compute number of test inputs for which neural network outputs the correct result.
    * Output is assumed to be the index of final layer neuron with highest activation
    */
  private def evaluate(testData: Array[TrainingExample]): Int = {
    testData.map { case TrainingExample(input, expectedOutput) =>
      val output = feedForward(input)
      if (argmax(output) == argmax(expectedOutput)) 1 else 0
    }.sum
  }

  /**
    * Compute meas squared error
    */
  private def evaluateError(testData: Array[TrainingExample]): Double = {
    testData.map { case TrainingExample(input, expectedOutput) =>
      val output = feedForward(input)
      sqrt(sum((output - expectedOutput) :^ 2d))
    }.sum / testData.length
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
