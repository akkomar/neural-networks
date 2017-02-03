package pl.akkomar.neuralnetworks

import breeze.linalg.{DenseVector, argmax}
import org.scalatest.{FlatSpec, GivenWhenThen, Matchers}

class NetworkTest extends FlatSpec with GivenWhenThen with Matchers {

  "Network" should "learn to function as XOR operation" in {
    Given("network with one hidden layer")
    //output vector has two elements - we treat first as `true`, second as `false`
    val network = new Network(Array(2, 3, 2))
    And("training data representing XOR operation")
    val data = Array(
      TrainingExample(DenseVector(1d, 1d), DenseVector(0d, 1d)),
      TrainingExample(DenseVector(0d, 0d), DenseVector(0d, 1d)),
      TrainingExample(DenseVector(0d, 1d), DenseVector(1d, 0d)),
      TrainingExample(DenseVector(1d, 0d), DenseVector(1d, 0d))
    )
    //we'll use the same data for evaluating error during learning
    val testData = data

    When("we train network")
    network.train(
      trainingData = data,
      epochs = 1000000,
      miniBatchSize = 2,
      eta = 0.001,
      testData = Some(testData)
    )

    Then("it returns correct outputs for all training examples")
    data.foreach { case TrainingExample(input, expectedOutput) =>
      argmax(network.feedForward(input)) shouldEqual argmax(expectedOutput)
    }
  }
}
