
/**
  * Sigmoid (Logistic) neuron - similar to perceptron, but small changes in their weights and bias
  * cause only a small change in their output (whereas for perceptrons these small changes in inputs
  * may cause big changes in output (as it iis returning 1 or 1)
  */
class SigmoidNeuron(bias: Double, weights: Seq[Double]) {
  //output is sigmoid function
  def evaluate(inputs: Seq[Double]): Double = {
    1 / (1 + math.exp(-bias - weights.zip(inputs).map { case (w, x) => w * x }.sum))
  }


}
