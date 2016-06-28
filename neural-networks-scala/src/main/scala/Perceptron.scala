class Perceptron(threshold: Double, weights: Seq[Double]) {
  def evaluate(inputs: Seq[Int]): Int = {
    val weightedSum = inputs.zipAll(weights, 0, 0.0).map { case (input, weight) => input * weight }.sum
    if (weightedSum <= threshold) 0 else 1
  }
}
