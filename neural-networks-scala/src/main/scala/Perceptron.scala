class Perceptron(threshold: Double, weights: Seq[Double]) {
  def evaluate(inputs: Seq[Int]): Int = {
    val weightedSum = inputs.zipAll(weights, 0, 0.0).map { case (input, weight) => input * weight }.sum
    if (weightedSum <= threshold) 0 else 1
  }
}

object PerceptronTest extends App {
  val perceptron = new Perceptron(threshold = 5, weights = Seq(6, 2, 2))
  println(perceptron.evaluate(Seq(0, 1, 1)))
  println(perceptron.evaluate(Seq(1, 0, 0)))
}