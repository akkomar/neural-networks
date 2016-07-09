class Perceptron(bias: Double, weights: Seq[Double]) {
  def evaluate(inputs: Seq[Int]): Int = {
    val weightedSum = inputs.zipAll(weights, 0, 0.0).map { case (input, weight) => input * weight }.sum
    if (weightedSum + bias <= 0) 0 else 1
  }
}

object PerceptronTest extends App {
  val perceptron = new Perceptron(bias = -5, weights = Seq(6, 2, 2))
  println(perceptron.evaluate(Seq(0, 1, 1)))
  println(perceptron.evaluate(Seq(1, 0, 0)))

  // perceptron that implements NAND gate
  val nandPerceptron = new Perceptron(bias = 3, Seq(-2, -2))
  assert(nandPerceptron.evaluate(Seq(0, 0)) == 1)
  assert(nandPerceptron.evaluate(Seq(1, 0)) == 1)
  assert(nandPerceptron.evaluate(Seq(1, 1)) == 0)
}

