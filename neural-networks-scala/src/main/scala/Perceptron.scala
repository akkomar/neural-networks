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

object BitAddingNetwork {
  // Network schema:
  //
  //  x1----->p2
  //    \    ↗  \
  //     ↘  /    ↘
  //      p1      p5--->bitwise sum x1+x2
  //     ↗ |\    ↗
  //    /  |  ↘ /
  //  x2------p3
  //       |
  //   (-4)-->p4------->carry bit - bitwise product x1*x2
  //
  //  all connections have weights -2, except for the one with p4
  //  (which has weight -4, or just two -2 connections with p1)

  val p1, p2, p3, p4, p5 = nandPerceptron()

  // returns two bits: sum and carry bit
  def evaluate(x1: Int, x2: Int): (Int, Int) = {
    val p1Output = p1.evaluate(Seq(x1, x2))
    val p2Output = p2.evaluate(Seq(x1, p1Output))
    val p3Output = p3.evaluate(Seq(x2, p1Output))
    val p4Output = p4.evaluate(Seq(p1Output, p1Output))
    val p5Output = p5.evaluate(Seq(p2Output, p3Output))
    (p5Output, p4Output)
  }

  private def nandPerceptron() = new Perceptron(bias = 3, Seq(-2, -2))
}

object BitAddingNetworkTest extends App {
  assert(BitAddingNetwork.evaluate(0, 0) ==(0, 0))
  assert(BitAddingNetwork.evaluate(0, 1) ==(1, 0))
  assert(BitAddingNetwork.evaluate(1, 0) ==(1, 0))
  assert(BitAddingNetwork.evaluate(1, 1) ==(0, 1))
}