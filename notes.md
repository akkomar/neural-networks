## Artificial neurons
### Perceptron
Code: [Perceptron.scala](neural-networks-scala/src/main/scala/Perceptron.scala)
### Sigmoid neurons
Code: [SigmoidNeuron.scala](neural-networks-scala/src/main/scala/SigmoidNeuron.scala)

## Neural Network
## Learning with gradient descent
* Stochastic Gradient Descent- standard learning algorithm for neural networks
* our cost function to minimize is **Mean Squared Error**. Theoretically we could maximize number of properly classified images instead, but that wouldn't be smooth function of weights and biases in the network (small changes to weights and biases wouldn't cause any change in number of correctly classified images - it would make it difficult to figure out how to change weights and biases to get improved performance)
*
