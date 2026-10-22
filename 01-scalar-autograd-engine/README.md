# Autograd From Scratch

Backpropagation and a small neural net built from scratch in Python, with no ML libraries. (Built with help from Andrej Karpathy's tutorials)

## What's in it

**Engine.** A `Value` class that wraps a single number and records the operations that produced it. Each operation stores its own local derivative so calling `backward()` on the output walks the graph in reverse and fills in the gradient of every value that contributed to it.

**Neural net.** `Neuron`, `Layer` and `MLP` classes built on top of the engine, plus a training loop doing gradient descent on a toy dataset. Hidden layers use ReLU, the output layer is linear, and the learning rate decays over training.

## Things that went wrong

- **tanh saturation (output layer).** tanh caps predictions in (-1, 1), so targets outside that range (like a target of -3.0) are structurally unreachable no matter how long training runs. On top of that as tanh's output gets pushed toward ±1, its derivative (`1 - tanh(x)²`) shrinks toward zero, so the more saturated a neuron gets, the weaker its gradient signal becomes — learning slows down right when the network is already struggling to hit the target. Eventually fixed by making the output layer linear (no activation at all).

- **The stale gradients.** Forward, backward and update all have to happen every iteration, and micrograd accumulates gradients rather than replacing them, so they need zeroing first.


- **Dying ReLU, output layer.**  tried ReLU as the final activation, asked for a target of -1, got 0 back. ReLU floors at zero so it can't reach negative targets. Fixed by making the last layer skip activation entirely (linear output), hidden layers keep ReLU.

- **Dead ReLU (whole network).** scaled up the training data and the network stopped learning entirely, loss frozen bit-for-bit across 900 steps. Bigger inputs pushed pre-activations hard negative on every example, killing every hidden neuron so the network collapsed to predicting one constant (the mean of the targets, confirmed by hand). Fixed by scaling inputs down before training.

## Running it

Open the notebook and run top to bottom.
