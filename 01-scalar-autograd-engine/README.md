# Autograd From Scratch

Backpropagation and a small neural net built from scratch in Python, with no ML libraries. (Built with help from Andrej Karpathy's tutorials) 

## What's in it

**Engine.** A `Value` class that wraps a single number and records the operations
that produced it. Each operation stores its own local derivative so calling
`backward()` on the output walks the graph in reverse and fills in the gradient
of every value that contributed to it.

**Neural net.** `Neuron`, `Layer` and `MLP` classes built on top of the engine,
plus a training loop doing gradient descent on a toy dataset.

## Things that went wrong

[The saturation one — tanh on the output layer can't produce targets outside
[-1, 1], so the gradient dies and the loss sits there. Took a while to spot.]

[The stale gradients — forward, backward and update all have to happen every
iteration, and micrograd accumulates gradients rather than replacing them, so
they need zeroing first.]

[The bracket error in __pow__.]


## Running it

Open the notebook and run top to bottom.
