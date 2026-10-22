# Bigram Language Model

A character-level language model that predicts the next letter of a name using only the previous letter. Built twice: once by counting, once as a neural network trained with gradient descent in PyTorch.

## What's in it

**Counting model.** Every name is padded with a `.` start/end token and every consecutive pair of characters is tallied in a 27×27 matrix. Normalising each row gives the probability of each next character. Sampling walks the table from `.` until it lands back on `.`.

**Neural network.** The same problem as a single linear layer: each input character is one-hot encoded, multiplied by a 27×27 weight matrix and then exponentiated and normalised (softmax) to get next-character probabilities. Trained with gradient descent on the negative log-likelihood.

**Evaluation.** Both models are scored with average negative log-likelihood over every bigram in the dataset (lower is better).

## Results

| Model | Loss |
|---|---|
| Counting (+1 smoothing) | 2.4544 |
| Neural net (1000 steps) | 2.4907 |

The neural net converges towards the counting answer, because with one-hot inputs the weight matrix is just learning the log of the count table. Sampled with the same seed, both produce almost the same names:

    kyason. kanana. milaveela. ra. drisan. ...

The L2 regularisation term on the weights plays the same role as the +1 smoothing: it pulls the probabilities towards uniform.

## Things that went wrong

**Zero counts → infinite loss.** Some bigrams never appear in the dataset, so their probability is 0 and `log(0)` is `-inf`, which makes the loss infinite. Fixed by adding 1 to every count before normalising (smoothing).

**Hardcoded dataset size.** The loss was indexed with `torch.arange(228146)`, the number of bigrams in this exact dataset. Works until the data changes. Replaced with `Xs.nelement()`.

**Unseeded weights.** `W` was initialised without a generator, so every run gave different losses and samples. Fixed by seeding it.

**Rebuilding the one-hot every step.** The training loop re-encoded the one-hot inputs on every iteration even though they never change. Moved it out of the loop.

## Running it

Open `Bigram.ipynb` and run top to bottom. It reads the dataset from `../names.txt` in the repo root (shared by all projects), so keep the notebook inside this folder.
