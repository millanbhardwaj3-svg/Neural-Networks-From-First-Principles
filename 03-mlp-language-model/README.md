# MLP Language Model

A character-level language model that predicts the next letter of a name from the previous three. Each character gets a learned embedding vector, and a multi-layer perceptron (MLP) turns the combined context into next-character probabilities. Implements the architecture from [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) (Bengio et al., 2003), applied to characters instead of words.

## What's in it

**Dataset.** Every name is turned into training examples with a sliding window: the previous 3 characters (padded with `.` at the start) predict the next one. The names are shuffled and split 80/10/10 into train, validation and test sets.

**Embeddings.** A lookup table `C` maps each of the 27 characters to a 10-dimensional vector. The vectors start random and are learned during training, so characters that behave similarly end up close together.

**Network.** The 3 embeddings are concatenated into one 30-number vector, passed through a 300-neuron tanh hidden layer and then a linear output layer giving 27 logits (one per possible next character). 17,697 parameters in total.

**Training.** 200,000 steps of minibatch gradient descent (batch size 50) on cross-entropy loss. Learning rate 0.1 which decayed to 0.01 halfway through.

## Results

| Model | Loss |
|---|---|
| Bigram (previous project) | 2.4544 |
| MLP, validation set | 2.1686 |

Sampled names:

    ahieann. sadalina. tayna. irilynna. payzly. dalani. ash. ...

## What I learned

**Why embeddings beat counting.** Extending the bigram table to 3 characters of context would need 27³ = 19,683 rows, most of them seen rarely or never. The MLP doesn't need a row per context: because it works on embeddings, what it learns about one context carries over to similar ones. That's the main idea of the Bengio paper.

**Minibatches.** Computing the gradient on 50 random examples instead of all ~180,000 makes each step far cheaper. The gradient is noisier (the loss jumps around from step to step), but it's still accurate enough to make good progress, and taking many fast steps beats a few slow ones.

**Finding the learning rate.** Tried learning rates spread exponentially from 0.001 to 1 over a short run and plotted loss against them. Loss was lowest around 0.1, so that's what training uses, with a 10× decay once progress slows.

**Train / val / test splits.** A model with enough parameters can memorise the training names. Validation loss on unseen names shows whether it's actually generalising. The test set is kept aside for one final check.

**A bad start.** The very first loss is 31.4, but a model guessing uniformly across 27 characters would score −log(1/27) ≈ 3.30. The unscaled random weights make the network confidently wrong at step 0, and the first few thousand steps are spent undoing that. Fixed in the next project with proper initialisation.

## Running it

Open `MLP.ipynb` and run top to bottom (training takes a few minutes). It reads the dataset from `../names.txt` in the repo root, so keep the notebook inside this folder.
