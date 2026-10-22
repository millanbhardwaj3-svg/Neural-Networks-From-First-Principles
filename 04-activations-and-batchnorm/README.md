# Activations, Initialisation and BatchNorm

Two notebooks on making a neural network train well (not just train). `BatchNorm.ipynb` fixes the initialisation of the MLP from project 03 and adds batch normalisation. `Diagnostics.ipynb` scales the same ideas up to a 5-hidden-layer network and checks its health with diagnostic plots. Implements ideas from [Batch Normalization](https://arxiv.org/abs/1502.03167) (Ioffe & Szegedy, 2015), [Delving Deep into Rectifiers](https://arxiv.org/abs/1502.01852) (He et al., 2015) for Kaiming initialisation and [Understanding the difficulty of training deep feedforward neural networks](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) (Glorot & Bengio, 2010) for saturation and checking gradients layer by layer.

Start with `BatchNorm.ipynb`, then `Diagnostics.ipynb`.

## Why initialisation matters

The MLP in project 03 started with a loss of **31.4**. A network guessing uniformly across 27 characters would score −log(1/27) ≈ **3.30**, so it started out ~10× worse than random. Training still worked, but the first few thousand steps were spent undoing a bad start instead of learning.

Two problems cause this:

**Overconfident output.** Large random weights at the output layer produce extreme logits so softmax puts nearly all its probability on one wrong character. Fix: scale the output weights by 0.01 and set its bias to 0 so the network starts out predicting roughly uniform probabilities.

**Saturated tanh neurons.** If the hidden layer's pre-activations are too large, tanh outputs get pushed to ±1, where its slope (1 − tanh²) is ~0. A saturated neuron passes almost no gradient back, so it stops learning. Fix: scale the weights by gain/√fan_in (Kaiming init), which keeps the pre-activations around unit variance no matter how many inputs feed into a neuron. The He et al. paper derives the gain for ReLU, for tanh the standard gain is 5/3, correcting for tanh squashing its inputs.

With both fixes, the initial loss is close to what uniform guessing gives.

## Why BatchNorm helps

Good init only gets the network to a good starting point. As weights change during training, the hidden layer's inputs can drift back into the saturated range. Batch normalisation fixes this throughout training: before the tanh, each neuron's pre-activation is normalised to zero mean and unit variance across the batch, then scaled and shifted by a learned gain and bias. The network gets well-behaved activations without depending as much on getting the initial weights exactly right.

Things learned implementing it:
- **The bias before BatchNorm is useless.** Normalising subtracts the mean, which cancels out any bias, so `b1` is removed. BatchNorm's own learned bias does that job instead.
- **Inference needs running statistics.** At sampling time there's one example and no batch to take a mean over. A running mean and std, tracked during training, are used instead.

## Results (BatchNorm.ipynb)

| Model | Initial loss | Val loss |
|---|---|---|
| MLP, project 03 | 31.4 | 2.1686 |
| + init fixes + BatchNorm | 3.30 | 2.1152 |

Sampled names: `teyas. kaia. elleyah. cha. gabmarvius. amik. elin. kion. ...`

The final improvement is small. With only one hidden layer, the bad init mainly wasted early steps rather than ruining the final model. These fixes matter more as networks get deeper, where bad activations and gradients compound layer after layer. `Diagnostics.ipynb` checks that they keep a deeper network healthy.

## Diagnostics (Diagnostics.ipynb)

A 5-hidden-layer version of the same model (each layer `Linear → BatchNorm → tanh`, 100 neurons), built from layer classes. It trains for only 1,000 steps — the goal is checking the network's health rather than its final loss. Each plot has a target to look for, and the way it fails tells you what to change.

**Activation distribution.** Histogram of each tanh layer's outputs.
*Want:* similar shapes in every layer, with few values at ±1.
*If not:* layers piling up at ±1 means they're saturating (weights too large); layers collapsing to 0 means the signal is dying (weights too small). Adjust the init gain.
*Here:* std is 0.63–0.65 in every layer, and saturation stays low, going from 3.1% in the first layer to 1.9% in the last.

**Gradient distribution.** Histogram of the gradient flowing back into each tanh layer.
*Want:* all layers overlapping.
*If not:* gradients shrinking towards the first layers means they're vanishing, so the early layers barely learn; growing means they're exploding.
*Here:* gradient std goes from 2.7e-3 in the first layer to 2.1e-3 in the last, nearly the same throughout.

**Weight gradients.** Gradient on each weight matrix, and its grad:data ratio (gradient size relative to the weight's size).
*Want:* similar ratios across layers, so no layer learns much faster or slower than the rest.
*Here:* the hidden layers are all around 2–3 × 10⁻², but the embedding table `C` shows a spike at 0, because characters that don't appear in a batch get no gradient that step.

**Update:data ratio.** How big each update is relative to the weights it changes, tracked over training (log10 scale).
*Want:* around −3, meaning each step changes the weights by ~1/1000. This is the most useful plot for picking a learning rate: well below −3 means the learning rate is too low, and well above means it's too high and training gets unstable.
*Here:* the hidden layers settle around −2.5, so a learning rate of 0.1 is on the upper end for this network. `C` starts lower at −3.7 and learns slowest, again because of its sparse gradients.

## Things that went wrong

**Forgetting BatchNorm at inference.** The evaluation and sampling code originally skipped the normalisation step. The network was trained with normalised inputs to tanh, so this was effectively a different network at test time: loss came out at 3.06, barely better than guessing, and sampled names never ended (`bryjcadelandrelleggapmishaun...`). Fixed by normalising with the running mean and std at inference, as described above.

## Running it

Run each notebook top to bottom (`BatchNorm.ipynb` takes a few minutes to train). Both read the dataset from `../names.txt` in the repo root, so keep both notebooks inside this folder.
