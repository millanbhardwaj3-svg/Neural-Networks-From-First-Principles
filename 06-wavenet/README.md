# WaveNet-Style Language Model

A character-level language model that predicts the next letter of a name from the previous 8, combining them gradually in a tree (pairs of characters, then pairs of pairs, then pairs of those) instead of all at once. Implements the hierarchical idea from [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499) (van den Oord et al., DeepMind, 2016), applied to characters instead of audio samples.

## Why move on from the MLP

The MLP from projects 03 and 04 has a structural limit: it squashes the whole context into the hidden layer **in one step**. The embeddings of every context character are concatenated into one long vector and multiplied by a single weight matrix.

- **Longer context just makes the problem worse.** Going from 3 to 8 characters means concatenating 8 embeddings into an 80-number vector, and the first layer has to make sense of all of it at once.
- **Nothing is reused across positions.** Every position gets its own separate block of weights, so a pattern learned for characters 1–2 has to be learned again from scratch for characters 5–6.
- **No sense of order or locality.** Neighbouring characters are clearly related (`th`, `ll`, `ie`), but the MLP treats the context as one flat vector with no notion of which characters are next to each other.

WaveNet addresses the same problem in audio, where a model needs thousands of past samples of context. Instead of looking at all of them at once, each layer combines neighbouring pairs so the effective context doubles with every layer.

## What's in it

**Hierarchy.** 8 context characters are embedded (10-d each) then fused in pairs over three layers: 8 → 4 → 2 → 1, each doing `Flatten_Consecutive(2) → Linear → BatchNorm → tanh`, with 200 hidden neurons, followed by a linear output layer with 27 logits. 170,897 parameters in total.

**Shared weights across positions.** The same `Linear` weights are applied to every pair: the 4 pairs at the first level all use the same weight matrix. That makes each layer equivalent to a 1-D convolution with kernel size 2 and stride 2, so a pattern learned in one place works everywhere in the context. This is the main idea taken from WaveNet, where dilated causal convolutions double their reach at each layer in the same way.

**Layer classes.** Built from small PyTorch-style modules (`Embedding`, `Flatten_Consecutive`, `Linear`, `BatchNorm1d`, `Tanh`, `Sequential`) written from scratch, so the model is defined as a list of layers.

**BatchNorm on 3-D inputs.** Inside the hierarchy, activations have shape `(batch, positions, channels)`. BatchNorm has to average over both the batch and the position dimensions, so each of the 200 channels gets one mean and variance shared across all positions — matching the idea of a channel being the same feature wherever it appears.

**Training.** 200,000 steps, batch size 32, learning rate 0.1 decayed to 0.01 halfway through. About 5 minutes to train.

## Results

| Model | Context | Parameters | Val loss |
|---|---|---|---|
| MLP (project 03) | 3 | 17,697 | 2.1686 |
| MLP + BatchNorm (project 04) | 3 | 12,097 | 2.1152 |
| **WaveNet-style (this project)** | **8** | **170,897** | **1.9845** |

Train loss 1.7612. Sampled names:

    teyas. kaisley. evalyn. tlyn. ...

This is the biggest improvement so far compared to the MLP. It comes from three changes at once (longer context, more parameters and the hierarchical structure), and this project doesn't isolate how much each one contributes — a flat MLP with the same 8-character context and a similar parameter count would be the fair comparison, and wasn't run here.

The gap between train (1.76) and val (1.98) is much larger than in project 04 (2.07 vs 2.12), so the bigger model is starting to overfit.

## What I learned

**Tracking tensor shapes is most of the work.** Each layer changes the shape (`(32, 8, 10)` → `(32, 4, 20)` → `[fill in]` → ...), and almost every bug in this project was a shape mismatch. Writing out the shape after each layer before writing any code would have caught most of them.

**Matrix multiply works on trailing dimensions.** `(32, 4, 20) @ (20, 200)` treats the first two dimensions as a batch and applies the same weights to all 4 positions. That single fact makes the weight sharing free: no loop over positions is needed.

## Things that went wrong

**Layer sizes for the deeper levels.** The second and third `Linear` layers were first written with `2 * emb_size` as their input size, instead of `2 * n_hidden` (400). Only the first level sees embeddings; every level after it sees the 200-wide output of the level below.

**Context length out of sync.** The dataset was initially built with 3 characters of context while the model expected 8, and the sampling loop also started with 3. The tree needs exactly 8 positions to pair down cleanly (8 → 4 → 2 → 1), so the mismatch crashed in `Flatten_Consecutive`.

**Embeddings not training.** `Embedding.parameters()` returned an empty list, so the embedding table was never updated by the optimiser (the model just trained on random, fixed embeddings). Easy to miss because the loss still went down.

**BatchNorm over the wrong dimensions.** The original BatchNorm only averaged over the batch dimension, which gives separate statistics for every position. It needed to average over `(0, 1)`. A related bug, `x.dim` instead of `x.dim()`, meant the dimension check never ran correctly either way.

## Running it

Open `WaveNet.ipynb` and run top to bottom (training takes about 5 minutes). It reads the dataset from `../names.txt` in the repo root, so keep the notebook inside this folder.
