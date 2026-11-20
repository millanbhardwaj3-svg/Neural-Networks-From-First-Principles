# Transformer from Scratch (GPT-style, decoder-only)

A character-level, decoder-only transformer built from scratch, following the architecture in Vaswani et al., 2017 ("Attention Is All You Need"). Trained on a public-domain Dostoevsky corpus (Constance Garnett's translations) instead of the usual toy dataset, to see how the architecture holds up on a larger, more complex body of text.

## Why move on from WaveNet

The WaveNet-style model fixed the MLP's single-step bottleneck by combining context hierarchically, but the window it combines is still fixed in size and treated identically regardless of what's actually in it. Self-attention removes that constraint: every position can look at every earlier position and learn how much to weight each one, dynamically, based on content rather than a fixed rule.

## What's in it

- Character-level tokenization and a 90/10 train/val split, same as previous stages.
- A bigram model as the baseline, same idea as the first `makemore` stage but refit here for comparison.
- The "mathematical trick" that bridges bigram-style context to attention: averaging every previous token uniformly via a normalized lower-triangular matrix (pure bag-of-words, nothing learned), then swapping that fixed average for softmax-normalized scores that are actually learned from the data — same mechanical shape, now the weights come from the model instead of being fixed.
- A full self-attention `Head`: query/key/value projections, scaled dot-product attention (scaled by `head_size`, not the full embedding width, since multi-head makes them different), causal masking via a triangular mask, softmax, dropout.
- `MultiHead` attention: several heads run in parallel, concatenated, then passed through an output projection that mixes information back across heads.
- A `FeedForward` block (linear -> ReLU -> linear, with a 4x inner expansion, matching Vaswani et al.) applied independently per token.
- A `Block` that bundles attention and feedforward with residual connections around each — following He et al.'s observation (in the ResNet paper) that identity shortcuts let gradients pass through addition unchanged, which matters once you're stacking several blocks deep — and pre-norm LayerNorm (Ba, Kiros & Hinton, 2016) instead of BatchNorm, since it normalizes each token across its own features rather than across the batch, so it doesn't depend on batch size or need running statistics.
- Six stacked blocks, six attention heads, 384-dimensional embeddings, 256-token context length, 0.2 dropout — scaled up from the toy hyperparameters used while building the architecture piece by piece.

## Results

| Model | Val loss |
|---|---|
| Bigram baseline | 2.843 |
| Full transformer (~10k iterations) | ~1.99 train / ~2.03 val |

Sample from the trained model:

```
Him a wreagred like so mores had no
know re in of go ho peasirh.

"Nother for a is the paphrept?"

"Nower intell, in keoplay wereas these det, leng wore?"

"No!"
```

Not coherent sentences, but correctly paired punctuation and quote marks, consistent paragraph breaks, correct capitalization, and a lot of individually correctly-spelled short words — a reasonable outcome for a character-level model at this scale and loss.

## What I learned

- **BatchNorm vs LayerNorm**: BatchNorm normalizes a feature across the batch, which means it needs running statistics and behaves differently at generation time (batch size 1) than during training. LayerNorm normalizes each token across its own features instead, so it's identical at any batch size and needs no running stats.
- **The bridge from bag-of-words to attention**: a fixed, uniform average over previous tokens and full self-attention are structurally the same operation — a weighted sum over previous positions. The only thing that changes is where the weights come from: fixed and uniform, versus learned and content-dependent.
- **`nn.Module` registration**: submodules only get tracked for `.parameters()`, `.to(device)`, and the state dict if `super().__init__()` actually runs and they're assigned through `nn.ModuleList`/`nn.Sequential` rather than a plain Python list.

## Things that went wrong

- **Attention scaling factor**: I scaled the dot product by the embedding dimension instead of `head_size`. Once multi-head split the embedding across several heads, those two numbers stopped being the same thing, and scaling by the wrong one gave a technically-runnable but incorrect attention distribution — no crash, just quietly wrong, which is the kind of bug that's easy to miss if you're only checking that the code runs rather than that it's doing the right computation.
- **Stale `tril` buffer**: `register_buffer` bakes the causal mask in at construction time, using whatever `context_length` is in scope then. I scaled `context_length` up later in the notebook without rebuilding the model, so the buffer stayed sized at the old, smaller value. Python's slicing doesn't error on an out-of-range stop index, so `self.tril[:T, :T]` silently returned the whole (too-small) buffer instead of failing immediately — it only surfaced once the mismatched shapes hit `masked_fill`.

## Potential next steps

- **Subword tokenization**: switching from character-level to a BPE/subword tokenizer (what GPT-2 actually uses) would be the biggest single lever — fewer, more meaningful tokens per sequence means the model spends less capacity learning how to spell words and more learning structure and style.
- **Learning rate schedule**: a warmup-then-decay schedule instead of a flat learning rate for the whole run.
- **Weight tying**: tying the token embedding and the final output projection, a standard trick that also cuts parameter count.
- **Longer training / more depth**: the loss likely hadn't fully plateaued by the end of this run, so more iterations or more layers given more compute time would probably help further.

## Running it

Clone the repo, open `Transformer.ipynb` in the `07-transformer/` folder, and run it top to bottom — `dostoyevsky.txt` is included alongside the notebook. Training runs on Apple Silicon via MPS if available, falling back to CPU otherwise.
