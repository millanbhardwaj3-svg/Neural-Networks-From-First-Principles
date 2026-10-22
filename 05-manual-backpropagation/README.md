# Manual Backpropagation

The same MLP as project 04 (embeddings → linear → BatchNorm → tanh → linear → cross-entropy), with every gradient in the backward pass worked out by hand instead of calling `loss.backward()`. Each manual gradient is checked against PyTorch's autograd to confirm it's correct.

Most of this project was maths rather than code: writing out the chain rule for each operation, tracking tensor shapes and simplifying the results on paper before turning them into one or two lines of PyTorch.

## What's in it

**Forward pass, broken into pieces.** The forward pass is written out as ~20 small steps (e.g. cross-entropy becomes max → subtract → exp → sum → invert → multiply → log → mean), so every intermediate tensor has its own gradient to derive.

**Backward pass, by hand.** Starting from the loss, each step's gradient is derived from the one after it using the chain rule, all the way back to the embedding table `C`. A `compare` function checks each one against PyTorch's gradient: exact match, close match (`allclose`), and the largest difference.

**Two simplified derivations.** The many small steps for cross-entropy and for BatchNorm are each collapsed into a single closed-form expression, derived on paper and checked against autograd.

## Results

All 26 manual gradients match PyTorch, from `logprobs` back to `C`. The two simplified versions match to within ~10⁻⁸:

| Gradient | Max difference from autograd |
|---|---|
| Cross-entropy, simplified | 6.8 × 10⁻⁹ |
| BatchNorm, simplified | 9.3 × 10⁻¹⁰ |

Some checks show `exact: False` but `approximate: True`. Doing the same maths in a different order gives slightly different floating-point rounding, so values agree to within about 10⁻⁸ rather than bit-for-bit.

## What I learned

**Patterns that come up again and again.** A few rules cover most of the backward pass:
- *A variable used in two places gets the sum of both paths' gradients.* `counts` feeds into the probabilities directly and through their sum; `logits` feeds into the subtraction directly and through the max. Missing the second path gives a wrong answer.
- *Broadcasting in the forward pass becomes summing in the backward pass.* If a `(1, 200)` tensor was stretched across a batch of 32, its gradient is the sum over those 32 rows.
- *Summing in the forward pass becomes broadcasting in the backward pass.* (The reverse of the above) if a batch of 32 was reduced to a single value, that value's gradient broadcasts back out to all 32 original positions.
- *Matrix multiply gradients follow from the shapes.* For `logits = h @ W2`, the only arrangements that produce the right shapes are `dh = dlogits @ W2.T` and `dW2 = h.T @ dlogits`.
- *An embedding lookup scatters gradients back.* Each row of `C` gets the summed gradient from every position in the batch where that character appeared.

**Cross-entropy collapses to one line.** After deriving each step by hand, about 10 steps of gradients reduce to:

    dlogits = (softmax(logits) − one_hot(target)) / batch_size

Each wrong character's logit is pushed down in proportion to its predicted probability, and the correct one is pulled up by (1 − its probability). That push is largest exactly where the model is most wrong.

**BatchNorm is the hardest derivation.** Each output depends on the batch mean and variance, which depend on *every* example in the batch, so every input affects every output. Deriving it by hand (including the n/(n−1) from Bessel's correction) collapses into a single expression that replaces about 8 separate steps.

**The bias before BatchNorm really is useless.** Project 04 removed `b1` because BatchNorm subtracts the mean. This notebook keeps it in and shows its gradient is about 3 × 10⁻⁹, compared with about 0.05 for `W1`. The network has no way to learn it.

## Things that went wrong

**Embedding gradients sent to every row.** The first version of the `C` gradient added each position's gradient to the whole table instead of just the row for that character (`dC += ...` instead of `dC[ix] += ...`). The fix is the scatter-back idea above.

**Small slips in the forward pass.** The variance was written as 1 divided by the sum instead of the sum divided by (n − 1), and one normalisation step used `−` where it should have used `×`. The backward pass was derived for the correct maths, so the gradient checks exposed the mismatch.

## Running it

Open `ManualBackpropagation.ipynb` and run top to bottom. Every `compare` line should show `approximate: True`. It reads the dataset from `../names.txt` in the repo root, so keep the notebook inside this folder.
