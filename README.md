# Niche

A small character-level GPT-style transformer trained from scratch on the (English, public-domain) works of Friedrich Nietzsche. It's a learning project — the goal is to build a decoder-only transformer by hand and watch it learn to imitate Nietzsche's prose, not to produce a state-of-the-art language model.

## Repo structure

```
Niche/
├── niche.ipynb                     # everything: data prep, model, training loop, generation
├── niche_classes.py                # model classes extracted from the notebook, importable
├── grab_nietzsche.py               # downloads the corpus from Project Gutenberg
├── data/                           # downloaded texts (git-ignored)
│   ├── grabbed_nietzsche.txt           # all books concatenated — this is what training reads
│   └── <book>.txt                      # individual books, cached so re-runs don't re-download
├── niche_model.pt                  # best-val checkpoint (git-ignored)
├── niche_attention_analysis.ipynb  # mech-interp: copying scores, OV/QK circuits, ablations
├── induction_head.md               # writeup of the interpretability findings
├── evidence.md                     # raw per-head copying-score numbers backing the writeup
├── figures/                        # figures for the writeup + make_figures.py to regenerate them
├── requirements.txt
└── README.md
```

The `niche.ipynb` notebook is the training project. Cells run top-to-bottom: imports → hyperparameters → load/encode data → train/test split → batching → model classes → **resume from checkpoint** → **training loop** → manual save → sample generation.

## Interpretability

After training, `niche_attention_analysis.ipynb` digs into the attention heads (copying scores, OV diagonals, attention patterns, targeted ablations). The findings — a copying head with the OV half of a bracket-completion mechanism but a QK circuit that can't route it — are written up in [`induction_head.md`](induction_head.md), with figures in `figures/` and raw numbers in [`evidence.md`](evidence.md).

Data is not committed (`.gitignore` excludes `data/` and `niche_model.pt`). Run `python grab_nietzsche.py` to fetch the 13 texts from Gutenberg into `data/`; it strips the Gutenberg boilerplate and writes the concatenated `grabbed_nietzsche.txt` that training consumes.

## How the transformer is structured

A standard decoder-only (GPT-style) transformer, written from scratch:

- **Tokenization:** character-level. The corpus has **177 unique characters** (includes accented Latin, Greek, and punctuation), so the vocab is tiny and every "token" is one character.
- **Embeddings:** a token embedding plus a learned positional embedding, summed.
- **Blocks:** `n_layers = 6` identical blocks, each = pre-norm multi-head self-attention + pre-norm feed-forward, both with residual connections.
  - **Attention:** `n_head = 4` causal heads. Q/K/V come from a single fused `Linear(n_embd, 3·n_embd)` projection, then split/reshaped per head. Uses `F.scaled_dot_product_attention` with `is_causal=True` (a manual masked-softmax path is kept for reference).
  - **Feed-forward:** `Linear(n_embd, 4·n_embd) → GELU → Linear(4·n_embd, n_embd)`.
  - LayerNorm before each sub-layer, `dropout = 0.2`.
- **Head:** final LayerNorm → `Linear(n_embd, vocab_size)` → cross-entropy against the next character.

### Configuration

| Hyperparameter | Value |
|---|---|
| `n_embd` (embedding width) | 256 |
| `n_head` | 4 |
| `n_layers` | 6 |
| `block_size` (context length) | 256 |
| `batch_size` | 64 |
| `dropout` | 0.2 |
| `learning_rate` | 3e-4 (AdamW) |
| vocab size | 177 |
| **parameters** | **~4.9M** |

Training runs on CUDA → MPS → CPU (whichever is available), with `bfloat16` autocast. On Apple Silicon (MPS) ~1000 steps takes ~2–2.5 minutes.

### Checkpointing

The training loop evaluates train/test loss every 1000 steps and **only writes `niche_model.pt` when test loss reaches a new low** — so the saved file holds the best-validation weights, not the last ones. Two safeguards make this robust to the notebook workflow:

- `best_val` is **seeded from the `val_loss` stored in `niche_model.pt` on disk** when the kernel starts, so restarting and re-running can never overwrite a good checkpoint with a worse one.
- The **resume cell sits directly above the training loop**, so "Run All" loads the saved weights and continues from them. To train from scratch, run the model-build cell and skip the resume cell.
- The manual save cell writes to a separate `niche_model_last.pt`, so it can't clobber the best checkpoint.

## Results

Best validation loss reached so far is **~1.13 nats/char** (cross-entropy), in a single continuous run that bottomed out around step 7000. Train loss continues to drift below that, so the train/test gap is ~0.07–0.08 nats — the onset of mild overfitting. Samples are recognizably Nietzsche-flavored English: grammatical local structure, plausible vocabulary, but no long-range coherence.

## Limitations

This is the honest part. The model is squarely in a **data-bound, over-parameterized** regime, and several design choices cap how good it can get.

### Dataset size is the binding constraint

- The corpus is **~5.8M characters** (5,231,466 train / 581,274 test on a 90/10 split). That is *small*.
- The `data/` folder may look like it holds ~12M chars, but the individual book files and `grabbed_nietzsche.txt` are **the same corpus stored twice** — there is no hidden extra data to concatenate. This is essentially all of Nietzsche's public-domain English translations, so the dataset is close to its practical ceiling; you can't meaningfully grow it without adding a *different* author (which changes what the model is).

### The model is over-parameterized for the data

- **~4.9M parameters vs ~5.2M training characters ≈ a 1:1 parameter-to-token ratio.** Compute-optimal (Chinchilla-style) training is closer to **20:1** tokens per parameter. We are roughly 20× under-data'd.
- A model with as many parameters as training tokens has more than enough capacity to start **memorizing** the training set, which is exactly why the train/test gap opens up as training continues.
- **Counterintuitive consequence:** making the model *bigger* would make overfitting *worse*, not better, on this corpus. The lever is more/different data or stronger regularization, not more capacity.

### Validation loss has essentially plateaued

- Test loss flattens around **~1.13 nats/char** and then bounces within noise while train loss keeps falling. That plateau — not the size of the gap — is the real signal that we're at the dataset's ceiling for this model.
- The plateau is judged from a **noisy 20-batch loss estimate**, so individual "new best" saves can fire on a lucky-low evaluation. Increasing the eval batch count makes the save decision cleaner at the cost of slower logging.

### Character-level tokenization

- Char-level keeps the vocab tiny and the project simple, but it forces the model to spend capacity learning spelling and word boundaries, and it caps the effective context: `block_size = 256` is only ~256 characters (≈40–50 words) of history. Subword/BPE tokenization would let the same parameters model much longer dependencies and is usually a bigger qualitative win than more characters.

### Other caveats

- **No learning-rate schedule** (flat 3e-4) and **no weight decay tuning** beyond AdamW's default — both could squeeze out marginal gains.
- **Optimizer state is not checkpointed.** Resuming reloads model weights but creates a fresh AdamW, so momentum/variance buffers reset (a brief warm-up bump, not a correctness issue).
- **Mixed translators.** The corpus stitches together translations from different eras and translators, so "Nietzsche's style" here is really an average over translation styles, not a single voice.
- Trained and benchmarked only on **Apple Silicon / MPS**; numbers and `bfloat16` behavior may differ on CUDA or CPU.

## Possible next steps

- Switch char-level → BPE/subword tokenization (most likely to improve sample quality).
- Add a cosine LR schedule with warm-up and tune weight decay / dropout.
- Checkpoint optimizer state for true resume.
- Accept the ~1.13 ceiling as the floor of this corpus, and treat the project as "done" for Nietzsche-only English text.
