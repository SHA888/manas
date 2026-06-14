# Manas Roadmap

Manas is an experimental self-growing local AI system written in Rust.

The long-term goal is to build a local-first AI system that can learn from text, files, and sources, persist its own memory, grow carefully over time, and eventually generate answers through a custom language path built from scratch.

Manas is **not** trying to replace large hosted LLMs. It is a learning and research project focused on custom neural memory, local persistence, source-aware growth, and a small transformer-style language system implemented in Rust.

## Current Status

| Version | Milestone | Status |
| --- | --- | --- |
| v0.1 | Local memory foundation | Done |
| v0.2 | Next-token prediction | Done |
| v0.3 | Real text generation | Done |
| v0.4 | Single-head causal attention | Done |
| v0.5 | Tiny transformer block | Done |
| v0.6 | Transformer-assisted prediction and generation | Done |
| v0.7 | Transformer output-head training | Done |
| v0.7.1 | Controlled neuron growth during language training | Done |
| v0.7.2 | Better inspect for language and transformer state | Done |
| v0.8 | Train transformer FeedForward layer | Done |
| v0.8.1 | Transformer training metrics | Done |

## Completed Milestones

### v0.1 — Local Memory Foundation

Manas started as a local learning and memory system.

Completed:

- Custom `Neuron`, `Layer`, and `Network`
- `.manas` persistent brain file
- Raw text learning
- File ingestion
- Vocabulary and token embeddings
- Source-aware neurons
- Source/freshness metadata
- Importance and protection scoring
- `inspect`, `trace`, `neurons`, and related debugging commands

Goal achieved:

> Manas can locally learn and persist knowledge in a custom Rust-based neural memory system.

---

### v0.2 — Next-Token Prediction

The first language milestone introduced token-order learning.

Completed:

- `manas-language` crate
- Sequence training examples
- `train-language` command
- `predict-next` command
- Hybrid sequence memory + neural predictor
- Suffix backoff for learned transitions
- Source metadata fix for language-trained neurons

Example behavior:

```text
Rust is a      -> systems
Ownership is   -> rust's
most unique    -> feature
```

Goal achieved:

> Manas can learn small ordered token sequences and predict the next token.

---

### v0.3 — Real Text Generation

Generation was built on top of next-token prediction.

Completed:

- `generate` command
- Repeated next-token prediction loop
- Prompt included in final generated text
- Loop/repetition protection
- Stable generation from learned sequences

Example behavior:

```text
Prompt: Rust is
Generated: rust is a systems programming language focused on safety and performance
```

Goal achieved:

> Manas can generate small learned text sequences locally.

---

### v0.4 — Single-Head Causal Attention

A standalone causal attention module was added.

Completed:

- Custom `CausalSelfAttention`
- Q/K/V/O projections
- Causal masking
- Numerically stable softmax
- Shape tests
- Causal-mask tests
- Empty-input tests
- NaN-safety tests

Goal achieved:

> Manas has a custom Rust causal self-attention module that can be reused inside transformer-style language blocks.

---

### v0.5 — Tiny Transformer Block

A minimal transformer-style block was added on top of attention.

Completed:

- `TinyTransformerBlock`
- `FeedForward` layer
- Attention -> residual -> feed-forward -> residual path
- ReLU activation
- Shape tests
- Residual tests
- Causal behavior smoke tests

Goal achieved:

> Manas has a tiny custom transformer block implemented from scratch.

---

### v0.6 — Transformer-Assisted Prediction and Generation

The transformer block was connected experimentally to prediction and generation.

Completed:

- `--use-transformer` flag for `predict-next`
- `--use-transformer` flag for `generate`
- Transformer forward path over token embeddings
- Last-token transformer output used for vocabulary scoring
- Conservative hybrid scoring with existing stable predictor

Goal achieved:

> Manas can route prediction and generation through an experimental transformer-assisted path without breaking the default system.

---

### v0.7 — Transformer Output-Head Training

The transformer-assisted path started learning through a trainable output head.

Completed:

- `--train-transformer` flag for `train-language`
- Transformer output projection training
- Cross-entropy-style next-token objective for output head
- Transformer state persistence
- Reload test verified in a new terminal

Example behavior:

```text
predict-next "Rust is a" --use-transformer -> systems
```

Goal achieved:

> The transformer path is no longer only connected; it now receives trained next-token signal through its output head.

---

### v0.7.1 — Controlled Neuron Growth

Language training previously caused uncontrolled neuron growth when the same text was trained again.

Fixed:

- Re-training the same text no longer grows hundreds of new neurons
- New text still grows a small bounded number of neurons
- Brain size stays smaller and more stable
- Transformer prediction still works after optimization

Goal achieved:

> Manas can keep learning without exploding neuron count on repeated language-training runs.

### v0.7.2 — Better Inspect for Language and Transformer State

Inspect was updated to show the full Manas system state clearly.

Completed:

- `manas inspect` now shows 5 separate sections: Core Network, Language System, Transformer, Storage, and Total
- Transformer param counting (attention, FFN, output head)
- Sequence memory status and entry count
- Sidecar file size reporting for all sidecars
- Language metadata: unique texts and repeated training counts
- `--verbose` flag for extended output
- Renamed labels: "Neurons" → "Core neurons", "Layers" → "Core network layers"
- Old stats continue to work correctly when sidecars are missing

Goal achieved:

> `manas inspect` accurately shows the full Manas system state including language, transformer, and sidecar visibility.

---

### v0.8 — Train Transformer Feed-Forward Layer

The FeedForward layer inside the transformer block is now trained alongside the output head.

Completed:

- `FeedForward::train_step()` — full forward-cache, backprop through w1/b1/w2/b2, ReLU derivative, gradient clipping [-1, 1], NaN/inf safety
- `TinyTransformerBlock::forward_with_ffn_inputs()` — returns per-position FFN inputs for backprop
- `TransformerLanguageModel` gains `ffn_trained: bool` field
- `TRANSFORMER_FILE_VERSION` bumped to 2 with FFN weight persistence
- `TransformerPredictor::from_model()` copies the trained block instead of rebuilding it
- `train_transformer_output_head()` now trains both output head AND FFN
- `manas inspect` shows `FFN trained : yes/no`
- 5 new tests (A: FFN weights change, B: attention stays frozen, C: prediction works, D: generation works, E: persistence roundtrip)
- Attention Q/K/V/O remain frozen

Example behaviour:

```text
$ manas train-language "text" --train-transformer
# Now trains both output head and FFN weights
$ manas inspect
  Output head trained : yes
  FFN trained         : yes
```

Goal achieved:

> The transformer FFN layer learns from next-token signal through backpropagated gradients while attention remains frozen.

---

---

### v0.8.1 — Transformer Training Metrics

Transformer training now reports detailed metrics instead of a single loss number.

Completed:

- `TransformerTrainReport` struct with epochs, examples, loss, accuracy, and status fields
- Per-epoch loss tracking (first, final, average)
- Loss improvement percentage (safe with zero first-loss)
- Top-1 and top-3 accuracy computed after training via transformer logits
- Invalid/NaN update counting for gradient safety
- Output head, FFN, and attention status in report
- Formatted CLI output with all metrics
- 5 new tests (A: report populated, B: accuracy math, C: improvement calc, D: zero-loss safe, E: format labels)

Example CLI output:

```text
Transformer training
  epochs                         : 100
  examples                       : 10
  language lr                    : 0.0500
  transformer lr                 : 0.0100
  avg train loss                   : 0.1234
  first epoch loss                 : 0.4567
  final epoch loss                 : 0.0234
  improvement                      : 94.88%
  pure transformer top-1 accuracy  : 80.00%
  pure transformer top-3 accuracy  : 100.00%
  output head                    : trained
  feed-forward                   : trained
  attention                      : frozen
  invalid updates                : 0
```

Goal achieved:

> Transformer training is now measurable with per-epoch loss, accuracy, improvement, and status reporting.

---

## Next Milestones

## v0.8.2 — Safer Transformer Training

Before training deeper parts of the transformer, training safety should improve.

### Planned Safety Features

- Gradient clipping
- NaN detection
- Infinite-value detection
- Loss explosion guard
- Learning-rate safety checks
- Optional rollback if training corrupts state

### Goal

> Make transformer training safer before attention weights are trained.

---

## v0.9 — Train Attention Projections

After FFN training is stable, train the attention projection matrices.

### Planned Trainable Weights

- `w_q`
- `w_k`
- `w_v`
- `w_o`

### Scope

- Single-head attention only
- Small context windows
- Small vocab first
- Gradient clipping required
- Strong tests required

### Goal

> Make the transformer learn context and token order more deeply instead of relying mostly on sequence memory.

---

## v0.9.1 — Improve Transformer Score Weight

Currently the system uses a conservative hybrid score so transformer experiments do not break generation.

Example:

```text
final_score = 0.60 * existing_hybrid_score + 0.40 * transformer_score
```

After transformer training improves, slowly increase transformer contribution.

Possible future setting:

```text
final_score = 0.40 * existing_hybrid_score + 0.60 * transformer_score
```

### Goal

> Move more prediction responsibility from memory shortcut to the trained transformer path.

---

## v1.0 — Stable Mini Local Language Model Release

This is the first stable language milestone.

### Should Include

- `train-language`
- `predict-next`
- `generate`
- `--use-transformer`
- `--train-transformer`
- Controlled neuron growth
- Persistent brain + sidecars
- Better inspect output
- Transformer output-head training
- FFN training if stable
- Clean README examples
- Strong tests

### Honest Claim

> Manas can locally learn small text sequences and generate text using a custom Rust memory + transformer-assisted language path.

### Not a Claim

Manas should not claim to be a ChatGPT replacement.

---

## v1.1 — Better Tokenizer, Casing, and Punctuation

Current generation normalizes text heavily.

Example:

```text
Rust is -> rust is
Ownership -> ownership
```

### Planned Improvements

- Case preservation
- Punctuation tokens
- Apostrophe handling
- Sentence boundary tokens
- Unknown-token handling
- Better decode behavior
- Optional special tokens:
  - `<BOS>`
  - `<EOS>`
  - `<UNK>`

### Goal

> Make generated text look more natural and preserve original formatting better.

---

## v1.2 — Language Training from Files and Folders

Add file/folder training for the language model.

### Planned Commands

```bash
manas train-language-file ./docs/rust.md --train-transformer
manas train-language-folder ./docs --train-transformer
```

### Goals

- Train language sequences from real documents
- Preserve source metadata
- Avoid uncontrolled growth
- Support repeated file training safely
- Track file fingerprints

---

## v1.3 — Retrieval + Generation with Sources

This milestone connects Manas memory and language generation.

### Planned Pipeline

```text
user question
-> retrieve relevant source/chunk memory
-> build generation context
-> generate answer
-> show source information
```

### Goals

- Use local learned knowledge during answers
- Show where answer knowledge came from
- Combine source-aware memory with generation
- Move beyond simple sequence replay

---

## v1.4 — Dynamic Transformer Growth

This is one of the most important long-term Manas ideas.

### Planned Growth Behaviors

- If loss stays high, grow FFN hidden dimension
- If a new topic/source appears, add specialized memory neurons
- If context is insufficient, increase max context safely
- If confidence is low, later add another attention head
- If repeated training is stable, avoid unnecessary growth

### Goal

> Make Manas a controlled self-growing transformer-style local AI system.

---

## v1.5 — Benchmarks, Tests, Docs, and Demo Scripts

Add stronger project quality before larger releases.

### Planned Work

- Accuracy tests
- Generation quality tests
- Brain size tests
- Neuron growth tests
- Training speed tests
- Memory usage tests
- Persistence tests
- CLI demo scripts
- README cleanup
- Architecture diagrams
- Example datasets

### Goal

> Make Manas easier to test, explain, benchmark, and demonstrate.

## Long-Term Direction

After v1.5, possible future directions:

- Multi-head attention
- Multiple transformer blocks
- Better optimizer
- Full transformer backprop
- Source-aware retrieval-augmented generation
- Local document question-answering
- Dynamic architecture growth
- Smaller/faster brain serialization
- Web UI or TUI
- Evaluation suite

## Principles

Manas should continue following these principles:

1. **Local first** — learning and memory should work locally.
2. **Custom Rust implementation** — no Candle, Hugging Face, burn, tch, or external ML framework for core learning.
3. **Source aware** — learned knowledge should preserve where it came from.
4. **Controlled growth** — Manas should grow, but not explode.
5. **Honest claims** — Manas is experimental and should not be marketed as a ChatGPT replacement.
6. **Small safe milestones** — every version should be testable before moving forward.

## Immediate Next Step

The next coding milestone is:

```text
v0.8.2 — Safer Transformer Training
```
