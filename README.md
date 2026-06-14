# Manas

> *मनस् (manas) — mind, intellect, the seat of thought*

**Manas is an experimental self-growing local AI system that learns from text, files, and the internet — 100% locally, one piece of knowledge at a time.**

> **Experimental Project Notice**
>
> Manas is an experimental Rust project exploring local-first learning, dynamic neural growth, source-aware memory, parameter tracking, and persistent `.manas` brain files.
>
> It is not a production-ready AI system and not a replacement for large language models.
>
> The project is currently a research/learning prototype. Features like self-growth, file ingestion, source-aware neurons, freshness tracking, and persistent memory are early implementations and may change over time.

| Area | Approach |
|---|---|
| Learning | **Online learning** — learns from any input in real time |
| Capacity | **Dynamic growth** — adds neurons when needed |
| Infrastructure | **100% local** — runs on your laptop, no API keys |
| Memory | **Importance scoring** — designed to preserve learned knowledge |
| Knowledge | **Freshness system** — re-searches outdated knowledge |

---

## Quick Start

```bash
# Learn from text
manas learn "Rust is a systems programming language with zero-cost abstractions"

# Train next-token prediction (v0.2)
manas train-language "Rust is a systems programming language" --epochs 50

# Train next-token prediction with transformer output head + FFN + attention w_o/w_v/w_q/w_k (v0.7-v0.9.3)
manas train-language "Rust is a systems programming language" --epochs 50 --train-transformer

# Train with growth control (v0.7.1) — cap new neurons, or disable growth entirely
manas train-language "Rust is a systems programming language" --epochs 50 --max-new-neurons 5
manas train-language "Duplicate text" --epochs 50 --no-grow

# Predict the next word (default: hybrid memory + neural)
manas predict-next "Rust is a" --top-k 5

# Predict next word with experimental transformer assistance (v0.6)
manas predict-next "Rust is a" --use-transformer --top-k 5

# Generate text (autoregressive, default: stable v0.3)
manas generate "Rust is a" --max-tokens 10

# Generate text with experimental transformer assistance (v0.6)
manas generate "Rust is a" --use-transformer --max-tokens 10

# Learn from files and folders
manas ingest --folder ./my-notes/
manas ingest --file ./article.md

# Learn from the internet
manas ingest --url https://doc.rust-lang.org/book/

# Query the web and learn automatically
manas query "latest Rust version features"

# See brain statistics
manas inspect

# Keep knowledge fresh
manas refresh --category fast

# List all ingested files
manas files

# Show activated neurons + decoded keywords for a topic
manas trace "Rust ownership"

# Show neurons with their source metadata
manas neurons --all

# Set freshness category
manas tag "Rust version" --freshness fast
```

---

## Architecture

Manas is built from 7 Rust crates, each with a single responsibility:

```
┌─────────────────────────────────────────────────────────────┐
│                         manas-cli                            │
│   learn | query | ingest | predict-next | generate | ...    │
└───────────────────────────┬─────────────────────────────────┘
                            │
          ┌─────────────────┼─────────────────┬───────────────┐
          ▼                 ▼                  ▼               ▼
  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
  │ manas-ingest │  │ manas-agent  │  │ manas-memory │  │manas-language│
  │ text/files/  │  │ web search   │  │ importance   │  │ next-token   │
  │ folders/urls │  │ html scrape  │  │ protection   │  │ prediction   │
  │ 7 parsers    │  │ freshness    │  │ compression  │  │ seq memory   │
  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
         └─────────────────┼─────────────────┼─────────────────┘
                           ▼                 ▼
                  ┌──────────────────────────────────┐
                  │           manas-learn             │
                  │ tokenizer → embed → forward → loss│
                  │ → backprop → grow → tag_neurons() │
                  └────────────────┬─────────────────┘
                                   ▼
                  ┌──────────────────────────────────┐
                  │           manas-core              │
                  │ neurons, layers, forward pass     │
                  │ weight updates, growth logic      │
                  └────────────────┬─────────────────┘
                                   ▼
                  ┌──────────────────────────────────┐
                  │          manas-store              │
                  │ .manas binary, append-only I/O    │
                  │ CRC32 integrity, brain.manas.seq  │
                  └──────────────────────────────────┘
```

### Crates

| Crate | Purpose |
|---|---|
| **manas-core** | Neural network engine — Neuron, Layer, Network structs, forward pass, growth logic |
| **manas-store** | Custom `.manas` binary format — append-only read/write, CRC32 checksums |
| **manas-learn** | Online learning — tokenizer, embedder, backpropagation, loss-driven growth, **decoder** |
| **manas-ingest** | Input pipeline — 7 file format parsers, folder walker, text chunking |
| **manas-memory** | Knowledge preservation — importance scoring, protection levels, compression |
| **manas-agent** | Internet connection — DuckDuckGo search, HTML scraping, freshness checker |
| **manas-language** | Next-token prediction — sequence memory, hybrid memory+neural predictor, autoregressive generation, custom transformer block with trainable output head, FFN, and partial attention `w_o`/`w_v`/`w_q`/`w_k` training |
| **manas-cli** | Command-line interface — 16 commands for all operations |

---

## How It Works

### Learning

```
Learning:
  Input text → Tokenize → Embed → Forward pass
    → Calculate MSE loss → Backpropagate → Update weights
    → If loss > threshold: grow a new neuron
    → For files/internet: grow 1 source-owned neuron per unique source
    → Tag neurons with source + freshness (only if Unknown)
    → Recalculate importance scores → Save to .manas file

Inference (decoding):
  Query text → Tokenize → Embed → Forward pass
    → Output vector → Nearest tokens in embedding table
    → Display closest known tokens with similarity scores

Next-token prediction (v0.2):
  Input text → Tokenize → Build sequence examples (sliding window)
    → For each (context, target):
      → Embed context → Forward pass → Loss → Backprop
      → Record transition in SequenceMemory (including suffix contexts)
    → After training: hybrid prediction
      → 0.8 × memory_score + 0.2 × neural_score
      → Context-token penalization
      → Predict next token or generate autoregressively
```

### The Neuron

Each neuron is the atomic unit of knowledge:
- **Weights** — learned connection strengths
- **Importance score** — how valuable this knowledge is (0.0–1.0)
- **Protection level** — Open (learn freely), Guarded (small updates), Frozen (never touch)
- **Freshness category** — Timeless (never stale), Slow (30d), Fast (7d), Realtime (1d); set once alongside source, never overwritten
- **Source** — where the knowledge came from (text, file, internet); set once alongside freshness, never overwritten

### Knowledge Preservation

```
Importance = 0.40 × activation_frequency
           + 0.30 × recency_score
           + 0.20 × weight_magnitude
           + 0.10 × age_grace

Score ≥ 0.85 → Frozen. Protected from modification. Core knowledge is preserved.
Score ≥ 0.60 → Guarded. Small updates only (clamped deltas).
Score < 0.60 → Open. Full learning allowed.
Score < 0.10 → Compress candidate. Merged into archive (never deleted).
```

### Staying Fresh

Knowledge is categorized by freshness:

| Category | Refresh After | Examples |
|---|---|---|
| Timeless (0) | Never | Math, logic, language rules |
| Slow (1) | 30 days | History, geography |
| Fast (2) | 7 days | Tech, software versions |
| Realtime (3) | 1 day | News, prices, events |

Auto-detected from keywords in the text. Stale neurons trigger automatic internet re-search.

---

## Current Capabilities

- **Learn from raw text** — tokenizes, embeds, forward pass, backprop, grows neurons as needed
- **Ingest local files** — 7 format parsers (txt, md, json, html, csv, yaml, toml), folder walker, text chunking
- **Persist state** — stores vocab, embeddings, neurons, and metadata in a single `.manas` file
- **Source-aware growth** — grows a dedicated neuron per unique file or URL, retaining provenance
- **Source metadata on all neurons** — every neuron (including language-trained ones) is stamped with `src=raw text`, `src=file:...`, or `src=url:...`; never overwritten once set
- **Parameter tracking** — reports network params, embedding params, and total params
- **Inspection commands** — `inspect`, `trace`, `neurons`, `files` give visibility into the network's state
- **Freshness system** — categorizes knowledge (timeless/slow/fast/realtime) and flags stale neurons
- **Web search & scrape** — queries DuckDuckGo, scrapes HTML, and ingests results
- **Next-token prediction (v0.2)** — `train-language`, `predict-next`, `generate` commands with hybrid sequence memory + neural predictor
- **Single-head causal attention (v0.4)** — custom `CausalSelfAttention` module with QKV projections, scaled dot-product, and causal masking; not yet integrated into generation by default
- **Tiny transformer block (v0.5)** — `TinyTransformerBlock` combining causal attention + feed-forward with residual connections; experimental, not yet the default predictor
- **Transformer-assisted prediction (v0.6)** — `--use-transformer` flag for `predict-next` and `generate`; hybrid scoring (75% memory+neural, 25% transformer); experimental, default path unchanged
- **Transformer output-head training (v0.7)** — `--train-transformer` flag for `train-language`; cross-entropy training of output projection head; dynamic weighting (40% transformer when trained); block weights frozen
- **Neural growth optimization (v0.7.1)** — `--max-new-neurons` / `--no-grow` flags; growth capped per call and restricted to first epoch only; duplicate-text detection via `LanguageMeta` sidecar (`brain.manas.langmeta`) prevents re-growth on repeated training
- **Enhanced system inspect (v0.7.2)** — `manas inspect` now shows separate sections for Core Network, Language System, Transformer, Storage, and Total; reports sidecar file sizes, transformer param counts, sequence memory status, and language metadata; `--verbose` flag for extended output
- **Transformer FFN training (v0.8)** — `--train-transformer` now trains both the output head and the FeedForward layer inside the transformer block; gradient clipping to [-1, 1], NaN/inf safety; attention Q/K/V/O remain frozen; `manas inspect` reports `FFN trained : yes/no`
- **Transformer training metrics (v0.8.1)** — `--train-transformer` now prints detailed metrics: per-epoch loss, pure transformer top-1/top-3 accuracy, loss improvement %, invalid update count, output head/FFN/attention status. Separate `--transformer-learning-rate` flag (default 0.01). `--transformer-only` flag on `predict-next` for pure-transformer debug predictions.
- **Safer transformer training (v0.8.2)** — norm-based gradient clipping, loss explosion detection, instability rollback, pre-save finite check, separate "Training safety" output block. CLI flags: `--transformer-max-grad-norm`, `--transformer-max-loss`, `--no-transformer-rollback`.
- **Attention cache + persistence prep (v0.9.0)** — `CausalSelfAttention::forward_with_cache()` now exposes Q/K/V, causal attention weights, and weighted values for future backprop. Transformer sidecar version 3 persists attention weights and `attention_trained`; old v2 transformer files still load with deterministic untrained attention. `is_finite_model()` now checks attention weights, and `manas inspect` reports `Attention trained : yes/no`.
- **Attention output projection training (v0.9.1)** — `--train-transformer` now trains only `CausalSelfAttention.w_o` using the cached weighted value vector and the gradient flowing into the attention output. `w_q`, `w_k`, and `w_v` remain frozen; there is no softmax/QK backprop, scoring change, generation change, or sidecar version bump. Training and inspect report partial attention as `Attention trained : partial` and `Attention projections : o`.
- **Attention value projection training (v0.9.2)** — `--train-transformer` now also trains `CausalSelfAttention.w_v` from cached final-position attention probabilities and the context gradient. `w_q` and `w_k` remain frozen; no softmax/QK backprop, scoring change, generation change, model-size change, or sidecar version bump. Transformer sidecar v3 stores an optional projection bitmask so inspect can report `Attention projections : o,v` while legacy v3 files still load as `o`.
- **Attention query/key projection training (v0.9.3)** — `--train-transformer` now trains `CausalSelfAttention.w_q` and `CausalSelfAttention.w_k` together through the final-position causal softmax gradient. Output head, FFN, `w_o`, and `w_v` continue training; scoring weights, generation behavior, model dimensions, and sidecar version remain unchanged. Inspect reports partial attention as `Attention projections : o,v,q,k`.

## Current Limitations

- **Query output is not local-first yet** — currently relies on web search rather than answering from the local network alone
- **Answer generation is basic** — there is no generative text output; decoded tokens show the closest embeddings
- **Next-token prediction is experimental** — v0.2 works for short contexts but is not trained on large corpora; generation quality is limited
- **Attention is experimental (v0.4/v0.9.3)** — single-head causal attention is implemented with forward-cache, persistence, and partial `w_o`/`w_v`/`w_q`/`w_k` training; multi-head attention, layer norm, and dynamic transformer growth are not implemented
- **Transformer block is experimental (v0.5+)** — `TinyTransformerBlock` supports trained output-head, FFN, and single-head attention projection training, but it is still a tiny custom research block rather than a full LLM stack
- **Transformer-assisted prediction is experimental (v0.6-v0.9.3)** — `--use-transformer` uses the trained output head, FeedForward layer, and partial attention projection training when available; scoring weights and default prediction/generation behavior are unchanged
- **Growth control is experimental (v0.7.1)** — `max_new_neurons` cap and first-epoch-only growth help control network explosion; duplicate-text detection via `LanguageMeta` sidecar prevents re-growth on repeated training but is not retroactive
- **File/chunk learning is experimental** — chunking heuristics and per-chunk learning are still being refined
- **One neuron per source is an anchor** — the source neuron acts as a pointer, not a full document understanding
- **Not production-ready** — this is a research prototype; APIs, storage, and behavior may change

---

## The .manas File Format

A single file stores the entire brain:

```
[FILE HEADER]     64 bytes — magic, version, timestamps, counts
[VOCAB BLOCK]     Variable — token string table + embeddings
[LAYER INDEX]     Variable — byte offsets for each layer
[LAYER BLOCK] × N Each layer's neuron data
[ARCHIVE BLOCK]   Compressed/merged old neurons (restorable)
[CHECKSUM]        4 bytes CRC32
```

Append-only — new neurons are added without rewriting the whole file. Starts at ~1 KB, grows forever.

---

## CLI Reference

```bash
# Learning
manas learn "text"                       Learn from raw text
manas ingest --file path                  Learn from a file
manas ingest --folder path                Learn from a folder (recursive)
manas ingest --url URL                    Learn from a web page
manas ingest --dry-run                    Preview without learning

# Language (v0.2)
manas train-language "text"              Train next-token prediction
  --epochs 50                            Training epochs
  --learning-rate 0.05                   Learning rate
  --max-context 5                        Sliding context window size
  --max-new-neurons 10                   Max new neurons to grow (v0.7.1)
  --no-grow                              Disable all neuron growth (v0.7.1)
  --train-transformer                    Train output head + FFN + attention w_o/w_v/w_q/w_k (v0.9.3)

manas predict-next "context"             Predict next token(s)
  --top-k 5                              Number of candidates
  --max-context 5                        Context window

manas generate "prompt"                  Generate text autoregressively
  --max-tokens 20                        Tokens to generate
  --max-context 5                        Context window
  --top-k 1                              Candidates considered (top-1 is deterministic)
  --temperature 1.0                      Sampling temperature (reserved)

# Querying
manas query "question"                    Search web + learn + display results
manas refresh --category cat              Refresh stale knowledge from web

# Inspection
manas inspect                             Show brain stats with full system state (v0.7.2)
manas inspect --verbose                   Extended verbose output (v0.7.2)
manas files                               List ingested files
manas trace "topic"                       Show activated neurons + decoded keywords
manas neurons --all                       List all neurons with metadata

# Management
manas export --out file                   Export brain
manas import --file path                  Import brain
manas verify                              Check file integrity
manas restore --all                       Restore archived neurons
manas tag "topic" --freshness cat         Set freshness category
```

---

## Installation

### Prerequisites

- Rust 2024 edition (Rust 1.85+)
- Cargo

### Build from source

```bash
git clone https://github.com/AarambhDevHub/manas.git
cd manas
cargo build --release
./target/release/manas --help
```

---

## Project Structure

```
manas/
├── Cargo.toml                  Workspace root
├── README.md                   This file
├── LICENSE-MIT                 MIT license
├── LICENSE-APACHE              Apache 2.0 license
├── .gitignore                  Git ignore rules
├── ARCHITECTURE.md             Full system design document
├── teach/                      Teaching files (user-created)
├── manas-core/                 Neural network engine
├── manas-memory/               Importance & protection system
├── manas-store/                .manas file format
├── manas-learn/                Online learning engine
├── manas-language/             Next-token prediction & sequence memory
├── manas-ingest/               Input pipeline
├── manas-agent/                Internet agent
├── manas-cli/                  Command-line interface
└── manas-benches/              Performance benchmarks
```

---

## Benchmarks

All benchmarks run in release mode on a standard laptop:

| Operation | Time |
|---|---|
| Tokenize (short text) | 0.27 µs |
| Tokenize (long text) | 1.26 µs |
| Forward pass (2 layers, 80 neurons) | 1.20 µs |
| Forward pass (3 layers, 448 neurons) | 47.99 µs |
| Backprop (2 layers, 80 neurons) | 9.04 µs |
| Learn (short text, full cycle) | 21.80 µs |
| Save to .manas | 139.80 µs |
| Load from .manas | 9.95 µs |

---

## Philosophy

- **Local ownership** over cloud dependency
- **Lifelong learning** over frozen models
- **Growth** over capacity limits
- **Preservation** over overwriting
- **Freshness** over staleness
- **Simplicity** over complexity

Your local network lives on your machine. It starts at ~1 KB and grows as you teach it.

---

## License

Licensed under either of:

- MIT license ([LICENSE-MIT](LICENSE-MIT))
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))

at your option.
