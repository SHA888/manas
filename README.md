# Manas — Your Personal AI Brain

> *मनस् (manas) — mind, intellect, the seat of thought*

**Manas is a self-growing neural network that learns from text, files, and the internet — 100% locally, one piece of knowledge at a time.**

Current AI models are pre-trained on fixed datasets, frozen in size, cloud-dependent, and disconnected from the present. Manas solves all five problems at once:

| Problem | Manas Solution |
|---|---|
| Can't learn after training | **Online learning** — learns from any input in real time |
| Fixed parameter count | **Dynamic growth** — adds neurons when needed |
| Cloud dependent | **100% local** — runs on your laptop, no API keys |
| Catastrophic forgetting | **Importance scoring** — protected neurons never overwritten |
| Stale knowledge | **Freshness system** — auto re-searches outdated knowledge |

---

## Quick Start

```bash
# Learn from text
manas learn "Rust is a systems programming language with zero-cost abstractions"

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

# Set freshness category
manas tag "Rust version" --freshness fast
```

---

## Architecture

Manas is built from 7 Rust crates, each with a single responsibility:

```
┌─────────────────────────────────────────────────────────────┐
│                         manas-cli                            │
│   learn | query | ingest | refresh | inspect | export ...    │
└───────────────────────────┬─────────────────────────────────┘
                            │
          ┌─────────────────┼─────────────────┐
          ▼                 ▼                  ▼
  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
  │ manas-ingest │  │ manas-agent  │  │ manas-memory │
  │ text/files/  │  │ web search   │  │ importance   │
  │ folders/urls │  │ html scrape  │  │ protection   │
  │ 7 parsers    │  │ freshness    │  │ compression  │
  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
         └─────────────────┼─────────────────┘
                           ▼
                  ┌──────────────────┐
                  │   manas-learn    │
                  │ tokenizer → embed│
                  │ → forward → loss │
                  │ → backprop → grow│
                  └────────┬─────────┘
                           ▼
                  ┌──────────────────┐
                  │   manas-core     │
                  │ neurons, layers  │
                  │ forward pass     │
                  │ weight updates   │
                  └────────┬─────────┘
                           ▼
                  ┌──────────────────┐
                  │  manas-store     │
                  │ .manas binary    │
                  │ append-only I/O  │
                  │ CRC32 integrity  │
                  └──────────────────┘
```

### Crates

| Crate | Purpose |
|---|---|
| **manas-core** | Neural network engine — Neuron, Layer, Network structs, forward pass, growth logic |
| **manas-store** | Custom `.manas` binary format — append-only read/write, CRC32 checksums |
| **manas-learn** | Online learning — tokenizer, embedder, backpropagation, loss-driven growth, **decoder** |
| **manas-ingest** | Input pipeline — 7 file format parsers, folder walker, text chunking |
| **manas-memory** | Never-forget system — importance scoring, protection levels, compression |
| **manas-agent** | Internet connection — DuckDuckGo search, HTML scraping, freshness checker |
| **manas-cli** | Command-line interface — 13 commands for all operations |

---

## How It Works

### Learning

```
Learning:
  Input text → Tokenize → Embed → Forward pass
    → Calculate MSE loss → Backpropagate → Update weights
    → If loss > threshold: grow a new neuron
    → For files/internet: grow 1 source-owned neuron per unique source
    → Recalculate importance scores → Save to .manas file

Inference (decoding):
  Query text → Tokenize → Embed → Forward pass
    → Output vector → Nearest tokens in embedding table
    → Display closest known tokens with similarity scores
```

### The Neuron

Each neuron is the atomic unit of knowledge:
- **Weights** — learned connection strengths
- **Importance score** — how valuable this knowledge is (0.0–1.0)
- **Protection level** — Open (learn freely), Guarded (small updates), Frozen (never touch)
- **Freshness category** — Timeless (never stale), Slow (30d), Fast (7d), Realtime (1d); set once alongside source, never overwritten
- **Source** — where the knowledge came from (text, file, internet); set once alongside freshness, never overwritten

### Never Forgetting

```
Importance = 0.40 × activation_frequency
           + 0.30 × recency_score
           + 0.20 × weight_magnitude
           + 0.10 × age_grace

Score ≥ 0.85 → Frozen. Never modified. Core knowledge is permanent.
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

# Querying
manas query "question"                    Search web + learn + display results
manas refresh --category cat              Refresh stale knowledge from web

# Inspection
manas inspect                             Show brain stats
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

Your brain lives on your machine. It starts at ~1 KB and grows as you teach it — forever.

---

## License

Licensed under either of:

- MIT license ([LICENSE-MIT](LICENSE-MIT))
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))

at your option.
