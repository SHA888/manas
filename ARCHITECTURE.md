# ARCHITECTURE.md — Manas

> **"A self-growing local learning system — designed to preserve learned knowledge"**
>
> Manas (Sanskrit: *मनस्* — mind, intellect, the seat of thought) is an experimental
> self-growing local AI system written in Rust that starts with zero knowledge, learns
> from text, local files, and the internet, and is designed to preserve learned knowledge.

---

## Table of Contents

1. [Vision](#1-vision)
2. [Core Principles](#2-core-principles)
3. [System Overview](#3-system-overview)
4. [Full Architecture Diagram](#4-full-architecture-diagram)
5. [Crate Structure](#5-crate-structure)
6. [Crate Details](#6-crate-details)
   - [manas-core](#61-manas-core)
   - [manas-memory](#62-manas-memory)
   - [manas-store](#63-manas-store)
   - [manas-learn](#64-manas-learn)
   - [manas-ingest](#65-manas-ingest)
   - [manas-agent](#66-manas-agent)
   - [manas-cli](#67-manas-cli)
7. [The .manas Binary Format](#7-the-manas-binary-format)
8. [The Growth System](#8-the-growth-system)
9. [The Memory Importance System](#9-the-memory-importance-system)
10. [The Freshness System](#10-the-freshness-system)
11. [The Local File Ingestion System](#11-the-local-file-ingestion-system)
12. [The Internet Agent System](#12-the-internet-agent-system)
13. [Data Flow — Full Pipeline](#13-data-flow--full-pipeline)
14. [Neuron Lifecycle](#14-neuron-lifecycle)
15. [Error Handling Strategy](#15-error-handling-strategy)
16. [Milestone Plan](#16-milestone-plan)
17. [CLI Reference](#17-cli-reference)
18. [Future Vision](#18-future-vision)

---

## 1. Vision

Current AI models are:

- Pre-trained on fixed datasets — they cannot learn new things after training
- Static in size — their parameter count is frozen forever
- Cloud-dependent — they require expensive APIs or servers
- Forgetful — fine-tuning on new data destroys old knowledge (catastrophic forgetting)
- Disconnected from the present — their knowledge has a hard cutoff date

**Manas approaches these challenges as follows:**

| Challenge | Manas Approach |
|---|---|
| Learning after training | Online learning — learns from any input in real time |
| Fixed parameter count | Dynamic growth — adds neurons when needed |
| Cloud dependency | 100% local — runs on your laptop |
| Catastrophic forgetting | Importance scoring — designed to preserve learned knowledge |
| Stale knowledge | Freshness system — re-searches outdated knowledge |

The end result is a local network that lives on your machine in a single `.manas` file,
starts at ~1 KB, and grows as you teach it.

---

## 2. Core Principles

### Principle 1 — Preserve Knowledge
Knowledge is persisted and protected against accidental overwriting.
If a neuron must change, its old state is archived before updating.

### Principle 2 — Grow When Needed
The network never hits a capacity ceiling. When it cannot represent something well
(measured by loss), it grows a new neuron instead of forcing existing neurons to compromise.

### Principle 3 — Stay Fresh
All time-sensitive knowledge has a timestamp and a freshness category. Stale knowledge
triggers a silent internet re-search before being used to answer.

### Principle 4 — Learn from Anywhere
Text comes from three sources and all are treated equally by the learning pipeline:
- Raw text typed by the user
- Local files and folders on disk
- Live internet pages fetched by the agent

### Principle 5 — Full Local Ownership
The model is a single `.manas` file on disk. No cloud, no account, no API key required
for inference. The user owns their local network completely.

---

## 3. System Overview

```
Input Sources
─────────────
  [Raw Text]  [Local Files / Folders]  [Internet]
       │               │                   │
       └───────────────┼───────────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  manas-ingest   │  ← unified input pipeline
              │  (normalize,    │     cleans and tokenizes
              │   tokenize,     │     all input sources
              │   tag source)   │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  manas-learn    │  ← online backpropagation
              │  (backprop,     │     one sample at a time
              │   loss calc,    │     signals core to grow
              │   growth signal)│     when loss is too high
              └────────┬────────┘
                       │
              ┌────────┴────────┬──────────────┐
              │                 │              │
              ▼                 ▼              ▼
    ┌──────────────┐   ┌──────────────────┐   ┌──────────────────┐
    │  manas-core  │   │  manas-memory    │   │ manas-language   │
    │  (neurons,   │   │  (importance     │   │ (next-token      │
    │   layers,    │   │   scoring,       │   │  prediction,     │
    │   growth,    │   │   protection,    │   │  sequence        │
    │   forward    │   │   compression)   │   │  memory,         │
    │   pass)      │   └──────────────────┘   │  hybrid pred.)   │
    └──────┬───────┘                           └────────┬─────────┘
           │                                            │
            └────────────────┬───────────────────────────┘
                             ▼
                   ┌────────────────┐
                   │  manas-store   │  ← custom .manas binary + .manas.seq,
                   │  (.manas file, │     .manas.transformer, .manas.langmeta
                   │   read/write,  │     append-only growth
                   │   append)      │     full neuron metadata
                   └────────────────┘
```

---

## 4. Full Architecture Diagram

```
┌────────────────────────────────────────────────────────────────────────┐
│                         Manas System                                   │
│                                                                        │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │                       manas-cli                               │     │
│  │   learn | query | ingest | predict-next | generate | inspect  │     │
│  └───────────────────────────┬───────────────────────────────────┘     │
│                               │                                        │
│          ┌────────────────────┼──────────────────┬─────────────────┐   │
│          │                    │                  │                 │   │
│          ▼                    ▼                  ▼                 ▼   │
│  ┌──────────────┐   ┌──────────────────┐  ┌──────────────┐  ┌─────────┐│
│  │ manas-ingest │   │   manas-agent    │  │ manas-memory │  │manas-   ││
│  │              │   │                  │  │              │  │language ││
│  │ • raw text   │   │ • web search     │  │ • importance │  │• next-  ││
│  │ • .txt .md   │   │ • html scrape    │  │   scoring    │  │  token  ││
│  │ • .rs .toml  │   │ • freshness      │  │ • neuron     │  │  pred.  ││
│  │ • .json .csv │   │   checker        │  │   protection │  │• seq    ││
│  │ • .pdf .html │   │ • feeds ingest   │  │ • compression│  │  memory ││
│  │ • folder     │   │   pipeline       │  │   of cold    │  │• hybrid ││
│  │   recursive  │   │                  │  │   neurons    │  │  predict││
│  └──────┬───────┘   └────────┬─────────┘  └──────┬───────┘  └────┬────┘│
│         │                    │                   │               │     │
│         └────────────────────┼───────────────────┼───────────────┘     │
│                              │                   │                     │
│                              ▼                   ▼                     │
│                   ┌──────────────────────────────────────┐             │
│                   │          manas-learn                 │             │
│                   │                                      │             │
│                   │ • tokenizer    • embedding           │             │
│                   │ • forward pass • loss calculation    │             │
│                   │ • backpropagation • growth signal    │             │
│                   │ • tag_neurons() — source + freshness │             │
│                   └──────────────────┬───────────────────┘             │
│                                      │                                 │
│                                      ▼                                 │
│                   ┌──────────────────────────────────────┐             │
│                   │            manas-core                │             │
│                   │                                      │             │
│                   │ • Neuron struct  • Layer struct      │             │
│                   │ • Network struct • forward()         │             │
│                   │ • grow_neuron()  • update_weights()  │             │
│                   └──────────────────┬───────────────────┘             │
│                                      │                                 │   
│                                      ▼                                 │   
│                   ┌──────────────────────────────────────┐             │
│                   │          manas-store                 │             │
│                   │                                      │             │
│                   │ • .manas file I/O  • header r/w      │             │
│                   │ • neuron append    • full load/save  │             │
│                   │ • integrity check  • .manas.seq I/O  │             │
│                   │ • .manas.transformer I/O              │             │
│                   │ • .manas.langmeta I/O                 │             │
│                   └──────────────────────────────────────┘             │
│                                      │                                 │
│         [brain.manas + brain.manas.seq + brain.manas.transformer + brain.manas.langmeta]                │
│                    starts: ~1 KB                                       │
│                    grows:  incrementally                               │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Crate Structure

```
manas/
├── Cargo.toml                  ← workspace root
├── ARCHITECTURE.md             ← this file
├── ROADMAP.md
├── README.md
│
├── manas-core/                 ← neurons, layers, growth, forward pass
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── neuron.rs
│       ├── layer.rs
│       ├── network.rs
│       └── activation.rs
│
├── manas-memory/               ← importance scoring, protection, compression
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── scorer.rs
│       ├── protector.rs
│       └── compressor.rs
│
├── manas-store/                ← .manas binary format, file I/O
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── format.rs
│       ├── reader.rs
│       ├── writer.rs
│       └── integrity.rs
│
├── manas-learn/                ← tokenizer, backprop, online learning loop, decoder
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── tokenizer.rs
│       ├── embedder.rs
│       ├── backprop.rs
│       ├── decoder.rs
│       └── trainer.rs
│
├── manas-language/         ← next-token prediction, seq memory, hybrid predictor
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── attention.rs    ← single-head causal self-attention (v0.4)
│       └── transformer.rs  ← tiny transformer block (v0.5)
│
├── manas-ingest/               ← unified input pipeline (text, files, folders)
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── raw_text.rs
│       ├── file_reader.rs
│       ├── folder_walker.rs
│       ├── format/
│       │   ├── markdown.rs
│       │   ├── plaintext.rs
│       │   ├── rust_source.rs
│       │   ├── json.rs
│       │   ├── toml.rs
│       │   ├── csv.rs
│       │   └── html.rs
│       └── normalizer.rs
│
├── manas-agent/                ← internet search, html scraping, freshness
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── searcher.rs
│       ├── scraper.rs
│       ├── freshness.rs
│       └── pipeline.rs
│
└── manas-cli/                  ← command line interface
    ├── Cargo.toml
    └── src/
        └── main.rs
```

---

## 6. Crate Details

### 6.1 `manas-core`

The neural network engine — the core network runtime.

#### Key Structs

```rust
/// A single neuron — the atomic unit of Manas
pub struct Neuron {
    pub id:               u64,
    pub weights:          Vec<f32>,
    pub bias:             f32,
    pub activation:       Activation,

    // memory metadata
    pub importance_score: f32,
    pub born_at:          u64,       // unix timestamp
    pub last_activated:   u64,       // unix timestamp
    pub activation_count: u64,       // total times this neuron has fired

    // knowledge metadata
    pub learned_at:           u64,   // when it first absorbed this knowledge
    pub last_verified:        u64,   // when it last re-checked internet
    pub freshness_category:   u8,    // 0=timeless 1=slow 2=fast 3=realtime
    pub source:               Source, // where the knowledge came from
    pub is_protected:         bool,  // if true, weights cannot be modified
}

/// A layer of neurons
pub struct Layer {
    pub id:         u32,
    pub neurons:    Vec<Neuron>,
    pub activation: Activation,
}

/// The full network
pub struct Network {
    pub layers:        Vec<Layer>,
    pub total_neurons: u64,
    pub created_at:    u64,
    pub version:       u8,
}

/// Where knowledge came from
pub enum Source {
    RawText,
    LocalFile { path: String },
    Internet  { url: String },
    Unknown,
}

/// Activation function type
pub enum Activation {
    ReLU,
    Sigmoid,
    Tanh,
    Linear,
}
```

#### Key Methods

```rust
impl Network {
    /// Run a forward pass — inference
    pub fn forward(&self, input: &[f32]) -> Vec<f32>;

    /// Add a new neuron to a specific layer
    pub fn grow_neuron(&mut self, layer_id: u32, input_size: usize);

    /// Add a completely new layer
    pub fn grow_layer(&mut self, neuron_count: usize);

    /// Update weights of a specific neuron (respects is_protected)
    pub fn update_weights(&mut self, neuron_id: u64, delta: &[f32]);

    /// Record that a neuron was activated (updates last_activated, count)
    pub fn record_activation(&mut self, neuron_id: u64);

    /// Total parameter count across all neurons
    pub fn parameter_count(&self) -> u64;
}
```

#### Growth Trigger Logic

```
After each forward pass → calculate MSE loss

IF loss > GROWTH_THRESHOLD (default: 0.35)
    AND we tried updating weights 3 times and loss stayed high
THEN
    find the layer with highest average loss contribution
    grow_neuron(that_layer_id, input_size)
    re-run forward pass
    save to .manas
```

---

### 6.2 `manas-memory`

The importance system. Decides which neurons matter and which can be compressed.

#### Importance Score Calculation

Each neuron's importance score is a weighted combination of:

```
importance = (
    0.40 * activation_frequency    +   // how often it fires
    0.30 * recency_score           +   // how recently it fired
    0.20 * weight_magnitude        +   // how large its weights are
    0.10 * age_score                   // newer neurons get a grace period
)
```

All values normalized to [0.0, 1.0].

#### Protection Rules

```rust
pub enum ProtectionStatus {
    /// Fully protected — no modification allowed
    Frozen,

    /// Softly protected — small updates only (max delta ±0.01)
    Guarded,

    /// Normal — full learning allowed
    Open,
}
```

A neuron is set to `Frozen` when:
- `importance_score > 0.85`
- `activation_count > 10_000`
- `is_protected == true` (manually set)

A neuron is set to `Guarded` when:
- `importance_score > 0.60`
- `age < 7 days` (newly born neurons, protected while settling)

#### Compression

When the model reaches a user-defined size limit:

1. Find all neurons with `importance_score < 0.10`
2. Cluster similar neurons by weight cosine similarity
3. Merge clusters into a single averaged neuron
4. Archive the originals in a separate `.manas.archive` section
5. Total neuron count decreases, but no knowledge is permanently destroyed

---

### 6.3 `manas-store`

The custom binary file format. Reads and writes `.manas` files.

#### File Layout

```
[FILE HEADER]         fixed 64 bytes
[LAYER INDEX]         variable — list of layer positions in file
[LAYER BLOCK] × N    each layer's neuron data
[ARCHIVE BLOCK]       compressed/merged old neurons
[CHECKSUM]            CRC32 of entire file
```

Full byte-level format is documented in [Section 7](#7-the-manas-binary-format).

#### Key Operations

```rust
pub struct ManasBrain {
    pub path: PathBuf,
}

impl ManasBrain {
    /// Load full network from .manas file
    pub fn load(&self) -> Result<Network>;

    /// Save full network to .manas file
    pub fn save(&self, network: &Network) -> Result<()>;

    /// Append a single new neuron without rewriting whole file
    pub fn append_neuron(&self, layer_id: u32, neuron: &Neuron) -> Result<()>;

    /// Update a single neuron's weights in-place
    pub fn update_neuron(&self, neuron_id: u64, neuron: &Neuron) -> Result<()>;

    /// Verify file integrity via checksum
    pub fn verify(&self) -> Result<bool>;

    /// Print human-readable stats
    pub fn inspect(&self) -> Result<BrainStats>;

    /// Save network + vocab + embeddings (full trainer state)
    pub fn save_with_vocab(&self, network: &Network, vocab: &HashMap<u32, (String, Vec<f32>)>) -> Result<()>;

    /// Load just the vocab + embeddings
    pub fn load_vocab(&self) -> Result<HashMap<u32, (String, Vec<f32>)>>;
}
```

#### Append-Only Growth Strategy

When a new neuron is grown, Manas does **not** rewrite the whole file.
It seeks to the end of the last layer block and appends the new neuron bytes,
then updates the layer's neuron count in the header.

This means `.manas` files grow incrementally with zero full-file rewrites.

---

### 6.4 `manas-learn`

The learning engine. Converts text into knowledge and updates the network.

#### Pipeline

```
Raw text string
      │
      ▼
 Tokenizer
 (split into tokens, lowercase, strip noise)
      │
      ▼
 Embedder
 (convert tokens to f32 vectors via simple lookup table)
      │
      ▼
 Forward pass  →  prediction vector
      │
      ▼
 Loss function  →  MSE loss value
      │
      ▼
 Backpropagation  →  weight deltas
      │
      ▼
 Growth check
    loss > threshold?
    YES → signal manas-core to grow neuron
    NO  → apply deltas directly
      │
      ▼
 manas-memory checks protection status before applying
      │
      ▼
 manas-store saves updated weights
```

#### Tokenizer

A lightweight tokenizer — no BPE, no external model.

```rust
pub struct Tokenizer {
    pub vocab: HashMap<String, u32>,   // token → id
    pub id_to_token: HashMap<u32, String>, // id → token (reverse lookup)
    pub vocab_size: u32,               // grows as new words are seen
}

impl Tokenizer {
    /// Tokenize a string → list of token ids
    pub fn encode(&mut self, text: &str) -> Vec<u32>;

    /// New token seen → add to vocab, grow vocab_size
    pub fn learn_token(&mut self, token: &str) -> u32;

    /// Reverse lookup: token id → word
    pub fn decode(&self, id: u32) -> Option<&str>;
}
```

Every new word the model sees is added to the vocab. The vocab is persisted in
the `.manas` file's `[VOCAB BLOCK]` alongside the trained embeddings.

#### Embedder

Maps token ids to f32 vectors.

```rust
pub struct Embedder {
    pub dim: usize,                          // embedding dimension, default 64
    pub table: HashMap<u32, Vec<f32>>,       // token_id → embedding vector
}
```

New tokens get a randomly initialized embedding vector when first seen.
The embedder is trained alongside the network via backpropagation. Embeddings
are persisted in the `.manas` file and restored on load.

#### TrainerSnapshot

The `TrainerSnapshot` is a serializable snapshot of the trainer's state (vocab +
embeddings). It bridges the gap between the in-memory trainer and the `.manas` file.

```rust
pub struct TrainerSnapshot {
    pub vocab: HashMap<String, u32>,
    pub id_to_token: HashMap<u32, String>,
    pub embed_table: HashMap<u32, Vec<f32>>,
    pub embed_dim: usize,
}

impl Trainer {
    /// Export current vocab + embeddings
    pub fn snapshot(&self) -> TrainerSnapshot;

    /// Restore vocab + embeddings from a snapshot
    pub fn restore(&mut self, snapshot: &TrainerSnapshot);
}
```

On save, `manas-store` writes the snapshot's entries into the `[VOCAB BLOCK]`.
On load, `manas-store` reads the block back and the CLI restores the trainer
before any `trace` or `decode` call. This ensures the decoder always uses the
trained embeddings rather than random initial values.

#### Decoder

Finds tokens whose embeddings are closest to the query's embedding vector — the semantic neighborhood of the input.

```rust
pub struct DecodeResult {
    pub tokens:      Vec<(String, f32)>,  // word + cosine similarity
    pub output_norm: f32,                 // magnitude of query embedding vector
}
```

```
Query → Tokenize → Embed → Average embedding
    → Cosine similarity with every token embedding
    → Sort by score → Return top 20 closest tokens
```

The decoder is used by `manas trace` to show the network's semantic response to a query:

```
$ manas trace "Manas self-growing neural network"

Top 10 activated neurons:
  n12     L0  act=0.2982  ...
  ...

Closest known tokens (decoded):
  network              sim=0.1499
  neural               sim=0.0818
  manas                sim=0.0268
```

---

### 6.5 `manas-ingest`

The unified input pipeline. Normalizes all input sources into clean text before
handing off to `manas-learn`. This is where **local file learning** lives.

#### Supported Input Sources

| Source | Handler | Notes |
|---|---|---|
| Raw string | `raw_text.rs` | Direct pass-through |
| `.txt` | `plaintext.rs` | Strip control chars |
| `.md` | `markdown.rs` | Strip markdown syntax, keep text |
| `.rs` | `rust_source.rs` | Extract doc comments + code structure |
| `.toml` | `toml.rs` | Convert key-value pairs to text |
| `.json` | `json.rs` | Flatten to key: value text |
| `.csv` | `csv.rs` | Row-by-row text with headers |
| `.html` | `html.rs` | Strip tags, extract visible text |
| folder | `folder_walker.rs` | Recursive walk, all supported files |

#### Local File Learning

```rust
pub enum IngestSource {
    Text(String),
    File(PathBuf),
    Folder(PathBuf),
    Url(String),         // handled by manas-agent, then re-enters here
}

pub struct IngestPipeline;

impl IngestPipeline {
    /// Ingest any source — returns stream of clean text chunks
    pub fn process(&self, source: IngestSource) -> impl Iterator<Item = TextChunk>;
}

pub struct TextChunk {
    pub text:       String,
    pub source:     Source,       // tracks origin for neuron metadata
    pub chunk_id:   u64,
    pub file_path:  Option<String>,
    pub url:        Option<String>,
}
```

#### Folder Walker

When a folder path is given, `folder_walker.rs` recursively walks the directory:

```
/my-notes/
├── rust-notes.md       ✅ ingest
├── ideas.txt           ✅ ingest
├── config.toml         ✅ ingest
├── data.csv            ✅ ingest
├── images/
│   └── diagram.png     ❌ skip (not text)
└── subdir/
    └── more-notes.md   ✅ ingest (recursive)
```

Each file is tagged with its path in the `Source::LocalFile { path }` field,
so neurons know where their knowledge came from.

#### CLI Usage for Local Learning

```bash
# Learn from a single file
manas ingest --file ./notes.md

# Learn from an entire folder recursively
manas ingest --folder ./my-notes/

# Learn from multiple sources at once
manas ingest --folder ./rust-docs/ --folder ./personal-notes/ --file ./extra.md

# Preview what would be ingested without actually learning
manas ingest --folder ./my-notes/ --dry-run
```

---

### 6.6 `manas-agent`

The internet connection. Searches the web, scrapes clean text, and manages freshness.

#### Web Search

```rust
pub struct Searcher;

impl Searcher {
    /// Search for a query and return top N result URLs
    pub async fn search(&self, query: &str, top_n: usize) -> Result<Vec<SearchResult>>;
}

pub struct SearchResult {
    pub url:     String,
    pub title:   String,
    pub snippet: String,
}
```

#### Scraper

```rust
pub struct Scraper;

impl Scraper {
    /// Fetch a URL and extract clean readable text
    pub async fn scrape(&self, url: &str) -> Result<String>;
}
```

The scraper:
1. Fetches the HTML with a standard HTTP client
2. Strips all tags, scripts, styles, navigation
3. Extracts only visible paragraph text
4. Returns clean UTF-8 string to `manas-ingest`

#### Freshness Checker

```rust
pub struct FreshnessChecker;

impl FreshnessChecker {
    /// Check all neurons — return list of stale neuron ids
    pub fn find_stale(&self, network: &Network) -> Vec<u64>;

    /// Refresh a specific neuron — re-search internet and re-learn
    pub async fn refresh_neuron(&self, neuron_id: u64, network: &mut Network) -> Result<()>;

    /// Refresh all stale neurons (called on query or on schedule)
    pub async fn refresh_all_stale(&self, network: &mut Network) -> Result<RefreshReport>;
}
```

---

### 6.7 `manas-cli`

The command line interface. The entry point for all user interaction. `teach` is the high-level local teaching path, and `ask` is the local source-backed answering path. `learn`, `ingest`, and `train-language` remain lower-level commands for direct core memory, source-aware ingestion, and language/transformer training. Normal `query` behavior remains the existing retrieval/search flow unless `--answer` routes it into the same local answer helper as `ask`.

Full CLI reference is in [Section 17](#17-cli-reference).

---

### 6.8 `manas-language`

The language modeling crate. Provides next-token prediction, a transition-count sequence memory, a hybrid memory+neural predictor, autoregressive text generation, and a small custom transformer path with trained output-head, FFN, partial attention `w_o`/`w_v`/`w_q`/`w_k` projection training, and reliability-aware hybrid score weighting.

#### Key Components

```rust
/// Transition-count table: context → (target → count)
pub struct SequenceMemory {
    pub transitions: HashMap<Vec<u32>, HashMap<u32, u32>>,
}
```

##### SequenceMemory

Records every transition `(context_tokens, target_token)` seen during training. For each transition, all suffix sub-contexts are also stored (e.g. for context `[10, 20, 30]`, suffixes `[30]`, `[20, 30]`, `[10, 20, 30]` are all recorded). This enables **suffix backoff** during prediction — if the full context is unseen, progressively shorter suffixes are tried.

- `record(context, target)` — stores the transition and all suffix contexts
- `lookup_suffix(context)` — returns `(target_id, count)` sorted by count, trying shorter suffixes on miss
- `save_to_file()` / `load_from_file()` — custom binary format stored alongside the brain as `brain.manas.seq`

##### Hybrid Prediction

```rust
final_score = 0.8 × mem_score + 0.2 × neural_score
```

Where:
- `mem_score` = normalized transition count from sequence memory (suffix backoff)
- `neural_score` = cosine similarity between network output and each token embedding
- Context tokens are penalized by `-0.3` unless backed by sequence memory

The memory weight (0.8) dominates for seen transitions; the neural weight (0.2) fills gaps for novel contexts.

#### Data Flow

```
train-language:
  Input text → Tokenize → Build sequences (sliding window)
    → Hash text for duplicate detection (v0.7.1)
    → Load/update LanguageMeta sidecar
    → For each (context, target):
      → Embed context → Forward → Loss → Backprop
      → Grow neuron only in first epoch and under max_new_neurons cap
      → Record in SequenceMemory (all suffix contexts)
      → Tag updated neurons with source + freshness
    → Save brain + SequenceMemory sidecar + LanguageMeta sidecar

predict-next:
  Load brain + SequenceMemory
    → Tokenize context
    → Hybrid prediction: memory (suffix backoff) + neural (forward pass)
    → Return top-k candidates

generate:
  Load brain + SequenceMemory
    → Tokenize prompt → Loop:
      → Stop if context has no memory match (past training data)
      → Hybrid prediction → append best token
      → Stop on 3+ consecutive repeats
      → Stop on cycle detection (pattern of 2-8 tokens repeats)
    → Decode tokens → Print "Generated:\n<text>"
```

#### CLI Commands

```bash
manas train-language "text"  --epochs 50  --learning-rate 0.05  --max-context 5  --max-new-neurons 10  --no-grow
manas predict-next "prompt"  --top-k 5    --max-context 5
manas generate "prompt"      --max-tokens 20  --max-context 5  --top-k 1  --temperature 1.0
```

#### Source Metadata

Language-trained neurons are stamped with `Source::RawText` and a detected freshness category. The `tag_neurons()` method (public on `Trainer`) stamps only neurons with `Source::Unknown`, preserving provenance from previous `learn`/`ingest` calls.

#### Tiny Transformer Block (v0.5)

`TinyTransformerBlock` stacks `CausalSelfAttention` + `FeedForward` with residual connections:

```txt
inputs
→ causal self-attention
→ residual add: x + attention_output
→ feed-forward per token (ReLU)
→ residual add: x + feed_forward_output
→ outputs
```

`FeedForward` uses a single hidden layer: `w1 @ x + b1 → ReLU → w2 @ hidden + b2`. Weights initialized with the same deterministic random scheme as attention.

**v0.6** — the block is experimentally connected to `predict-next --use-transformer` and `generate --use-transformer` via the `TransformerPredictor` in `lib.rs`. Scoring uses cosine-similarity against vocab embeddings. Default generation path (v0.3) unchanged.

**v0.7** — a `TransformerLanguageModel` wraps the block with a trainable linear output head (`output_w`, `output_b`). The `--train-transformer` flag on `train-language` trains the output head via cross-entropy loss while keeping the block frozen. The trained model is persisted in a `brain.manas.transformer` sidecar file. When the output head is available, the transformer weight in the hybrid score increases from 0.25 to 0.40. The block itself is not serialized — it's deterministically rebuilt from `(embed_dim, hidden_dim)` on load.

**v0.7.1** — neural growth optimization for `train-language`. Growth is now capped by `max_new_neurons` (default 10) and only attempted during the **first epoch** of training, preventing repeated per-epoch explosion. A `LanguageMeta` struct persisted as `brain.manas.langmeta` tracks text hashes for **duplicate-text detection** — repeated training of the same text automatically sets the growth cap to 0. CLI flags `--max-new-neurons <N>` and `--no-grow` give the user direct control. The `LanguageTrainReport` now reports `neurons_grown`.

**v0.8** — transformer training now includes the `FeedForward` layer. `FeedForward::train_step()` performs forward-pass caching, full backprop through w1/b1/w2/b2 with ReLU derivative, gradient clipping to [-1, 1], and NaN/inf skip. `train_transformer_output_head()` now trains both the output head and the FFN: it computes dL/d(block output) by backpropagating through the output head, then calls `feed_forward.train_step()` on the last token's FFN input. The `TransformerLanguageModel` persists FFN weights in the sidecar (version 2 format) and tracks an `ffn_trained` flag. `TransformerPredictor::from_model()` copies the trained block. Attention Q/K/V/O remain frozen. `manas inspect` reports `FFN trained : yes/no`.

**v0.8.1** — transformer training now returns a `TransformerTrainReport` with detailed metrics: per-epoch loss tracking, first/final/avg loss, improvement percentage, pure transformer top-1/top-3 accuracy, invalid/NaN update count, and output-head/FFN/attention status. Added `evaluate_transformer_on_examples()` which computes both loss and accuracy from the same pure-transformer forward pass, skipping examples whose target is not in `vocab_order` (consistent with training). Added `TransformerEvalReport` struct. CLI features: `--transformer-learning-rate` (default 0.01) separates transformer LR from language LR; `--transformer-only` on `predict-next` shows pure transformer scores without hybrid mixing.

**v0.8.2** — safer transformer training with norm-based gradient clipping (`gradient_norm()` / `clip_by_norm()` helpers), loss explosion detection (NaN/inf, max_loss, epoch-explosion factor), `TransformerTrainingSafety` config struct (defaults: max_gradient_norm=5.0, max_loss=50.0, loss_explosion_factor=5.0, rollback_on_unstable=true), model snapshot rollback on serious instability, `is_finite_model()` pre-save guard, and `train_transformer_output_head_with_safety()` entry point. `TransformerTrainReport` extended with `max_gradient_norm_seen`, `avg_gradient_norm`, `clipped_updates`, `unstable_updates`, `rolled_back`. CLI shows a dedicated "Training safety" block with `--transformer-max-grad-norm`, `--transformer-max-loss`, `--no-transformer-rollback` flags.

**v0.9.0** — attention training foundation only. `CausalSelfAttention::forward_with_cache()` returns the normal forward output plus Q/K/V projections, causal attention weights, and weighted values so future attention backprop can reuse exact forward-pass state. `TransformerLanguageModel` now tracks `attention_trained: bool`; the transformer sidecar is version 3 and persists `w_q`, `w_k`, `w_v`, and `w_o` while preserving v2 loading by rebuilding deterministic untrained attention. `is_finite_model()` checks attention weights. No attention projection training, scoring change, or generation change is included in v0.9.0.

**v0.9.1** — transformer training now updates only the attention output projection `w_o`. The training step uses the cached final-position weighted value vector as `context_last` and the gradient flowing into the attention output as `grad_attention_output_last`, then computes `grad_w_o = outer(grad_attention_output_last, context_last)`. Minimal FFN backward support returns `dL/d(ffn_input)` so the attention-output gradient is the residual gradient plus the FFN input gradient. The update goes through the existing safety path: norm tracking, clipping, invalid-gradient rejection, finite-model checks, and rollback. `w_q`, `w_k`, and `w_v` remain frozen; no softmax/QK gradients, scoring change, generation change, tokenizer change, model-size change, or sidecar version bump are included. `manas inspect` and training reports show partial attention as `Attention trained : partial` and `Attention projections : o`.

**v0.9.2** — transformer training now also updates the attention value projection `w_v`. It first computes `grad_context_last = w_o^T * grad_attention_output_last` using the pre-update output projection, then uses the cached final-position attention row to distribute that gradient into value vectors: `grad_v_j = attention_prob(last, j) * grad_context_last`, followed by `grad_w_v += outer(grad_v_j, input_j)`. The existing output head, FFN, and `w_o` training continue. `w_q` and `w_k` remain frozen, and there is still no backprop through attention scores, softmax, Q, or K. Safety metrics include `w_v` gradient norms, clipping, invalid update counts, finite checks, and rollback. The transformer sidecar remains version 3; new files append an optional projection bitmask so inspect can report `Attention projections : o,v`, while legacy v3 files without the bitmask load as `o`.

**v0.9.3** — transformer training now also updates the attention query/key projections `w_q` and `w_k` together for the final token position. It reuses the same attention cache as the forward pass and computes the causal softmax gradient only over allowed positions `j <= i`: `grad_a_j = dot(grad_context_last, v_j)`, `grad_score_j = a_j * (grad_a_j - sum_l a_l * grad_a_l)`, then accumulates `grad_w_q += outer(grad_q_i, x_i)` and `grad_w_k += outer(grad_k_j, x_j)`. Output head, FFN, `w_o`, and `w_v` continue training. The system remains single-head, keeps transformer sidecar version 3, and does not change tokenizer, sequence memory, scoring weights, model dimensions, generation behavior, layer norm, or dynamic growth. Inspect and training reports still say partial attention and display `Attention projections : o,v,q,k`.

**v0.9.4** — attention training safety and observability were tightened without changing training math, scoring weights, generation behavior, model dimensions, or sidecar version. `AttentionTrainStepReport` now records attempted updates and pre-clip gradient norms consistently for `w_o`, `w_v`, and `w_q/w_k`. `TransformerTrainReport` adds attention-specific attempts/applied/clipped/invalid counters plus max/avg attention gradient norms, while global safety counters remain unchanged and are not double-counted. The CLI prints a compact "Attention safety" block. `TransformerLanguageModel::save_to_file()` refuses non-finite transformer models, assisted and transformer-only prediction filter non-finite scores before sorting, and rollback on serious instability restores the output head, FFN, all attention projections, `ffn_trained`, `attention_trained`, and the projection bitmask. Inspect remains conservative: `Attention trained : partial` and `Attention projections : o,v,q,k`.

**v0.9.5** — transformer-assisted prediction now uses reliability-aware score weighting instead of the old fixed 0.25/0.40 blend. `TransformerPredictor` carries runtime metadata copied from `TransformerLanguageModel`: `ffn_trained`, `attention_projection_mask`, and `model_finite`. The base transformer weight is `0.15` for untrained cosine fallback, `0.30` for output-head-only, `0.35` for output head + FFN, `0.45` for attention `o`, `0.50` for attention `o,v`, and `0.55` for attention `o,v,q,k`. A simple confidence factor reduces transformer influence when top probability or top-1/top-2 margin is weak. Strong base-memory candidates cap transformer influence, learned sequence-memory candidates use a stricter cap to preserve exact transitions, and non-finite transformer state falls back to base memory/neural scores. `--transformer-only` remains pure transformer output. No tokenizer, sequence memory format, persistence format, sidecar version, training math, attention architecture, CLI default, or generation CLI behavior changes are included.

**v0.9.6** — `manas teach <INPUT>` unifies local teaching UX without changing the underlying learning systems. Direct text teaching learns core memory and trains sequence memory; file/folder teaching preserves source paths while teaching core/source-aware memory, sequence memory, and optional transformer weights. The command supports `.md` and `.txt` files, recursively teaches folders with deterministic ordering, skips unsupported or empty files safely, and provides `--dry-run` without writing brain or sidecar files. Existing `learn`, `ingest`, and `train-language` commands remain available as lower-level controls. No tokenizer, sequence-memory format, transformer sidecar version, transformer dimensions, training math, scoring weight, attention architecture, generation behavior, or dependency change is included.

**v0.9.7** — `manas ask "question"` adds local-first answering from taught source memory. The answer path loads the local brain, collects `Source::LocalFile` paths from neurons, re-reads existing `.md` and `.txt` files, splits them into deterministic sentence snippets, ranks snippets by local token overlap and source metadata, and returns a short extracted answer with source paths. If evidence is weak it reports related local memory without answering confidently; if evidence is missing it says there is not enough local memory. `manas query "question" --answer` uses the same helper, while normal `query` remains unchanged. No internet, cloud API, external embedding service, transformer sidecar change, tokenizer change, training math change, scoring-weight change, or teach behavior change is included.

#### Single-Head Causal Attention (v0.4)

`CausalSelfAttention` is a standalone module in `attention.rs` with QKV projections, scaled dot-product scores, causal masking, and an output projection. It is implemented as custom Rust with no external dependencies. Not yet integrated into the default prediction path.

```rust
pub struct CausalSelfAttention {
    pub embed_dim: usize,
    pub w_q: Vec<f32>,   // embed_dim × embed_dim
    pub w_k: Vec<f32>,   // embed_dim × embed_dim
    pub w_v: Vec<f32>,   // embed_dim × embed_dim
    pub w_o: Vec<f32>,   // embed_dim × embed_dim
}
```

Weights are initialized with small random values (`N(0, 0.02)` scaled by `1/sqrt(d)`). The forward pass computes Q, K, V for each input token, applies causal masking (position `i` only attends to `0..=i`), scaled dot-product attention, softmax, weighted sum of V, and output projection.

`AttentionForwardCache` stores the forward intermediates needed by the next training step:

```rust
pub struct AttentionForwardCache {
    pub qs: Vec<Vec<f32>>,
    pub ks: Vec<Vec<f32>>,
    pub vs: Vec<Vec<f32>>,
    pub attention_weights: Vec<Vec<f32>>,
    pub weighted_values: Vec<Vec<f32>>,
}
```

`CausalSelfAttention::forward_with_cache()` returns `(outputs, cache)`. `forward()` delegates through the same path, so inference and prediction behavior stay unchanged.

`CausalSelfAttention::train_output_projection_step()` is the v0.9.1 partial attention trainer. It accepts a cached context vector, an output gradient, a learning rate, and a max gradient norm. It updates only `w_o`, reports whether the update was attempted, applied, clipped, or invalid, records the pre-clip gradient norm, rejects non-finite gradients without mutation, and leaves `w_q`, `w_k`, and `w_v` untouched.

`CausalSelfAttention::train_value_projection_step()` is the v0.9.2 partial attention trainer. It accepts the original token embeddings, the cached final-position attention weights, `grad_context_last`, a learning rate, and a max gradient norm. It updates only `w_v`, reports whether the update was attempted, applied, clipped, or invalid, records the pre-clip gradient norm, rejects non-finite gradients without mutation, and leaves `w_q`, `w_k`, and `w_o` untouched.

`CausalSelfAttention::train_query_key_projection_step()` is the v0.9.3 partial attention trainer. It accepts the original token embeddings, cached Q/K/V projections, the cached final-position attention row, `grad_context_last`, a learning rate, and a max gradient norm. It updates only `w_q` and `w_k` through the causal softmax score gradient, clips their combined gradient norm, reports attempted/applied/clipped/invalid state and the pre-clip gradient norm, rejects non-finite gradients without mutation, and leaves `w_v` and `w_o` untouched.

Helpers: `mat_vec_mul`, `dot`, `softmax` (numerically stable, subtracts max before exp).

---

## 7. The `.manas` Binary Format

```
Offset    Size     Field
────────────────────────────────────────────────────────
[FILE HEADER — 64 bytes fixed]
0         5        magic bytes: "MANAS"
5         1        format version: u8
6         8        created_at: u64 (unix timestamp)
14        8        last_modified: u64
22        8        total_neurons: u64
30        4        total_layers: u32
34        4        vocab_size: u32
38        8        total_texts_learned: u64
46        2        flags: u16 (bit 0 = has_archive, bit 1 = compressed)
48        4        checksum_offset: u32
52        12       reserved (zero padded)

[VOCAB BLOCK]
0         4        vocab_entry_count: u32
per entry:
  4        token_id: u32
  1        token_len: u8
  N        token bytes (UTF-8)
  2        embed_dim: u16            (actual embedding dimension, default 64)
  D×4      embedding: [f32; D]       (embed_dim × 4 bytes)

[LAYER INDEX]
per layer:
  4        layer_id: u32
  8        byte_offset: u64       (where this layer's block starts)
  4        neuron_count: u32

[LAYER BLOCK × N]
  4        layer_id: u32
  4        neuron_count: u32
  1        default_activation: u8

  [NEURON BLOCK × M]
    8        id: u64
    2        weight_count: u16
    W×4      weights: [f32]
    4        bias: f32
    1        activation: u8
    4        importance_score: f32
    8        born_at: u64
    8        last_activated: u64
    8        activation_count: u64
    8        learned_at: u64
    8        last_verified: u64
    1        freshness_category: u8  (0=timeless,1=slow,2=fast,3=realtime)
    1        source_type: u8         (0=text,1=file,2=internet,3=unknown)
    2        source_len: u16
    N        source_bytes (UTF-8 path or URL)
    1        is_protected: u8        (0=false, 1=true)
    1        protection_level: u8    (0=open, 1=guarded, 2=frozen)

[ARCHIVE BLOCK]
  (same format as LAYER BLOCK, for compressed/merged old neurons)

[CHECKSUM — 4 bytes]
  4        CRC32 of all bytes above
```

**File size growth:**
Every new neuron adds approximately `(weight_count × 4) + ~100` bytes.
A 128-weight neuron adds ~612 bytes. The file grows only by the new neuron's size.

---

## 8. The Growth System

### When Growth Is Triggered

```
Each learning step:
  1. Forward pass → get prediction
  2. Calculate MSE loss
  3. Backpropagate → get weight deltas
  4. Apply deltas (if neuron not protected)
  5. Re-calculate loss after update

  IF new_loss > GROWTH_THRESHOLD (0.35)
  AND update_attempts >= 3
  AND layer_is_not_at_max_neurons:
      → trigger grow_neuron(highest_loss_layer)
      → re-run forward pass
      → save
```

### New Neuron Initialization

When a new neuron is grown:

- Weights: randomly initialized from `N(0, 0.1)` (small Gaussian noise)
- Bias: 0.0
- Importance score: 0.5 (neutral — neither protected nor at risk)
- Protection: `Guarded` for first 7 days (grace period while it settles)
- `born_at`: current unix timestamp
- Source: set to `Source::Unknown` initially; stamped together with `freshness_category` by `tag_neurons()` on first learn — both are set only once and never overwritten
- For source-aware growth: stamped with the file path or URL and detected freshness immediately
- For language training (`train-language`): stamped with `Source::RawText` and detected freshness category

### Source-Aware Growth

When ingesting files or internet pages, Manas also grows **source-owned neurons**
to ensure each knowledge source has dedicated capacity:

```
After each chunk is learned:

IF source is LocalFile or Internet
AND no existing neuron has this exact source path/URL
THEN
    grow 1 neuron in layer 0
    stamp source = chunk.source (file path or URL)
    stamp freshness_category = detected category
    recalculate importance and protection
```

Key properties:

- **Bounded**: at most 1 neuron per unique source (file path or URL)
- **Non-destructive**: existing neurons keep both their original source AND
  freshness_category — neither is overwritten when a new file is ingested
- **Per-source identity**: each file/URL gets a dedicated neuron that carries
  its origin as metadata, visible via `manas neurons --all`
- **Lightweight**: grows only 1 neuron even for large files (not 1 per chunk)

### Growth Control in Language Training (v0.7.1)

`train-language` growth is now controlled by `max_new_neurons`:

```
Each training call:
  1. Hash the input text → check LanguageMeta sidecar for known duplicates
  2. If known or --no-grow → max_new_neurons = 0 (no growth)
  3. For epoch in 0..epochs:
     allow_growth = (epoch == 0) && (neurons_grown < max_new_neurons)
     Only grow neurons when allow_growth is true
  4. After training: record text hash in LanguageMeta sidecar
```

This prevents neuron explosion when the same text is trained repeatedly
(e.g. training the same sentence twice would previously grow ~400+ neurons
over 100 epochs; now the second call detects the duplicate and grows 0).

### Layer Growth

If all neurons in every existing layer are `Frozen` and loss is still too high,
a new layer is appended to the network. This is rarer than neuron growth within a layer.

---

## 9. The Memory Importance System

### Importance Score Formula

```
importance = (
    0.40 × clamp(activation_count / 10_000, 0.0, 1.0)   +
    0.30 × recency(last_activated, now)                   +
    0.20 × clamp(weight_l2_norm / 10.0, 0.0, 1.0)       +
    0.10 × age_grace(born_at, now)
)

recency(t, now) = exp(-λ × (now - t) / 86400)   where λ = 0.1

age_grace(born, now):
    if (now - born) < 7_days  → return 1.0   (grace period)
    else                      → return 0.0
```

### Score Bands

| Score | Status | Meaning |
|---|---|---|
| 0.85 – 1.00 | 🔒 Frozen | Core knowledge, protected from modification |
| 0.60 – 0.85 | 🛡 Guarded | Important, small updates only |
| 0.20 – 0.60 | ✅ Open | Normal learning allowed |
| 0.00 – 0.20 | 🗜 Compress candidate | Rarely used, may be merged |

### Compression

Compression is **never destructive**. Old neurons are moved to the `[ARCHIVE BLOCK]`
inside the `.manas` file. They can be restored at any time with `manas restore`.

---

## 10. The Freshness System

### Freshness Categories

| Category | Label | Auto-refresh after | Example knowledge |
|---|---|---|---|
| 0 | Timeless | Never | Math, logic, language rules |
| 1 | Slow | 30 days | Historical facts, geography |
| 2 | Fast | 7 days | Technology, software versions, docs |
| 3 | Realtime | 1 day | News, prices, current events |

### Auto-detection

When knowledge is first learned (via `learn`, `ingest`, or `train-language`),
the freshness category is auto-detected by scanning the input text for keywords:

```
"released", "version", "update", "latest" → category 2 (Fast)
"news", "today", "breaking", "market"     → category 3 (Realtime)
"since", "history", "was", "were"         → category 1 (Slow)
"always", "formula", "law", "proof"       → category 0 (Timeless)
default fallback                          → category 1 (Slow)
```

### Freshness Check on Query

```
User runs: manas query "What is the latest Rust version?"
      │
      ▼
Retrieve relevant neurons
      │
      ▼
For each neuron:
  stale = (now - last_verified) > category_threshold
      │
    stale?
   ┌──┴──┐
  YES    NO
   │      │
   ▼      ▼
Trigger  Use
refresh  directly
   │
   ▼
manas-agent searches internet
   │
   ▼
manas-ingest normalizes result
   │
   ▼
manas-learn re-learns on updated text
   │
   ▼
neuron last_verified updated to now
   │
   ▼
Answer with fresh knowledge
```

---

## 11. The Local File Ingestion System

### Supported File Types

| Extension | Parser | What is extracted |
|---|---|---|
| `.txt` | plaintext | Full text content |
| `.md` | markdown | Text with markdown syntax stripped |
| `.rs` | rust_source | Doc comments (`///`, `//!`), function signatures, module structure |
| `.toml` | toml | Key = value pairs converted to readable sentences |
| `.json` | json | Flattened key: value text |
| `.csv` | csv | Each row as a natural language sentence with headers |
| `.html` | html | Visible paragraph text, headings, links text only |

### Folder Walk Rules

```
Given: manas ingest --folder ./my-notes/

1. Walk directory recursively (follows symlinks with cycle detection)
2. For each file:
   a. Check extension against supported list
   b. If supported → read → parse → normalize → chunk → send to manas-learn
   c. If unsupported → skip silently (log to debug output)
3. Each chunk tagged with:
   Source::LocalFile { path: "/absolute/path/to/file.md" }
4. After folder walk complete → print summary:
   "Ingested 47 files | 12,483 text chunks | 0 errors"
```

### Chunking Strategy

Large files are split into chunks before learning, so neurons represent
focused pieces of knowledge rather than one massive blob:

```
Default chunk size: 512 characters
Overlap: 16 characters (context continuity between chunks)
Split boundary: prefer sentence/paragraph boundaries
```

### Change Detection (re-ingest only what changed)

Each ingested file gets a record stored in `.manas`:

```rust
pub struct IngestedFile {
    pub path:          String,
    pub last_seen:     u64,     // unix timestamp
    pub file_hash:     u64,     // xxHash of file contents
    pub chunk_count:   u32,
}
```

On re-ingest, if `file_hash` is unchanged, the file is skipped.
Only modified or new files are re-learned.

---

## 12. The Internet Agent System

### Search Backend

The agent uses a configurable search backend:

```toml
# ~/.config/manas/config.toml
[agent]
search_backend = "ddg"      # options: "ddg" (DuckDuckGo), "brave", "custom"
max_results_per_query = 5
scrape_timeout_secs = 10
respect_robots_txt = true
```

### Full Agent Pipeline

```
User query OR stale neuron
         │
         ▼
  Build search query
  (if from neuron: use knowledge tags + topic keywords)
         │
         ▼
  manas-agent::Searcher
  → returns top 5 URLs + snippets
         │
         ▼
  For each URL:
    manas-agent::Scraper
    → fetch HTML
    → strip noise
    → extract clean text
         │
         ▼
  manas-ingest::IngestPipeline
  → normalize text
  → chunk into 512-char pieces
  → tag source as Source::Internet { url }
         │
         ▼
  manas-learn::Trainer
  → learn from each chunk
  → update or grow neurons
         │
         ▼
  manas-store
  → save updated .manas file
```

---

## 13. Data Flow — Full Pipeline

### Learning from local file

```
manas ingest --file ./notes.md
         │
         ▼
manas-ingest:
  read file → parse markdown → strip syntax → normalize text
  → split into 512-char chunks
  → tag each chunk: Source::LocalFile { path: "./notes.md" }
         │
         ▼
manas-learn:
  tokenize chunk → embed tokens → forward pass
  → calculate loss → backprop → check growth
  → tag updated neurons with chunk.source + freshness (only if Unknown)
         │
         ▼
manas-core:
  if growth needed → grow_neuron()
  else → update_weights() with protection check
         │
         ▼
manas-memory:
  recalculate importance scores for affected neurons
         │
         ▼
manas-learn (source-aware):
  if source is LocalFile/Internet and no neuron has it → grow 1 source neuron
  stamp source + freshness on the new neuron
         │
         ▼
manas-store:
  append new neurons OR update existing neurons in .manas file
```

### Answering from local source memory

```
manas ask "What is Manas?"
          │
          ▼
manas-cli:
  load local brain → collect Source::LocalFile paths from neurons
          │
          ▼
local source reader:
  re-read existing .md/.txt files → split into sentence snippets
          │
          ▼
local ranker:
  score snippets by question-token overlap + source metadata tie-breakers
          │
          ▼
answer composer:
  high-confidence sentence → short extracted answer + source paths
  weak/no evidence → conservative no-answer message
```

This path is local-only. It does not construct the agent search pipeline, call web search, use hosted LLM APIs, or use external embedding services.

### Existing query/search path

```
manas query "What is ownership in Rust?"
          │
          ▼
manas-learn:
  tokenize query → embed → forward pass
          │
          ▼
manas-agent::FreshnessChecker:
  find relevant neurons → check last_verified vs threshold
  any stale? → refresh from internet first
          │
          ▼
manas-core:
  forward pass with current weights → generate answer vector
          │
          ▼
manas-learn::decoder:
  cosine-similarity between output vector and embedder table
  → rank by score → return top-20 closest tokens
          │
          ▼
manas-cli:
  print decoded keywords + neuron activations to terminal
```

---

## 14. Neuron Lifecycle

```
BORN
 │  randomly initialized weights
 │  importance_score = 0.5
 │  protection = Guarded (7-day grace)
 │
 ▼
LEARNING (days 0–7)
 │  absorbing new patterns
 │  protection prevents harsh overwriting
 │  importance_score rising or falling based on activation
 │
 ▼
SETTLED (day 7+)
 │  protection drops to Open (unless score is high)
 │  normal backprop updates apply
 │  importance_score fully dynamic now
 │
  ├──► HIGH IMPORTANCE (score > 0.85)
  │     → protection = Frozen
  │     → protected from modification
  │     → preserved core knowledge
 │
 ├──► MEDIUM IMPORTANCE (score 0.20–0.85)
 │     → stays Open
 │     → continues to update
 │     → re-scores every learning step
 │
 └──► LOW IMPORTANCE (score < 0.20)
       → compress candidate
       → merged with similar neurons
       → archived in .manas archive block
       → can be restored if needed
```

---

## 15. Error Handling Strategy

All public functions return `Result<T, ManasError>`.

```rust
pub enum ManasError {
    // Storage errors
    FileNotFound(PathBuf),
    CorruptFile { path: PathBuf, reason: String },
    ChecksumMismatch,

    // Learning errors
    TokenizerError(String),
    EmbeddingError(String),
    BackpropError(String),

    // Ingest errors
    UnsupportedFileType(String),
    FileReadError { path: PathBuf, source: std::io::Error },

    // Agent errors
    NetworkError(String),
    ScraperError(String),
    SearchBackendError(String),

    // Growth errors
    GrowthFailed(String),
    LayerLimitReached,
}
```

No panics in library code. The CLI converts errors to user-friendly messages.

---

## 16. Milestone Plan

| # | Milestone | Crates | Output |
|---|---|---|---|---|
| M1 | Neuron/layer/network structs, forward pass, growth logic | `manas-core` | Working dynamic neural net |
| M2 | `.manas` binary format, read/write, append | `manas-store` | Persistent brain file |
| M3 | Tokenizer, embedder, backprop, online learning loop | `manas-learn` | Model can learn from text |
| M4 | Raw text + local file + folder ingestion pipeline | `manas-ingest` | Learn from disk |
| M5 | Importance scoring, protection, compression | `manas-memory` | Knowledge preservation system |
| M6 | Full CLI — learn, ingest, query, inspect | `manas-cli` | Usable from terminal |
| M7 | Internet search agent, HTML scraper | `manas-agent` | Web learning |
| M8 | Freshness checker, auto re-search on stale knowledge | `manas-agent` | Always up to date |
| M9 | Full integration, end-to-end testing | all | Complete working system |
| M10 | Performance optimization, benchmarks | all | Performance-tuned prototype |
| M11 | **Next-token prediction (v0.2)** — sequence memory, hybrid predictor, source metadata | `manas-language` | Local next-token prediction |
| M12 | **Real text generation (v0.3)** — loop prevention, memory-boundary stop, cycle detection | `manas-language` | Stable autoregressive generation |
| M13 | **Single-head causal attention (v0.4)** — QKV, scaled dot-product, causal mask | `manas-language` | Custom attention module |
| M14 | **Tiny transformer block (v0.5)** — causal attention + FFN + residual | `manas-language` | Forward-only transformer block |
| M15 | **Transformer-assisted prediction (v0.6)** — `--use-transformer` flag, hybrid scoring, default path unchanged | `manas-language`, `manas-cli` | Experimental transformer integration |
| M16 | **Transformer output-head training (v0.7)** — `--train-transformer` flag, cross-entropy, output head only, dynamic weight (0.40 trained / 0.25 untrained) | `manas-language`, `manas-cli` | Transformer learns next-token prediction |
| M17 | **Neural growth optimization (v0.7.1)** — `max_new_neurons` cap, first-epoch-only growth, `LanguageMeta` sidecar for duplicate-text detection, `--max-new-neurons`/`--no-grow` CLI flags | `manas-language`, `manas-cli` | Controlled network growth |
| M18 | **Enhanced system inspect (v0.7.2)** — `manas inspect` shows Core Network, Language System, Transformer, Storage, and Total sections; reports sidecar file sizes, transformer param counts, sequence memory status, language metadata; `--verbose` flag | `manas-cli` | Full inspect visibility |
| M19 | **FFN training (v0.8)** — `FeedForward::train_step()`, `forward_with_ffn_inputs()`, FFN weight persistence (v2 sidecar), `ffn_trained` flag, gradients clipped to [-1, 1], NaN/inf safety, attention frozen | `manas-language`, `manas-cli` | Transformer FFN learns next-token signal |
| M20 | **Training metrics (v0.8.1)** — `TransformerTrainReport`, per-epoch loss, top-1/top-3 accuracy, improvement %, invalid update tracking, formatted CLI output | `manas-language`, `manas-cli` | Measurable transformer training |
| M21 | **Attention cache + persistence prep (v0.9.0)** — `AttentionForwardCache`, `forward_with_cache()`, attention finite checks, v3 transformer sidecar with attention weights, `attention_trained` inspect status | `manas-language`, `manas-cli` | Foundation for attention projection training |
| M22 | **Attention output projection training (v0.9.1)** — `train_output_projection_step()`, FFN input-gradient support, `w_o` update with safety metrics and rollback, `q/k/v` frozen, partial inspect/report status | `manas-language`, `manas-cli` | Safest attention projection starts learning |
| M23 | **Attention value projection training (v0.9.2)** — `train_value_projection_step()`, `w_v` update from cached final attention row, optional v3 projection bitmask, `q/k` frozen, partial `o,v` inspect/report status | `manas-language`, `manas-cli` | Attention value representations start learning |
| M24 | **Attention query/key projection training (v0.9.3)** — `train_query_key_projection_step()`, causal final-token softmax gradient, combined Q/K clipping, finite-difference tests, partial `o,v,q,k` inspect/report status | `manas-language`, `manas-cli` | Attention routing starts learning |
| M25 | **Attention safety and metrics cleanup (v0.9.4)** — attention-specific safety counters, finite save guard, non-finite prediction-score filtering, rollback of output head/FFN/attention flags and projections, stable `o,v,q,k` reporting | `manas-language`, `manas-cli` | Attention training is safer and easier to debug |
| M26 | **Reliability-aware transformer score weighting (v0.9.5)** — runtime reliability metadata, trained-projection weight tiers, confidence factor, sequence-memory cap, non-finite fallback to base scores, deterministic score sorting | `manas-language` | Transformer influence grows only when reliable |
| M27 | **Unified teaching command (v0.9.6)** — `manas teach <INPUT>` orchestrates core/source-aware memory, sequence memory, optional transformer training, `.md`/`.txt` folder teaching, and dry-run reporting | `manas-cli` | One command teaches text, files, and folders |
| M28 | **Local query answering (v0.9.7)** — `manas ask`, `query --answer`, local `.md`/`.txt` source snippet ranking, extracted answers, source display, and no-evidence fallback | `manas-cli` | Questions can be answered from taught local source memory |

---

## 17. CLI Reference

```bash
# ── LEARNING ──────────────────────────────────────────────────────

# Learn from raw text
manas learn "Rust is a systems programming language"

# Unified local teaching: core memory + sequence memory + optional transformer
manas teach "Manas is a local-first AI memory system"
manas teach ./notes.md --train-transformer
manas teach ./my-notes/ --dry-run

# Learn from a file
manas ingest --file ./notes.md

# Learn from a folder (recursive)
manas ingest --folder ./my-notes/

# Learn from a URL (agent fetches and cleans)
manas ingest --url https://doc.rust-lang.org/book/

# Preview ingest without learning
manas ingest --folder ./docs/ --dry-run

# ── LANGUAGE (v0.2) ───────────────────────────────────────────────

# Train next-token prediction
manas train-language "Rust is a systems programming language" --epochs 50

# Train next-token prediction with transformer output head, FFN, and attention w_o/w_v/w_q/w_k (v0.9.5)
manas train-language "Rust is a systems programming language" --epochs 50 --train-transformer

# Train next-token prediction with growth control (v0.7.1)
manas train-language "Rust is a systems programming language" --epochs 50 --max-new-neurons 5
manas train-language "Duplicate text" --epochs 50 --no-grow

# Predict the next word (hybrid memory + neural)
manas predict-next "Rust is a" --top-k 5

# Predict next word with experimental transformer assistance (v0.6)
manas predict-next "Rust is a" --use-transformer --top-k 5

# Generate text autoregressively (default: stable v0.3)
manas generate "Rust is a" --max-tokens 10

# Generate text with experimental transformer assistance (v0.6)
manas generate "Rust is a" --use-transformer --max-tokens 10

# ── QUERYING ──────────────────────────────────────────────────────

# Answer from taught local source memory
manas ask "What is Manas?"
manas ask "What is Manas?" --top-k 5 --max-answer-tokens 80

# Existing retrieval/search query
manas query "What is ownership in Rust?"

# Compatibility: use local answer path through query
manas query "What is Manas?" --answer

# Ask with forced freshness check
manas query "Latest Rust version" --refresh

# Ask without freshness check (faster, use cached knowledge only)
manas query "What is a trait?" --no-refresh

# ── MAINTENANCE ───────────────────────────────────────────────────

# Refresh all stale knowledge from internet
manas refresh

# Refresh only a specific freshness category
manas refresh --category fast

# Show brain stats (v0.7.2 shows full system state)
manas inspect
# Output:
#  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#   Manas Brain — brain.manas
#  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#  Core Network
#  ─────────────────────────────────────
#   Core network layers : 8
#   Core neurons        : 4,821
#   Core network params : 128,498
#
#  Language System
#  ─────────────────────────────────────
#   Vocab size          : 12,483
#   Embedding dim       : 64
#   Embedding params    : 798,912
#   Sequence memory     : enabled
#   Sequence entries    : 15,342
#   Training runs       : 31,209
#   Unique texts        : 4,102
#   Repeated trainings  : 712
#
#  Transformer
#  ─────────────────────────────────────
#   Enabled             : yes
#   Blocks              : 1
#   Attention heads     : 1
#   Embed dim           : 64
#   FFN hidden dim      : 128
#   Output head trained : yes
#   FFN trained         : yes
#   Attention trained     : partial
#   Attention projections : o,v,q,k
#   Attention params    : 16,384
#   FFN params          : 16,512
#   Output head params  : 799,872
#   Transformer params  : 832,768
#
#  Storage
#  ─────────────────────────────────────
#   Brain file          : 9,437,184  (9.00 MB)
#   Sequence file       : 1,245,312  (1.19 MB)
#   Transformer file    : 3,201,792  (3.05 MB)
#   Language metadata   : 164,352    (160.50 KB)
#   Total storage       : 14,048,640 (13.40 MB)
#
#  Total
#  ─────────────────────────────────────
#   Total params        : 1,760,178
#   Last updated        : 2 hours ago
#  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Show with verbose output
manas inspect --verbose

# List all ingested files
manas files

# Trace neuron activations + decode the brain's output
manas trace "Rust ownership"
# Output:
#  Top 10 activated neurons:
#    n12     L0  act=0.2982  ...
#    ...
#  Closest known tokens (decoded):
#    network              sim=0.1499
#    neural               sim=0.0818

# ── FILE MANAGEMENT ───────────────────────────────────────────────

# Export brain to a file
manas export --out my_brain.manas

# Import brain from a file
manas import --file my_brain.manas

# Verify file integrity
manas verify

# ── ADVANCED ──────────────────────────────────────────────────────

# Show all neurons and their stats (verbose)
manas neurons --all

# Show neurons from a specific source
manas neurons --source file:./notes.md

# Restore archived neurons
manas restore --all

# Set freshness category manually for a topic
manas tag "Rust version" --freshness fast
```

---

## 18. Future Vision

Once the core system (M1–M10) is complete, Manas can grow in these directions:

### Phase 2 — Multi-Modal Input
- Image input via CLIP-style embeddings
- PDF native parser (beyond HTML conversion)
- Audio transcription → text pipeline

### Phase 3 — Multi-Brain Sync
- Two Manas instances can share knowledge over local network
- Team brain: multiple users contribute to a shared `.manas` file
- Diff/merge two `.manas` files (like git for brains)

### Phase 4 — Agent Mode
- Manas proactively searches the internet on a schedule
- Builds its own knowledge without user input
- Monitors a list of topics and keeps them fresh automatically

### Phase 5 — Vayu Integration
- Reactive web UI (Leptos) for visual brain inspection
- See neurons, layers, knowledge sources as interactive graph
- Runs locally on `localhost:7070`

---

*Built with ❤️ in Rust. This is your local network. It lives on your machine. It grows with you.*
