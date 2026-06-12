use clap::{Parser, Subcommand};
use manas_agent::{AgentPipeline, FreshnessChecker};
use manas_core::{ManasError, Network, Neuron, Source};
use manas_ingest::{IngestPipeline, IngestSource};
use manas_learn::{Trainer, TrainerSnapshot, decode, detect_freshness_category};
use manas_store::ManasBrain;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

// ─── CLI definition ───────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "manas", about = "Your personal AI brain", version = "0.1.0")]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    #[arg(short = 'b', long, default_value = "./brain.manas", global = true)]
    brain: String,
}

#[derive(Subcommand)]
enum Commands {
    Learn {
        text: String,
    },
    Ingest {
        #[arg(long)]
        file: Option<String>,
        #[arg(long)]
        folder: Option<String>,
        #[arg(long)]
        url: Option<String>,
        #[arg(long)]
        dry_run: bool,
    },
    Query {
        text: String,
    },
    Refresh {
        #[arg(long)]
        category: Option<String>,
    },
    Inspect,
    Files,
    Trace {
        text: String,
    },
    Export {
        #[arg(long)]
        out: Option<String>,
    },
    Import {
        #[arg(long)]
        file: String,
    },
    Verify,
    Neurons {
        #[arg(long)]
        all: bool,
    },
    Restore {
        #[arg(long)]
        all: bool,
    },
    Tag {
        text: String,
        #[arg(long)]
        freshness: String,
    },
}

// ─── Entry point ──────────────────────────────────────────────────────────────

fn main() {
    let cli = Cli::parse();
    let brain_path = PathBuf::from(&cli.brain);

    let result = match &cli.command {
        Commands::Learn { text } => cmd_learn(text, &brain_path),
        Commands::Ingest {
            file,
            folder,
            url,
            dry_run,
        } => cmd_ingest(
            file.as_deref(),
            folder.as_deref(),
            url.as_deref(),
            *dry_run,
            &brain_path,
        ),
        Commands::Query { text } => cmd_query(text, &brain_path),
        Commands::Refresh { category } => cmd_refresh(category.as_deref(), &brain_path),
        Commands::Inspect => cmd_inspect(&brain_path),
        Commands::Files => cmd_files(&brain_path),
        Commands::Trace { text } => cmd_trace(text, &brain_path),
        Commands::Export { out } => cmd_export(out.as_deref(), &brain_path),
        Commands::Import { file } => cmd_import(file, &brain_path),
        Commands::Verify => cmd_verify(&brain_path),
        Commands::Neurons { all } => cmd_neurons(*all, &brain_path),
        Commands::Restore { all } => cmd_restore(*all, &brain_path),
        Commands::Tag { text, freshness } => cmd_tag(text, freshness, &brain_path),
    };

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

// ─── Shared helpers ───────────────────────────────────────────────────────────

fn snapshot_to_vocab_map(snap: &TrainerSnapshot) -> HashMap<u32, (String, Vec<f32>)> {
    let mut map = HashMap::new();
    for (&id, token) in &snap.id_to_token {
        if let Some(emb) = snap.embed_table.get(&id) {
            map.insert(id, (token.clone(), emb.clone()));
        }
    }
    map
}

fn load_or_create_network(brain: &ManasBrain) -> Network {
    if brain.path.exists() {
        brain.load().unwrap_or_else(|_| Network::new())
    } else {
        Network::new()
    }
}

fn restore_trainer_from_brain(trainer: &mut Trainer, brain: &ManasBrain) {
    if let Ok(vocab) = brain.load_vocab()
        && !vocab.is_empty()
    {
        let embed_dim = vocab.values().next().map(|(_, e)| e.len()).unwrap_or(64);
        let snap = TrainerSnapshot {
            vocab: vocab.iter().map(|(&id, (t, _))| (t.clone(), id)).collect(),
            id_to_token: vocab.iter().map(|(&id, (t, _))| (id, t.clone())).collect(),
            embed_table: vocab.iter().map(|(&id, (_, e))| (id, e.clone())).collect(),
            embed_dim,
        };
        trainer.restore(&snap);
    }
}

fn save_brain(brain: &ManasBrain, network: &Network, trainer: &Trainer) -> Result<(), ManasError> {
    let snap = trainer.snapshot();
    brain.save_with_vocab(network, &snapshot_to_vocab_map(&snap))
}

fn format_duration(unix_ts: u64) -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let diff = now.saturating_sub(unix_ts);
    if diff < 60 {
        format!("{} seconds ago", diff)
    } else if diff < 3600 {
        format!("{} minutes ago", diff / 60)
    } else if diff < 86400 {
        format!("{} hours ago", diff / 3600)
    } else {
        format!("{} days ago", diff / 86400)
    }
}

// ─── Commands ─────────────────────────────────────────────────────────────────

/// `manas learn "some text"`
fn cmd_learn(text: &str, brain_path: &Path) -> Result<(), ManasError> {
    let brain = ManasBrain::new(brain_path);
    let mut network = load_or_create_network(&brain);
    let mut trainer = Trainer::new();
    restore_trainer_from_brain(&mut trainer, &brain);

    // FIX 2 — tag neurons as coming from raw user text
    trainer.source = Source::RawText;
    trainer.freshness_category = detect_freshness_category(text);

    let report = trainer.learn(&mut network, text)?;
    network.total_texts_learned += 1;
    save_brain(&brain, &network, &trainer)?;

    println!(
        "Learned {} tokens | loss: {:.4}{}",
        report.tokens_learned,
        report.loss,
        if report.growth_occurred {
            " | new neuron grown"
        } else {
            ""
        }
    );
    Ok(())
}

/// `manas ingest --file / --folder / --url`
fn cmd_ingest(
    file: Option<&str>,
    folder: Option<&str>,
    url: Option<&str>,
    dry_run: bool,
    brain_path: &Path,
) -> Result<(), ManasError> {
    let pipeline = IngestPipeline::new();
    let mut all_chunks = Vec::new();

    if let Some(f) = file {
        let chunks = pipeline.process(IngestSource::File(PathBuf::from(f)))?;
        all_chunks.extend(chunks);
    }
    if let Some(d) = folder {
        let chunks = pipeline.process(IngestSource::Folder(PathBuf::from(d)))?;
        all_chunks.extend(chunks);
    }
    if let Some(u) = url {
        let agent = AgentPipeline::new();
        match agent.scrape(u) {
            Ok(scraped) => {
                let normalized = manas_ingest::normalizer::normalize(&scraped);
                let chunks = manas_ingest::chunk_text(&normalized, 512, 64);
                for (i, chunk) in chunks.into_iter().enumerate() {
                    all_chunks.push(manas_ingest::TextChunk {
                        text: chunk,
                        source: Source::Internet { url: u.to_string() },
                        chunk_id: i as u64,
                        file_path: None,
                        url: Some(u.to_string()),
                    });
                }
            }
            Err(e) => eprintln!("Warning: failed to scrape '{}': {}", u, e),
        }
    }

    if dry_run {
        let sources = file.map(|_| 1).unwrap_or(0)
            + folder.map(|_| 1).unwrap_or(0)
            + url.map(|_| 1).unwrap_or(0);
        println!(
            "[dry-run] Would ingest {} chunks from {} source(s)",
            all_chunks.len(),
            sources
        );
        for chunk in &all_chunks {
            println!(
                "  chunk {} ({} chars)  src={:?}",
                chunk.chunk_id,
                chunk.text.len(),
                chunk.source
            );
        }
        return Ok(());
    }

    if all_chunks.is_empty() {
        println!("No content to ingest");
        return Ok(());
    }

    let brain = ManasBrain::new(brain_path);
    let mut network = load_or_create_network(&brain);
    let mut trainer = Trainer::new();
    restore_trainer_from_brain(&mut trainer, &brain);

    let mut total_tokens = 0u32;
    let mut total_loss = 0.0f32;
    let mut chunk_count = 0u32;

    for chunk in &all_chunks {
        // FIX 2 — stamp the exact file/url source onto neurons for this chunk
        trainer.source = chunk.source.clone();
        trainer.freshness_category = detect_freshness_category(&chunk.text);

        let report = trainer.learn(&mut network, &chunk.text)?;
        total_tokens += report.tokens_learned;
        total_loss += report.loss;
        chunk_count += 1;

        // Source-aware growth: grows at most 1 neuron per unique file/URL source
        // ensure_source_neuron internally checks duplicates before growing.
        trainer.ensure_source_neuron(&mut network)?;
    }

    network.total_texts_learned += 1;
    save_brain(&brain, &network, &trainer)?;

    let avg_loss = if chunk_count > 0 {
        total_loss / chunk_count as f32
    } else {
        0.0
    };
    println!(
        "Ingested {} chunks | {} tokens | avg loss: {:.4}",
        chunk_count, total_tokens, avg_loss
    );
    Ok(())
}

/// `manas query "question"`
fn cmd_query(text: &str, brain_path: &Path) -> Result<(), ManasError> {
    let agent = AgentPipeline::new();
    let brain = ManasBrain::new(brain_path);
    let mut network = load_or_create_network(&brain);
    let mut trainer = Trainer::new();
    restore_trainer_from_brain(&mut trainer, &brain);

    let freshness_cat = detect_freshness_category(text);
    let results = agent.search(text)?;

    if results.is_empty() {
        println!("No search results for: {}", text);
        return Ok(());
    }

    println!("Search results for \"{}\":", text);
    for (i, r) in results.iter().enumerate() {
        println!("  {}. {} — {}", i + 1, r.title, r.url);
    }

    let mut total_tokens = 0u32;
    let mut total_loss = 0.0f32;
    let mut page_count = 0u32;

    for result in &results {
        match agent.scrape(&result.url) {
            Ok(scraped) => {
                let chunks = manas_ingest::chunk_text(&scraped, 512, 64);
                for chunk in chunks {
                    // FIX 2 — neurons from web pages carry their URL
                    trainer.source = Source::Internet {
                        url: result.url.clone(),
                    };
                    trainer.freshness_category = freshness_cat;

                    let report = trainer.learn(&mut network, &chunk)?;
                    total_tokens += report.tokens_learned;
                    total_loss += report.loss;
                    page_count += 1;
                }
            }
            Err(_) => continue,
        }
    }

    network.total_texts_learned += 1;
    save_brain(&brain, &network, &trainer)?;

    let avg_loss = if page_count > 0 {
        total_loss / page_count as f32
    } else {
        0.0
    };
    println!(
        "Learned from {} pages | {} tokens | avg loss: {:.4}",
        page_count, total_tokens, avg_loss
    );
    Ok(())
}

/// `manas refresh [--category fast]`
fn cmd_refresh(category: Option<&str>, brain_path: &Path) -> Result<(), ManasError> {
    let brain = ManasBrain::new(brain_path);
    if !brain.path.exists() {
        println!("No brain file found at {}", brain.path.display());
        return Ok(());
    }

    let mut network = brain.load()?;
    let checker = FreshnessChecker::new();
    let agent = AgentPipeline::new();
    let mut trainer = Trainer::new();
    restore_trainer_from_brain(&mut trainer, &brain);

    let stale_ids = match category {
        Some(cat) => {
            let n = parse_freshness_category(cat)?;
            checker.find_stale_by_category(&network, n)
        }
        None => checker.find_stale(&network),
    };

    if stale_ids.is_empty() {
        println!("No stale neurons found");
        return Ok(());
    }

    println!("Found {} stale neuron(s)", stale_ids.len());

    let search_query = category.unwrap_or("latest updates");
    let results = agent.search(search_query)?;
    if results.is_empty() {
        println!("No search results to refresh with");
        return Ok(());
    }

    let mut refreshed_count = 0u32;
    let mut total_tokens = 0u32;

    for result in &results {
        match agent.scrape(&result.url) {
            Ok(scraped) => {
                let freshness_cat = detect_freshness_category(&scraped);
                let chunks = manas_ingest::chunk_text(&scraped, 512, 64);
                for chunk in chunks {
                    // FIX 2 — refreshed neurons carry the URL they were updated from
                    trainer.source = Source::Internet {
                        url: result.url.clone(),
                    };
                    trainer.freshness_category = freshness_cat;

                    let report = trainer.learn(&mut network, &chunk)?;
                    total_tokens += report.tokens_learned;
                    refreshed_count += 1;
                }
            }
            Err(_) => continue,
        }
    }

    network.total_texts_learned += 1;
    save_brain(&brain, &network, &trainer)?;
    println!(
        "Refreshed {} chunks | {} tokens learned",
        refreshed_count, total_tokens
    );
    Ok(())
}

/// `manas inspect`
fn cmd_inspect(brain_path: &Path) -> Result<(), ManasError> {
    let brain = ManasBrain::new(brain_path);
    if !brain.path.exists() {
        println!("No brain file found at {}", brain.path.display());
        return Ok(());
    }

    match brain.inspect() {
        Ok(stats) => {
            println!("{}", "━".repeat(35));
            println!(" Manas Brain — {}", stats.file_path);
            println!("{}", "━".repeat(35));
            println!(" Neurons       : {}", stats.neuron_count);
            println!(" Layers        : {}", stats.layer_count);
            println!(" Vocab size    : {}", stats.vocab_size);
            println!(" Brain size    : {} bytes", stats.brain_size);
            println!(" Texts learned : {}", stats.total_texts_learned);
            println!(" Last updated  : {}", format_duration(stats.last_modified));
            println!("{}", "━".repeat(35));
        }
        Err(ManasError::FileNotFound(_)) => {
            println!("No brain file found at {}", brain.path.display());
        }
        Err(e) => return Err(e),
    }
    Ok(())
}

/// `manas files`
fn cmd_files(brain_path: &Path) -> Result<(), ManasError> {
    let brain = ManasBrain::new(brain_path);
    if !brain.path.exists() {
        println!("No brain file found at {}", brain.path.display());
        return Ok(());
    }

    let network = brain.load()?;
    let mut files: std::collections::BTreeMap<String, u32> = std::collections::BTreeMap::new();

    for (_, n) in network.all_neurons() {
        if let Source::LocalFile { path } = &n.source {
            *files.entry(path.clone()).or_insert(0) += 1;
        }
    }

    if files.is_empty() {
        println!("No files have been ingested yet");
        return Ok(());
    }

    println!("{} ingested file(s):", files.len());
    for (path, count) in &files {
        println!("  {} — {} neuron(s)", path, count);
    }
    Ok(())
}

/// `manas trace "topic"`
fn cmd_trace(text: &str, brain_path: &Path) -> Result<(), ManasError> {
    let brain = ManasBrain::new(brain_path);
    if !brain.path.exists() {
        println!("No brain file found at {}", brain.path.display());
        return Ok(());
    }

    let network = brain.load()?;
    let mut trainer = Trainer::new();
    restore_trainer_from_brain(&mut trainer, &brain);

    let tokens = trainer.tokenizer.encode(text);
    if tokens.is_empty() {
        println!("No tokens found in query");
        return Ok(());
    }
    for &id in &tokens {
        trainer.embedder.embed_or_init(id);
    }
    let input = trainer.embedder.average_embed(&tokens);

    if network.layers.is_empty() {
        println!("Network has no layers yet");
        return Ok(());
    }

    let (_output, layer_acts) = network.forward_with_activations(&input);

    let mut all_acts: Vec<(u64, u32, f32)> = Vec::new();
    for (layer_idx, acts) in layer_acts.iter().enumerate() {
        for (nid, val) in acts {
            all_acts.push((*nid, layer_idx as u32, *val));
        }
    }
    all_acts.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    let top = &all_acts[..all_acts.len().min(10)];
    let all_neurons: Vec<(u32, &Neuron)> = network.all_neurons();

    println!("Top {} activated neurons:", top.len());
    for (nid, layer_id, act_val) in top {
        if let Some((_, n)) = all_neurons.iter().find(|(_, n)| n.id == *nid) {
            // FIX 2 — source is now populated so this will show real values
            let src_desc = match &n.source {
                Source::RawText => "raw text".to_string(),
                Source::LocalFile { path } => path.clone(),
                Source::Internet { url } => url.clone(),
                Source::Unknown => "unknown".to_string(),
            };
            println!(
                "  n{:<6} L{}  act={:.4}  imp={:.3}  fresh={}  src={}",
                nid, layer_id, act_val, n.importance_score, n.freshness_category, src_desc
            );
        }
    }

    let result = decode(&network, &trainer.embedder, &trainer.tokenizer, text);
    if !result.tokens.is_empty() {
        println!("\nClosest known tokens (decoded):");
        for (word, sim) in result.tokens.iter().take(10) {
            println!("  {:<20} sim={:.4}", word, sim);
        }
    }
    Ok(())
}

/// `manas export [--out path]`
fn cmd_export(out: Option<&str>, brain_path: &Path) -> Result<(), ManasError> {
    let dest = out.map(PathBuf::from).unwrap_or_else(|| {
        let mut p = PathBuf::from("brain_export.manas");
        let mut i = 1;
        while p.exists() {
            p = PathBuf::from(format!("brain_export_{}.manas", i));
            i += 1;
        }
        p
    });

    if !brain_path.exists() {
        return Err(ManasError::FileNotFound(brain_path.to_path_buf()));
    }

    std::fs::copy(brain_path, &dest).map_err(|e| ManasError::FileReadError {
        path: brain_path.to_path_buf(),
        source: e,
    })?;

    println!("Exported brain to {}", dest.display());
    Ok(())
}

/// `manas import --file path`
fn cmd_import(file: &str, brain_path: &Path) -> Result<(), ManasError> {
    let src = PathBuf::from(file);
    if !src.exists() {
        return Err(ManasError::FileNotFound(src));
    }

    std::fs::copy(&src, brain_path).map_err(|e| ManasError::FileReadError {
        path: brain_path.to_path_buf(),
        source: e,
    })?;

    println!("Imported brain from {} to {}", file, brain_path.display());
    Ok(())
}

/// `manas verify`
fn cmd_verify(brain_path: &Path) -> Result<(), ManasError> {
    let brain = ManasBrain::new(brain_path);
    if !brain.path.exists() {
        println!("No brain file found at {}", brain.path.display());
        return Ok(());
    }

    match brain.verify() {
        Ok(true) => println!("Brain file integrity verified ✓"),
        Ok(false) => println!("Checksum mismatch — file may be corrupt"),
        Err(e) => return Err(e),
    }
    Ok(())
}

/// `manas neurons [--all]`
fn cmd_neurons(all: bool, brain_path: &Path) -> Result<(), ManasError> {
    let brain = ManasBrain::new(brain_path);
    if !brain.path.exists() {
        println!("No brain file found at {}", brain.path.display());
        return Ok(());
    }

    let network = brain.load()?;
    let neurons = network.all_neurons();

    if neurons.is_empty() {
        println!("No neurons in brain");
        return Ok(());
    }

    let limit = if all {
        neurons.len()
    } else {
        20.min(neurons.len())
    };
    println!("{} neuron(s) (showing {}):", neurons.len(), limit);

    for (layer_id, n) in neurons.iter().take(limit) {
        let prot = match n.protection_level {
            manas_core::ProtectionLevel::Frozen => "FROZEN",
            manas_core::ProtectionLevel::Guarded => "guarded",
            manas_core::ProtectionLevel::Open => "open",
        };
        let fresh = match n.freshness_category {
            0 => "timeless",
            1 => "slow",
            2 => "fast",
            3 => "realtime",
            _ => "?",
        };
        // FIX 2 — source now shown correctly after learn/ingest
        let src = match &n.source {
            Source::RawText => "raw-text".to_string(),
            Source::LocalFile { path } => format!("file:{}", path),
            Source::Internet { url } => format!("web:{}", url),
            Source::Unknown => "unknown".to_string(),
        };
        println!(
            "  n{:<6} L{}  w={}  imp={:.3}  {} {}  src={}",
            n.id,
            layer_id,
            n.weights.len(),
            n.importance_score,
            prot,
            fresh,
            src
        );
    }

    if !all && neurons.len() > 20 {
        println!(
            "  ... and {} more (use --all to show all)",
            neurons.len() - 20
        );
    }
    Ok(())
}

/// `manas restore [--all]`
fn cmd_restore(all: bool, brain_path: &Path) -> Result<(), ManasError> {
    let brain = ManasBrain::new(brain_path);
    if !brain.path.exists() {
        println!("No brain file found at {}", brain.path.display());
        return Ok(());
    }

    let mut network = brain.load()?;
    let archived = brain.load_archive()?;

    if archived.is_empty() {
        println!("No archived neurons to restore");
        return Ok(());
    }

    let to_restore: Vec<Neuron> = if all {
        archived
    } else {
        archived.into_iter().take(10).collect()
    };

    let mut restored = 0u32;
    for neuron in &to_restore {
        if let Some(layer) = network.layers.last_mut() {
            let mut n = neuron.clone();
            n.protection_level = manas_core::ProtectionLevel::Open;
            layer.neurons.push(n);
            network.total_neurons += 1;
            restored += 1;
        }
    }

    brain.save(&network)?;
    println!("Restored {} neuron(s) to last layer", restored);
    Ok(())
}

/// `manas tag "topic" --freshness fast`
fn cmd_tag(text: &str, freshness: &str, brain_path: &Path) -> Result<(), ManasError> {
    let cat = parse_freshness_category(freshness)?;

    let brain = ManasBrain::new(brain_path);
    if !brain.path.exists() {
        println!("No brain file found at {}", brain.path.display());
        return Ok(());
    }

    let mut network = brain.load()?;
    let mut trainer = Trainer::new();
    restore_trainer_from_brain(&mut trainer, &brain);

    let tokens = trainer.tokenizer.encode(text);
    for &id in &tokens {
        trainer.embedder.embed_or_init(id);
    }
    let input = trainer.embedder.average_embed(&tokens);

    let (_output, layer_acts) = network.forward_with_activations(&input);

    let mut activated_ids: Vec<u64> = Vec::new();
    for acts in &layer_acts {
        for (nid, _) in acts {
            if !activated_ids.contains(nid) {
                activated_ids.push(*nid);
            }
        }
    }

    let mut tagged = 0u32;
    for layer in &mut network.layers {
        for neuron in &mut layer.neurons {
            if activated_ids.contains(&neuron.id) || text == "all" {
                neuron.freshness_category = cat;
                tagged += 1;
            }
        }
    }

    brain.save(&network)?;
    println!("Tagged {} neuron(s) as {}", tagged, freshness);
    Ok(())
}

// ─── Tiny helper ──────────────────────────────────────────────────────────────

fn parse_freshness_category(s: &str) -> Result<u8, ManasError> {
    match s.to_lowercase().as_str() {
        "timeless" | "0" => Ok(0),
        "slow" | "1" => Ok(1),
        "fast" | "2" => Ok(2),
        "realtime" | "3" => Ok(3),
        other => {
            println!(
                "Unknown category '{}'. Use: timeless, slow, fast, realtime",
                other
            );
            Err(ManasError::GrowthFailed(format!(
                "unknown category: {}",
                other
            )))
        }
    }
}
