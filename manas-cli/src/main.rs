use clap::{Parser, Subcommand};
use manas_agent::{AgentPipeline, FreshnessChecker};
use manas_core::{ManasError, Network, Neuron, Source};
use manas_ingest::{IngestPipeline, IngestSource};
use manas_language::{
    LanguageMeta, NextTokenPredictor, SequenceMemory, TransformerLanguageModel,
    TransformerPredictor, TransformerTrainingSafety, build_sequence_examples,
    generate_text_with_memory, generate_text_with_transformer, language_meta_path, seq_memory_path,
    text_hash, train_next_token_examples, train_transformer_output_head_with_safety,
    transformer_model_path,
};
use manas_learn::{Trainer, TrainerSnapshot, decode, detect_freshness_category};
use manas_store::ManasBrain;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
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
    Teach {
        input: String,
        #[arg(long, default_value = "5")]
        max_context: usize,
        #[arg(long, default_value = "10")]
        epochs: usize,
        #[arg(long, default_value = "0.05")]
        learning_rate: f32,
        #[arg(long)]
        train_transformer: bool,
        #[arg(long, default_value = "0.01")]
        transformer_learning_rate: f32,
        #[arg(long, default_value = "5.0")]
        transformer_max_grad_norm: f32,
        #[arg(long, default_value = "50.0")]
        transformer_max_loss: f32,
        #[arg(long)]
        no_transformer_rollback: bool,
        #[arg(long)]
        dry_run: bool,
    },
    Query {
        text: String,
        #[arg(long)]
        answer: bool,
    },
    Ask {
        text: String,
        #[arg(long, default_value = "5")]
        top_k: usize,
        #[arg(long, default_value = "80")]
        max_answer_tokens: usize,
        #[arg(long)]
        hide_sources: bool,
        #[arg(long)]
        no_generate: bool,
        #[arg(long)]
        use_transformer: bool,
    },
    Refresh {
        #[arg(long)]
        category: Option<String>,
    },
    Inspect {
        #[arg(long)]
        verbose: bool,
    },
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
    TrainLanguage {
        text: String,
        #[arg(long, default_value = "5")]
        max_context: usize,
        #[arg(long, default_value = "10")]
        epochs: usize,
        #[arg(long, default_value = "0.05")]
        learning_rate: f32,
        #[arg(long)]
        train_transformer: bool,
        #[arg(long, default_value = "0.01")]
        transformer_learning_rate: f32,
        #[arg(long, default_value = "10")]
        max_new_neurons: usize,
        #[arg(long)]
        no_grow: bool,
        #[arg(long, default_value = "5.0")]
        transformer_max_grad_norm: f32,
        #[arg(long, default_value = "50.0")]
        transformer_max_loss: f32,
        #[arg(long)]
        no_transformer_rollback: bool,
    },
    PredictNext {
        text: String,
        #[arg(long, default_value = "5")]
        max_context: usize,
        #[arg(long, default_value = "10")]
        top_k: usize,
        #[arg(long)]
        use_transformer: bool,
        #[arg(long)]
        transformer_only: bool,
    },
    Generate {
        prompt: String,
        #[arg(long, default_value = "20")]
        max_tokens: usize,
        #[arg(long, default_value = "5")]
        max_context: usize,
        #[arg(long, default_value = "1")]
        top_k: usize,
        #[arg(long, default_value = "1.0")]
        temperature: f32,
        #[arg(long)]
        use_transformer: bool,
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
        Commands::Teach {
            input,
            max_context,
            epochs,
            learning_rate,
            train_transformer,
            transformer_learning_rate,
            transformer_max_grad_norm,
            transformer_max_loss,
            no_transformer_rollback,
            dry_run,
        } => cmd_teach(
            input,
            TeachOptions {
                max_context: *max_context,
                epochs: *epochs,
                learning_rate: *learning_rate,
                train_transformer: *train_transformer,
                transformer_learning_rate: *transformer_learning_rate,
                transformer_max_grad_norm: *transformer_max_grad_norm,
                transformer_max_loss: *transformer_max_loss,
                no_transformer_rollback: *no_transformer_rollback,
                dry_run: *dry_run,
            },
            &brain_path,
        ),
        Commands::Query { text, answer } => {
            if *answer {
                cmd_ask(text, AskOptions::default(), &brain_path)
            } else {
                cmd_query(text, &brain_path)
            }
        }
        Commands::Ask {
            text,
            top_k,
            max_answer_tokens,
            hide_sources,
            no_generate,
            use_transformer,
        } => cmd_ask(
            text,
            AskOptions {
                top_k: *top_k,
                max_answer_tokens: *max_answer_tokens,
                show_sources: !hide_sources,
                no_generate: *no_generate,
                use_transformer: *use_transformer,
            },
            &brain_path,
        ),
        Commands::Refresh { category } => cmd_refresh(category.as_deref(), &brain_path),
        Commands::Inspect { verbose } => cmd_inspect(*verbose, &brain_path),
        Commands::Files => cmd_files(&brain_path),
        Commands::Trace { text } => cmd_trace(text, &brain_path),
        Commands::Export { out } => cmd_export(out.as_deref(), &brain_path),
        Commands::Import { file } => cmd_import(file, &brain_path),
        Commands::Verify => cmd_verify(&brain_path),
        Commands::Neurons { all } => cmd_neurons(*all, &brain_path),
        Commands::Restore { all } => cmd_restore(*all, &brain_path),
        Commands::Tag { text, freshness } => cmd_tag(text, freshness, &brain_path),
        Commands::TrainLanguage {
            text,
            max_context,
            epochs,
            learning_rate,
            train_transformer,
            transformer_learning_rate,
            max_new_neurons,
            no_grow,
            transformer_max_grad_norm,
            transformer_max_loss,
            no_transformer_rollback,
        } => cmd_train_language(
            text,
            *max_context,
            *epochs,
            *learning_rate,
            *train_transformer,
            *transformer_learning_rate,
            *max_new_neurons,
            *no_grow,
            *transformer_max_grad_norm,
            *transformer_max_loss,
            *no_transformer_rollback,
            &brain_path,
        ),
        Commands::PredictNext {
            text,
            max_context,
            top_k,
            use_transformer,
            transformer_only,
        } => cmd_predict_next(
            text,
            *max_context,
            *top_k,
            *use_transformer,
            *transformer_only,
            &brain_path,
        ),
        Commands::Generate {
            prompt,
            max_tokens,
            max_context,
            top_k,
            temperature,
            use_transformer,
        } => cmd_generate(
            prompt,
            *max_tokens,
            *max_context,
            *top_k,
            *temperature,
            *use_transformer,
            &brain_path,
        ),
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

// ─── Teach helpers ────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TeachInputMode {
    Text,
    File,
    Folder,
}

impl TeachInputMode {
    fn as_str(self) -> &'static str {
        match self {
            TeachInputMode::Text => "text",
            TeachInputMode::File => "file",
            TeachInputMode::Folder => "folder",
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct TeachOptions {
    max_context: usize,
    epochs: usize,
    learning_rate: f32,
    train_transformer: bool,
    transformer_learning_rate: f32,
    transformer_max_grad_norm: f32,
    transformer_max_loss: f32,
    no_transformer_rollback: bool,
    dry_run: bool,
}

#[derive(Clone, Debug)]
struct TeachItem {
    text: String,
    source: Source,
}

#[derive(Clone, Debug)]
struct TeachDiscovery {
    mode: TeachInputMode,
    items: Vec<TeachItem>,
    files_discovered: usize,
    files_skipped: usize,
    read_errors: Vec<String>,
}

#[derive(Clone, Debug, Default)]
struct TeachTransformerSummary {
    training_ran: bool,
    examples: usize,
    invalid_updates: usize,
    unstable_updates: usize,
    rolled_back: bool,
    output_head_trained: bool,
    ffn_trained: bool,
    attention_projection_o_trained: bool,
    attention_projection_v_trained: bool,
    attention_projection_q_trained: bool,
    attention_projection_k_trained: bool,
}

impl TeachTransformerSummary {
    fn record(&mut self, report: &manas_language::TransformerTrainReport) {
        self.training_ran = true;
        self.examples += report.examples;
        self.invalid_updates += report.invalid_updates;
        self.unstable_updates += report.unstable_updates;
        self.rolled_back |= report.rolled_back;
        self.output_head_trained = report.output_head_trained;
        self.ffn_trained = report.ffn_trained;
        self.attention_projection_o_trained = report.attention_projection_o_trained;
        self.attention_projection_v_trained = report.attention_projection_v_trained;
        self.attention_projection_q_trained = report.attention_projection_q_trained;
        self.attention_projection_k_trained = report.attention_projection_k_trained;
    }

    fn attention_trained(&self) -> bool {
        self.attention_projection_o_trained
            || self.attention_projection_v_trained
            || self.attention_projection_q_trained
            || self.attention_projection_k_trained
    }
}

#[derive(Clone, Debug)]
struct TeachReport {
    mode: TeachInputMode,
    files_discovered: usize,
    files_taught: usize,
    files_skipped: usize,
    read_errors: Vec<String>,
    text_chunks_learned: usize,
    language_examples: usize,
    language_tokens: u32,
    transformer: TeachTransformerSummary,
    dry_run: bool,
}

fn input_looks_like_path(input: &str) -> bool {
    input.contains('/')
        || input.contains('\\')
        || matches!(
            teach_file_extension(Path::new(input)).as_str(),
            "md" | "txt"
        )
        || input.ends_with(std::path::MAIN_SEPARATOR)
}

fn teach_file_extension(path: &Path) -> String {
    path.extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("")
        .to_lowercase()
}

fn teach_supported_file(path: &Path) -> bool {
    matches!(teach_file_extension(path).as_str(), "md" | "txt")
}

fn read_teach_file(path: &Path) -> Result<String, ManasError> {
    let contents = fs::read_to_string(path).map_err(|e| ManasError::FileReadError {
        path: path.to_path_buf(),
        source: e,
    })?;
    let ext = teach_file_extension(path);
    let parsed = manas_ingest::file_reader::parse_by_extension(&contents, &ext)?;
    Ok(manas_ingest::normalizer::normalize(&parsed))
}

fn teach_item_from_file(path: &Path) -> Result<Option<TeachItem>, ManasError> {
    if !teach_supported_file(path) {
        return Err(ManasError::UnsupportedFileType(teach_file_extension(path)));
    }

    let text = read_teach_file(path)?;
    if text.trim().is_empty() {
        return Ok(None);
    }

    let path_str = path.display().to_string();
    Ok(Some(TeachItem {
        text,
        source: Source::LocalFile { path: path_str },
    }))
}

fn collect_folder_files(dir: &Path, files: &mut Vec<PathBuf>, errors: &mut Vec<String>) {
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) => {
            errors.push(format!("{}: {}", dir.display(), e));
            return;
        }
    };

    let mut paths = Vec::new();
    for entry in entries {
        match entry {
            Ok(entry) => paths.push(entry.path()),
            Err(e) => errors.push(format!("{}: {}", dir.display(), e)),
        }
    }
    paths.sort();

    for path in paths {
        if path.is_dir() {
            collect_folder_files(&path, files, errors);
        } else if path.is_file() {
            files.push(path);
        }
    }
}

fn collect_teach_items(input: &str) -> Result<TeachDiscovery, ManasError> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err(ManasError::GrowthFailed("no teachable text".to_string()));
    }

    let input_path = PathBuf::from(trimmed);
    if input_path.exists() {
        if input_path.is_file() {
            let item = teach_item_from_file(&input_path)?.ok_or_else(|| {
                ManasError::GrowthFailed(format!("no teachable text in {}", input_path.display()))
            })?;
            return Ok(TeachDiscovery {
                mode: TeachInputMode::File,
                items: vec![item],
                files_discovered: 1,
                files_skipped: 0,
                read_errors: Vec::new(),
            });
        }

        if input_path.is_dir() {
            let mut paths = Vec::new();
            let mut read_errors = Vec::new();
            collect_folder_files(&input_path, &mut paths, &mut read_errors);

            let mut items = Vec::new();
            let mut files_skipped = read_errors.len();
            for path in &paths {
                if !teach_supported_file(path) {
                    files_skipped += 1;
                    continue;
                }

                match teach_item_from_file(path) {
                    Ok(Some(item)) => items.push(item),
                    Ok(None) => files_skipped += 1,
                    Err(e) => {
                        files_skipped += 1;
                        read_errors.push(format!("{}: {}", path.display(), e));
                    }
                }
            }

            return Ok(TeachDiscovery {
                mode: TeachInputMode::Folder,
                items,
                files_discovered: paths.len(),
                files_skipped,
                read_errors,
            });
        }

        return Err(ManasError::FileReadError {
            path: input_path,
            source: std::io::Error::other("input is neither file nor folder"),
        });
    }

    if input_looks_like_path(trimmed) {
        return Err(ManasError::FileNotFound(input_path));
    }

    Ok(TeachDiscovery {
        mode: TeachInputMode::Text,
        items: vec![TeachItem {
            text: input.to_string(),
            source: Source::RawText,
        }],
        files_discovered: 0,
        files_skipped: 0,
        read_errors: Vec::new(),
    })
}

fn item_core_chunks(item: &TeachItem) -> Vec<String> {
    match item.source {
        Source::LocalFile { .. } => manas_ingest::chunk_text(
            &item.text,
            manas_ingest::CHUNK_SIZE,
            manas_ingest::CHUNK_OVERLAP,
        ),
        _ => vec![item.text.clone()],
    }
}

fn print_teach_report(report: &TeachReport, options: TeachOptions) {
    if report.dry_run {
        println!("Teaching dry run");
    } else {
        println!("Teaching complete");
    }
    println!();
    println!("Input");
    println!("  mode                  : {}", report.mode.as_str());
    println!("  files discovered      : {}", report.files_discovered);
    println!("  files taught          : {}", report.files_taught);
    println!("  files skipped         : {}", report.files_skipped);
    if !report.read_errors.is_empty() {
        println!("  read errors           : {}", report.read_errors.len());
        for error in &report.read_errors {
            println!("  warning               : {}", error);
        }
    }

    println!();
    println!("Core memory");
    println!(
        "  source ingest         : {}",
        if matches!(report.mode, TeachInputMode::File | TeachInputMode::Folder) {
            "yes"
        } else {
            "no"
        }
    );
    println!("  text chunks learned   : {}", report.text_chunks_learned);
    println!(
        "  source metadata       : {}",
        if matches!(report.mode, TeachInputMode::File | TeachInputMode::Folder) {
            "preserved"
        } else {
            "raw text"
        }
    );

    println!();
    println!("Language memory");
    println!(
        "  sequence training     : {}",
        if report.dry_run { "planned" } else { "yes" }
    );
    println!("  max context           : {}", options.max_context);
    println!("  epochs                : {}", options.epochs);
    println!("  total examples        : {}", report.language_examples);
    println!("  total tokens          : {}", report.language_tokens);

    println!();
    println!("Transformer");
    println!(
        "  transformer training  : {}",
        if options.train_transformer {
            "yes"
        } else {
            "no"
        }
    );
    if options.train_transformer {
        println!(
            "  output head           : {}",
            if report.transformer.output_head_trained {
                "trained"
            } else if report.dry_run {
                "planned"
            } else {
                "untrained"
            }
        );
        println!(
            "  feed-forward          : {}",
            if report.transformer.ffn_trained {
                "trained"
            } else if report.dry_run {
                "planned"
            } else {
                "untrained"
            }
        );
        println!(
            "  attention             : {}",
            if report.transformer.attention_trained() {
                "partial"
            } else if report.dry_run {
                "planned"
            } else {
                "no"
            }
        );
        println!(
            "  projections           : {}",
            format_attention_projections(
                report.transformer.attention_projection_o_trained,
                report.transformer.attention_projection_v_trained,
                report.transformer.attention_projection_q_trained,
                report.transformer.attention_projection_k_trained,
            )
        );
    }

    println!();
    println!("Safety");
    println!(
        "  invalid updates       : {}",
        report.transformer.invalid_updates
    );
    println!(
        "  unstable updates      : {}",
        report.transformer.unstable_updates
    );
    println!(
        "  rolled back           : {}",
        if report.transformer.rolled_back {
            "yes"
        } else {
            "no"
        }
    );
}

fn initial_teach_report(discovery: &TeachDiscovery, dry_run: bool) -> TeachReport {
    TeachReport {
        mode: discovery.mode,
        files_discovered: discovery.files_discovered,
        files_taught: if matches!(discovery.mode, TeachInputMode::Text) {
            0
        } else {
            discovery.items.len()
        },
        files_skipped: discovery.files_skipped,
        read_errors: discovery.read_errors.clone(),
        text_chunks_learned: 0,
        language_examples: 0,
        language_tokens: 0,
        transformer: TeachTransformerSummary::default(),
        dry_run,
    }
}

// ─── Local answer helpers ─────────────────────────────────────────────────────

const DIRECT_ANSWER_THRESHOLD: f32 = 0.75;
const WEAK_EVIDENCE_THRESHOLD: f32 = 0.25;

#[derive(Clone, Copy, Debug)]
struct AskOptions {
    top_k: usize,
    max_answer_tokens: usize,
    show_sources: bool,
    no_generate: bool,
    use_transformer: bool,
}

impl Default for AskOptions {
    fn default() -> Self {
        Self {
            top_k: 5,
            max_answer_tokens: 80,
            show_sources: true,
            no_generate: false,
            use_transformer: false,
        }
    }
}

#[derive(Clone, Debug)]
struct LocalSourceCandidate {
    path: String,
    neuron_count: u32,
    max_importance: f32,
}

#[derive(Clone, Debug)]
struct LocalEvidenceSnippet {
    text: String,
    source: String,
    score: f32,
    source_neuron_count: u32,
    source_importance: f32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum LocalAnswerKind {
    Answer,
    WeakEvidence,
    NoEvidence,
}

#[derive(Clone, Debug)]
struct LocalAnswerReport {
    kind: LocalAnswerKind,
    answer: Option<String>,
    sources: Vec<String>,
}

fn answer_stopwords() -> HashSet<&'static str> {
    [
        "a", "an", "and", "are", "as", "be", "by", "do", "does", "for", "from", "how", "i", "in",
        "into", "is", "it", "of", "on", "or", "tell", "that", "the", "this", "to", "was", "what",
        "when", "where", "which", "who", "why", "with",
    ]
    .into_iter()
    .collect()
}

fn answer_tokens(text: &str) -> Vec<String> {
    let stopwords = answer_stopwords();
    let mut tokens = Vec::new();
    let mut current = String::new();

    for ch in text.to_lowercase().chars() {
        if ch.is_alphanumeric() || ch == '-' || ch == '\'' {
            current.push(ch);
        } else if !current.is_empty() {
            if !stopwords.contains(current.as_str()) {
                tokens.push(current.clone());
            }
            current.clear();
        }
    }

    if !current.is_empty() && !stopwords.contains(current.as_str()) {
        tokens.push(current);
    }

    tokens
}

fn sentence_split(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut start = 0usize;

    for (idx, ch) in text.char_indices() {
        let next = text[idx + ch.len_utf8()..].chars().next();
        let sentence_boundary = match ch {
            '.' | '!' | '?' => next.is_none_or(|c| c.is_whitespace()),
            '\n' | '\r' => true,
            _ => false,
        };

        if sentence_boundary {
            let end = if matches!(ch, '.' | '!' | '?') {
                idx + ch.len_utf8()
            } else {
                idx
            };
            let sentence = text[start..end].trim();
            if !sentence.is_empty() {
                sentences.push(sentence.to_string());
            }
            start = idx + ch.len_utf8();
        }
    }

    let rest = text[start..].trim();
    if !rest.is_empty() {
        sentences.push(rest.to_string());
    }

    sentences
}

fn collect_local_source_candidates(network: &Network) -> Vec<LocalSourceCandidate> {
    let mut by_path: BTreeMap<String, (u32, f32)> = BTreeMap::new();

    for (_, neuron) in network.all_neurons() {
        if let Source::LocalFile { path } = &neuron.source {
            let entry = by_path.entry(path.clone()).or_insert((0, 0.0));
            entry.0 += 1;
            entry.1 = entry.1.max(neuron.importance_score);
        }
    }

    by_path
        .into_iter()
        .map(
            |(path, (neuron_count, max_importance))| LocalSourceCandidate {
                path,
                neuron_count,
                max_importance,
            },
        )
        .collect()
}

fn read_source_snippets(candidates: &[LocalSourceCandidate]) -> Vec<LocalEvidenceSnippet> {
    let mut snippets = Vec::new();

    for candidate in candidates {
        let path = Path::new(&candidate.path);
        if !path.exists() || !teach_supported_file(path) {
            continue;
        }

        let text = match read_teach_file(path) {
            Ok(text) => text,
            Err(_) => continue,
        };

        for sentence in sentence_split(&text) {
            if !sentence.trim().is_empty() {
                snippets.push(LocalEvidenceSnippet {
                    text: sentence,
                    source: candidate.path.clone(),
                    score: 0.0,
                    source_neuron_count: candidate.neuron_count,
                    source_importance: candidate.max_importance,
                });
            }
        }
    }

    snippets
}

fn rank_answer_snippets(
    question: &str,
    snippets: &[LocalEvidenceSnippet],
    top_k: usize,
) -> Vec<LocalEvidenceSnippet> {
    let query_tokens = answer_tokens(question);
    if query_tokens.is_empty() {
        return Vec::new();
    }

    let query_set: HashSet<&str> = query_tokens.iter().map(|s| s.as_str()).collect();
    let focus = query_tokens.last().cloned();
    let lower_question = question.to_lowercase();
    let definitional_question =
        lower_question.starts_with("what ") || lower_question.starts_with("who ");

    let mut ranked = Vec::new();
    for snippet in snippets {
        let snippet_tokens = answer_tokens(&snippet.text);
        if snippet_tokens.is_empty() {
            continue;
        }
        let snippet_set: HashSet<&str> = snippet_tokens.iter().map(|s| s.as_str()).collect();
        let overlap = query_set
            .iter()
            .filter(|token| snippet_set.contains(**token))
            .count();

        if overlap == 0 {
            continue;
        }

        let mut scored = snippet.clone();
        let mut score = overlap as f32 / query_set.len() as f32;
        let lower_snippet = snippet.text.to_lowercase();

        if let Some(focus) = &focus {
            let starts_with_definition = lower_snippet.starts_with(&format!("{} is ", focus))
                || lower_snippet.starts_with(&format!("{} are ", focus));
            if starts_with_definition {
                score += 0.60;
            }
            if definitional_question && lower_snippet.starts_with(&format!("{} is not ", focus)) {
                score -= 0.25;
            }
        }

        let source_count_bonus = (snippet.source_neuron_count.min(10) as f32) * 0.01;
        let source_importance_bonus = snippet.source_importance.clamp(0.0, 1.0) * 0.05;
        score += source_count_bonus + source_importance_bonus;

        scored.score = score;
        ranked.push(scored);
    }

    ranked.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.source.cmp(&b.source))
            .then_with(|| a.text.cmp(&b.text))
    });
    ranked.truncate(top_k.max(1));
    ranked
}

fn truncate_answer_tokens(answer: &str, max_tokens: usize) -> String {
    if max_tokens == 0 {
        return String::new();
    }

    let parts: Vec<&str> = answer.split_whitespace().collect();
    if parts.len() <= max_tokens {
        answer.to_string()
    } else {
        format!("{}...", parts[..max_tokens].join(" "))
    }
}

fn unique_sources(snippets: &[LocalEvidenceSnippet]) -> Vec<String> {
    let mut sources = Vec::new();
    for snippet in snippets {
        if !sources.contains(&snippet.source) {
            sources.push(snippet.source.clone());
        }
    }
    sources
}

fn compose_local_answer(ranked: &[LocalEvidenceSnippet], options: AskOptions) -> LocalAnswerReport {
    let _generation_requested = options.use_transformer && !options.no_generate;

    if ranked.is_empty() {
        return LocalAnswerReport {
            kind: LocalAnswerKind::NoEvidence,
            answer: None,
            sources: Vec::new(),
        };
    }

    let best = &ranked[0];
    let sources = unique_sources(ranked);
    if best.score >= DIRECT_ANSWER_THRESHOLD {
        return LocalAnswerReport {
            kind: LocalAnswerKind::Answer,
            answer: Some(truncate_answer_tokens(
                &best.text,
                options.max_answer_tokens,
            )),
            sources,
        };
    }

    if best.score >= WEAK_EVIDENCE_THRESHOLD {
        return LocalAnswerReport {
            kind: LocalAnswerKind::WeakEvidence,
            answer: None,
            sources,
        };
    }

    LocalAnswerReport {
        kind: LocalAnswerKind::NoEvidence,
        answer: None,
        sources: Vec::new(),
    }
}

fn answer_local_question(
    question: &str,
    options: AskOptions,
    brain_path: &Path,
) -> Result<LocalAnswerReport, ManasError> {
    if question.trim().is_empty() || !brain_path.exists() {
        return Ok(LocalAnswerReport {
            kind: LocalAnswerKind::NoEvidence,
            answer: None,
            sources: Vec::new(),
        });
    }

    let brain = ManasBrain::new(brain_path);
    let network = brain.load()?;
    let candidates = collect_local_source_candidates(&network);
    let snippets = read_source_snippets(&candidates);
    let ranked = rank_answer_snippets(question, &snippets, options.top_k);
    Ok(compose_local_answer(&ranked, options))
}

fn format_local_answer_report(report: &LocalAnswerReport, show_sources: bool) -> String {
    match report.kind {
        LocalAnswerKind::Answer => {
            let mut out = format!("Answer\n  {}", report.answer.as_deref().unwrap_or_default());
            if show_sources && !report.sources.is_empty() {
                out.push_str("\n\nSources");
                for source in &report.sources {
                    out.push_str(&format!("\n  - {}", source));
                }
            }
            out
        }
        LocalAnswerKind::WeakEvidence => {
            let mut out =
                "I found related local memory, but not enough to answer confidently.".to_string();
            if show_sources && !report.sources.is_empty() {
                out.push_str("\n\nSources");
                for source in &report.sources {
                    out.push_str(&format!("\n  - {}", source));
                }
            }
            out
        }
        LocalAnswerKind::NoEvidence => "Not enough local memory to answer this yet.".to_string(),
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

/// `manas teach <input>`
fn cmd_teach(input: &str, options: TeachOptions, brain_path: &Path) -> Result<(), ManasError> {
    const TEACH_MAX_NEW_NEURONS: usize = 10;

    let discovery = collect_teach_items(input)?;
    let mut report = initial_teach_report(&discovery, options.dry_run);

    for item in &discovery.items {
        let chunks = item_core_chunks(item);
        report.text_chunks_learned += chunks.len();

        let mut preview_trainer = Trainer::new();
        let tokens = preview_trainer.tokenizer.encode(&item.text);
        report.language_tokens += tokens.len() as u32;
        report.language_examples += build_sequence_examples(&tokens, options.max_context).len();
    }

    if options.dry_run {
        print_teach_report(&report, options);
        return Ok(());
    }

    if discovery.items.is_empty() {
        return Err(ManasError::GrowthFailed(
            "no teachable .md or .txt content found".to_string(),
        ));
    }

    let brain = ManasBrain::new(brain_path);
    let mut network = load_or_create_network(&brain);
    let mut trainer = Trainer::new();
    restore_trainer_from_brain(&mut trainer, &brain);

    let langmeta_path = language_meta_path(brain_path);
    let mut langmeta = if langmeta_path.exists() {
        LanguageMeta::load_from_file(&langmeta_path)?
    } else {
        LanguageMeta::new()
    };

    let seq_path = seq_memory_path(brain_path);
    let mut seq_memory = if seq_path.exists() {
        SequenceMemory::load_from_file(&seq_path)?
    } else {
        SequenceMemory::new()
    };

    report.text_chunks_learned = 0;
    report.language_examples = 0;
    report.language_tokens = 0;

    for item in &discovery.items {
        for chunk in item_core_chunks(item) {
            trainer.source = item.source.clone();
            trainer.freshness_category = detect_freshness_category(&chunk);

            let _learn_report = trainer.learn(&mut network, &chunk)?;
            report.text_chunks_learned += 1;

            if matches!(trainer.source, Source::LocalFile { .. }) {
                trainer.ensure_source_neuron(&mut network)?;
            }
        }

        trainer.source = item.source.clone();
        trainer.freshness_category = detect_freshness_category(&item.text);

        let hash = text_hash(&item.text);
        let effective_max = if langmeta.is_known(hash) {
            0
        } else {
            TEACH_MAX_NEW_NEURONS
        };

        let language_report = train_next_token_examples(
            &mut network,
            &mut trainer,
            &mut seq_memory,
            &item.text,
            options.max_context,
            options.epochs,
            options.learning_rate,
            effective_max,
        )?;

        langmeta.record(hash, options.max_context, language_report.examples_count);
        report.language_examples += language_report.examples_count;
        report.language_tokens += language_report.tokens_learned;
        network.total_texts_learned += 1;
    }

    if options.train_transformer {
        let embed_dim = trainer.embedder.dim;
        let hidden_dim = (embed_dim * 2).max(8);
        let transformer_path = transformer_model_path(brain_path);
        let mut model = if transformer_path.exists() {
            TransformerLanguageModel::load_from_file(&transformer_path)?
        } else {
            let mut vocab_order: Vec<u32> = trainer.embedder.table.keys().copied().collect();
            vocab_order.sort();
            TransformerLanguageModel::new(embed_dim, hidden_dim, vocab_order)
        };

        let tf_epochs = options.epochs.max(10);
        let safety = TransformerTrainingSafety {
            max_gradient_norm: options.transformer_max_grad_norm,
            max_loss: options.transformer_max_loss,
            rollback_on_unstable: !options.no_transformer_rollback,
            ..TransformerTrainingSafety::default()
        };

        for item in &discovery.items {
            let tokens = trainer.tokenizer.encode(&item.text);
            let examples = build_sequence_examples(&tokens, options.max_context);
            let tf_report = train_transformer_output_head_with_safety(
                &mut model,
                &trainer.embedder,
                &examples,
                options.max_context,
                tf_epochs,
                options.transformer_learning_rate,
                options.learning_rate,
                &safety,
            );
            report.transformer.record(&tf_report);
        }

        if manas_language::is_finite_model(&model) {
            model.save_to_file(&transformer_path)?;
        } else {
            println!("Warning: transformer model corrupted — not saving");
        }
    }

    langmeta.save_to_file(&langmeta_path)?;
    save_brain(&brain, &network, &trainer)?;
    seq_memory.save_to_file(&seq_path)?;

    print_teach_report(&report, options);
    Ok(())
}

/// `manas ask "question"`
fn cmd_ask(question: &str, options: AskOptions, brain_path: &Path) -> Result<(), ManasError> {
    let report = answer_local_question(question, options, brain_path)?;
    println!(
        "{}",
        format_local_answer_report(&report, options.show_sources)
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
                    trainer.source = Source::Internet {
                        url: result.url.clone(),
                    };
                    trainer.freshness_category = freshness_cat;

                    let report = trainer.learn(&mut network, &chunk)?;
                    total_tokens += report.tokens_learned;
                    total_loss += report.loss;
                    page_count += 1;

                    trainer.ensure_source_neuron(&mut network)?;
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

/// `manas inspect [--verbose]`
fn cmd_inspect(verbose: bool, brain_path: &Path) -> Result<(), ManasError> {
    let brain = ManasBrain::new(brain_path);
    if !brain.path.exists() {
        println!("No brain file found at {}", brain.path.display());
        return Ok(());
    }

    let stats = match brain.inspect() {
        Ok(s) => s,
        Err(ManasError::FileNotFound(_)) => {
            println!("No brain file found at {}", brain.path.display());
            return Ok(());
        }
        Err(e) => return Err(e),
    };

    let network = brain.load().ok();
    let net_params = network.as_ref().map(|n| n.parameter_count()).unwrap_or(0);
    let net_neurons = stats.neuron_count;
    let net_layers = stats.layer_count;

    let embed_params = brain
        .load_vocab()
        .ok()
        .map(|v| v.values().map(|(_, e)| e.len() as u64).sum::<u64>())
        .unwrap_or(0);

    // ── Sidecar file sizes ──────────────────────────────────────────
    let seq_path = seq_memory_path(brain_path);
    let tf_path = transformer_model_path(brain_path);
    let langmeta_path = language_meta_path(brain_path);

    let seq_bytes = file_size(&seq_path);
    let tf_bytes = file_size(&tf_path);
    let langmeta_bytes = file_size(&langmeta_path);

    // ── Sequence memory stats ───────────────────────────────────────
    let seq_entries = seq_bytes.and_then(|_| {
        SequenceMemory::load_from_file(&seq_path)
            .ok()
            .map(|sm| sm.transitions.len())
    });

    // ── Language metadata stats ─────────────────────────────────────
    let langmeta = langmeta_bytes.and_then(|_| LanguageMeta::load_from_file(&langmeta_path).ok());
    let total_training_runs = langmeta
        .as_ref()
        .map(|lm| {
            lm.texts
                .values()
                .map(|t| t.trained_count as u64)
                .sum::<u64>()
        })
        .unwrap_or(0);
    let unique_texts = langmeta
        .as_ref()
        .map(|lm| lm.texts.len() as u64)
        .unwrap_or(0);
    let repeated_trainings = langmeta
        .as_ref()
        .map(|lm| {
            lm.texts
                .values()
                .filter(|t| t.trained_count > 1)
                .map(|t| (t.trained_count - 1) as u64)
                .sum::<u64>()
        })
        .unwrap_or(0);

    // ── Transformer stats ───────────────────────────────────────────
    let (
        tf_enabled,
        tf_embed_dim,
        tf_hidden_dim,
        tf_vocab_size,
        tf_output_trained,
        tf_ffn_trained,
        tf_attention_trained,
        tf_attention_projection_o_trained,
        tf_attention_projection_v_trained,
        tf_attention_projection_q_trained,
        tf_attention_projection_k_trained,
    ) = match TransformerLanguageModel::load_from_file(&tf_path) {
        Ok(model) => (
            true,
            Some(model.embed_dim),
            Some(model.hidden_dim),
            Some(model.vocab_order.len()),
            model.output_w.iter().any(|&v| v != 0.0) || model.output_b.iter().any(|&v| v != 0.0),
            model.ffn_trained,
            model.attention_trained,
            model.attention_projection_o_trained(),
            model.attention_projection_v_trained(),
            model.attention_projection_q_trained(),
            model.attention_projection_k_trained(),
        ),
        Err(_) => (
            false, None, None, None, false, false, false, false, false, false, false,
        ),
    };

    let attn_params = tf_embed_dim.map(|d| (4 * d * d) as u64).unwrap_or(0);
    let ffn_params = tf_embed_dim
        .zip(tf_hidden_dim)
        .map_or(0, |(d, h)| (2 * d * h + h + d) as u64);
    let output_head_params = tf_embed_dim
        .zip(tf_vocab_size)
        .map(|(d, vs)| (d * vs + vs) as u64)
        .unwrap_or(0);
    let transformer_params = attn_params + ffn_params + output_head_params;

    // ── Print output ────────────────────────────────────────────────
    let sep = "━".repeat(37);
    let sub = "─".repeat(37);

    println!("{}", sep);
    println!(" Manas Brain — {}", stats.file_path);
    println!("{}", sep);

    // Core Network
    println!("\nCore Network");
    println!("{}", sub);
    println!("  Core network layers : {}", net_layers);
    println!("  Core neurons        : {}", net_neurons);
    println!("  Core network params : {}", net_params);
    if verbose {
        println!("  Growth mode         : width-growth");
        println!(
            "  Layer growth        : {}",
            if net_layers > 0 {
                "disabled"
            } else {
                "enabled"
            }
        );
    }

    // Language System
    println!("\nLanguage System");
    println!("{}", sub);
    println!("  Vocab size          : {}", stats.vocab_size);
    println!(
        "  Embedding dim       : {}",
        embed_params / stats.vocab_size.max(1) as u64
    );
    println!("  Embedding params    : {}", embed_params);
    println!(
        "  Sequence memory     : {}",
        if seq_bytes.is_some() {
            "enabled"
        } else {
            "missing"
        }
    );
    match seq_entries {
        Some(n) => println!("  Sequence entries    : {}", n),
        None => println!("  Sequence entries    : N/A"),
    }
    println!("  Training runs       : {}", stats.total_texts_learned);
    if verbose {
        println!("  Metadata runs       : {}", total_training_runs);
    }
    println!("  Unique texts        : {}", unique_texts);
    println!("  Repeated trainings  : {}", repeated_trainings);

    // Transformer
    println!("\nTransformer");
    println!("{}", sub);
    if tf_enabled {
        println!("  Enabled             : yes");
        println!("  Blocks              : 1");
        println!("  Attention heads     : 1");
        if let Some(d) = tf_embed_dim {
            println!("  Embed dim           : {}", d);
        }
        if let Some(h) = tf_hidden_dim {
            println!("  FFN hidden dim      : {}", h);
        }
        println!(
            "  Output head trained : {}",
            if tf_output_trained { "yes" } else { "no" }
        );
        println!(
            "  FFN trained         : {}",
            if tf_ffn_trained { "yes" } else { "no" }
        );
        println!(
            "  Attention trained     : {}",
            format_inspect_attention_status(tf_attention_trained)
        );
        println!(
            "  Attention projections : {}",
            format_attention_projections(
                tf_attention_projection_o_trained,
                tf_attention_projection_v_trained,
                tf_attention_projection_q_trained,
                tf_attention_projection_k_trained,
            )
        );
        println!("  Attention params    : {}", attn_params);
        println!("  FFN params          : {}", ffn_params);
        println!("  Output head params  : {}", output_head_params);
        println!("  Transformer params  : {}", transformer_params);
    } else {
        println!("  Enabled             : no");
    }

    // Storage
    println!("\nStorage");
    println!("{}", sub);
    let brain_sz = stats.brain_size;
    println!(
        "  Brain file          : {}  ({})",
        brain_sz,
        format_file_size(brain_sz)
    );
    match seq_bytes {
        Some(sz) => println!("  Sequence file       : {}  ({})", sz, format_file_size(sz)),
        None => println!("  Sequence file       : missing"),
    }
    match tf_bytes {
        Some(sz) => println!("  Transformer file    : {}  ({})", sz, format_file_size(sz)),
        None => println!("  Transformer file    : missing"),
    }
    match langmeta_bytes {
        Some(sz) => println!("  Language metadata   : {}  ({})", sz, format_file_size(sz)),
        None => println!("  Language metadata   : missing"),
    }
    let total_storage =
        brain_sz + seq_bytes.unwrap_or(0) + tf_bytes.unwrap_or(0) + langmeta_bytes.unwrap_or(0);
    println!(
        "  Total storage       : {}  ({})",
        total_storage,
        format_file_size(total_storage)
    );

    // Total
    println!("\nTotal");
    println!("{}", sub);
    let total_params = net_params + embed_params + transformer_params;
    println!("  Total params        : {}", total_params);
    println!(
        "  Last updated        : {}",
        format_duration(stats.last_modified)
    );
    println!("{}", sep);

    Ok(())
}

/// Get file size in bytes, or `None` if the file doesn't exist.
fn file_size(path: &Path) -> Option<u64> {
    std::fs::metadata(path).ok().map(|m| m.len())
}

/// Format bytes into a human-readable string.
fn format_file_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;
    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

fn format_inspect_attention_status(attention_trained: bool) -> &'static str {
    if attention_trained { "partial" } else { "no" }
}

fn format_training_attention_status(
    attention_frozen: bool,
    attention_projection_o_trained: bool,
    attention_projection_v_trained: bool,
    attention_projection_q_trained: bool,
    attention_projection_k_trained: bool,
) -> &'static str {
    if attention_frozen {
        "frozen"
    } else if attention_projection_o_trained
        || attention_projection_v_trained
        || attention_projection_q_trained
        || attention_projection_k_trained
    {
        "partially trained"
    } else {
        "trainable"
    }
}

fn format_attention_projections(
    attention_projection_o_trained: bool,
    attention_projection_v_trained: bool,
    attention_projection_q_trained: bool,
    attention_projection_k_trained: bool,
) -> String {
    let mut parts = Vec::new();
    if attention_projection_o_trained {
        parts.push("o");
    }
    if attention_projection_v_trained {
        parts.push("v");
    }
    if attention_projection_q_trained {
        parts.push("q");
    }
    if attention_projection_k_trained {
        parts.push("k");
    }

    if parts.is_empty() {
        "none".to_string()
    } else {
        parts.join(",")
    }
}

fn format_transformer_train_report(tf_report: &manas_language::TransformerTrainReport) -> String {
    format!(
        "Transformer training\n\
         \x20 epochs                           : {}\n\
         \x20 examples                         : {}\n\
         \x20 language lr                      : {:.4}\n\
         \x20 transformer lr                   : {:.4}\n\
         \x20 avg train loss                   : {:.4}\n\
         \x20 first epoch loss                 : {}\n\
         \x20 final epoch loss                 : {}\n\
         \x20 improvement                      : {}\n\
         \x20 pure transformer top-1 accuracy  : {:.2}%\n\
         \x20 pure transformer top-3 accuracy  : {:.2}%\n\
         \x20 output head                      : {}\n\
         \x20 feed-forward                     : {}\n\
         \x20 attention                        : {}\n\
         \x20 attention projections            : {}\n\
         \n\
         Training safety\n\
         \x20 max grad norm before clipping    : {:.4}\n\
         \x20 avg grad norm                    : {:.4}\n\
         \x20 clipped updates                  : {}\n\
         \x20 invalid updates                  : {}\n\
         \x20 unstable updates                 : {}\n\
         \x20 rolled back                      : {}\n\
         \n\
         Attention safety\n\
         \x20 projections trained              : {}\n\
         \x20 attention update attempts        : {}\n\
         \x20 attention updates applied        : {}\n\
         \x20 attention clipped updates        : {}\n\
         \x20 attention invalid updates        : {}\n\
         \x20 max attention grad norm          : {:.4}\n\
         \x20 avg attention grad norm          : {:.4}",
        tf_report.epochs,
        tf_report.examples,
        tf_report.language_lr,
        tf_report.transformer_lr,
        tf_report.avg_loss,
        tf_report
            .first_loss
            .map_or("N/A".to_string(), |v| format!("{:.4}", v)),
        tf_report
            .final_loss
            .map_or("N/A".to_string(), |v| format!("{:.4}", v)),
        tf_report
            .improvement_pct
            .map_or("N/A".to_string(), |v| format!("{:.2}%", v)),
        tf_report.top1_accuracy,
        tf_report.top3_accuracy,
        if tf_report.output_head_trained {
            "trained"
        } else {
            "untrained"
        },
        if tf_report.ffn_trained {
            "trained"
        } else {
            "untrained"
        },
        format_training_attention_status(
            tf_report.attention_frozen,
            tf_report.attention_projection_o_trained,
            tf_report.attention_projection_v_trained,
            tf_report.attention_projection_q_trained,
            tf_report.attention_projection_k_trained,
        ),
        format_attention_projections(
            tf_report.attention_projection_o_trained,
            tf_report.attention_projection_v_trained,
            tf_report.attention_projection_q_trained,
            tf_report.attention_projection_k_trained,
        ),
        tf_report.max_gradient_norm_seen,
        tf_report.avg_gradient_norm,
        tf_report.clipped_updates,
        tf_report.invalid_updates,
        tf_report.unstable_updates,
        if tf_report.rolled_back { "yes" } else { "no" },
        format_attention_projections(
            tf_report.attention_projection_o_trained,
            tf_report.attention_projection_v_trained,
            tf_report.attention_projection_q_trained,
            tf_report.attention_projection_k_trained,
        ),
        tf_report.attention_update_attempts,
        tf_report.attention_updates_applied,
        tf_report.attention_clipped_updates,
        tf_report.attention_invalid_updates,
        tf_report.max_attention_grad_norm,
        tf_report.avg_attention_grad_norm,
    )
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

    let result = decode(&trainer.embedder, &trainer.tokenizer, text);
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

// ─── Language commands ─────────────────────────────────────────────────────────

/// `manas train-language "text" [--max-context 5] [--epochs 10] [--learning-rate 0.05] [--train-transformer] [--max-new-neurons 10] [--no-grow]`
#[allow(clippy::too_many_arguments)]
fn cmd_train_language(
    text: &str,
    max_context: usize,
    epochs: usize,
    learning_rate: f32,
    train_transformer: bool,
    transformer_learning_rate: f32,
    max_new_neurons: usize,
    no_grow: bool,
    transformer_max_grad_norm: f32,
    transformer_max_loss: f32,
    no_transformer_rollback: bool,
    brain_path: &Path,
) -> Result<(), ManasError> {
    let brain = ManasBrain::new(brain_path);
    let mut network = load_or_create_network(&brain);
    let mut trainer = Trainer::new();
    restore_trainer_from_brain(&mut trainer, &brain);

    trainer.source = Source::RawText;
    trainer.freshness_category = detect_freshness_category(text);

    // ── Language metadata for growth control ──────────────────────
    let langmeta_path = language_meta_path(brain_path);
    let mut langmeta = if langmeta_path.exists() {
        LanguageMeta::load_from_file(&langmeta_path)?
    } else {
        LanguageMeta::new()
    };

    let hash = text_hash(text);
    let is_known = langmeta.is_known(hash);

    // Determine effective max_new_neurons:
    //   --no-grow          ⇒ 0
    //   known text         ⇒ 0 (disable growth for repeats)
    //   otherwise          ⇒ max_new_neurons from CLI
    let effective_max = if no_grow || is_known {
        0
    } else {
        max_new_neurons
    };

    // Load existing sequence memory or create fresh
    let seq_path = seq_memory_path(brain_path);
    let mut seq_memory = if seq_path.exists() {
        SequenceMemory::load_from_file(&seq_path)?
    } else {
        SequenceMemory::new()
    };

    let report = train_next_token_examples(
        &mut network,
        &mut trainer,
        &mut seq_memory,
        text,
        max_context,
        epochs,
        learning_rate,
        effective_max,
    )?;

    // Update language metadata
    langmeta.record(hash, max_context, report.examples_count);
    langmeta.save_to_file(&langmeta_path)?;

    network.total_texts_learned += 1;
    save_brain(&brain, &network, &trainer)?;

    // Save sequence memory alongside the brain
    seq_memory.save_to_file(&seq_path)?;

    // ── Optional transformer output-head training (v0.7) ──────────
    if train_transformer {
        let embed_dim = trainer.embedder.dim;
        let hidden_dim = (embed_dim * 2).max(8);

        let transformer_path = transformer_model_path(brain_path);
        let mut model = if transformer_path.exists() {
            TransformerLanguageModel::load_from_file(&transformer_path)?
        } else {
            let mut vocab_order: Vec<u32> = trainer.embedder.table.keys().copied().collect();
            vocab_order.sort();
            TransformerLanguageModel::new(embed_dim, hidden_dim, vocab_order)
        };

        let tokens = trainer.tokenizer.encode(text);
        let examples = build_sequence_examples(&tokens, max_context);

        let tf_epochs = epochs.max(10);
        let safety = TransformerTrainingSafety {
            max_gradient_norm: transformer_max_grad_norm,
            max_loss: transformer_max_loss,
            rollback_on_unstable: !no_transformer_rollback,
            ..TransformerTrainingSafety::default()
        };
        let tf_report = train_transformer_output_head_with_safety(
            &mut model,
            &trainer.embedder,
            &examples,
            max_context,
            tf_epochs,
            transformer_learning_rate,
            learning_rate,
            &safety,
        );

        // Only save if model is finite (not corrupted)
        if manas_language::is_finite_model(&model) {
            model.save_to_file(&transformer_path)?;
        } else {
            println!("Warning: transformer model corrupted — not saving");
        }

        println!("{}", format_transformer_train_report(&tf_report));
    }

    println!(
        "Trained {} epochs on {} examples | avg loss: {:.4} | tokens: {}",
        epochs, report.examples_count, report.average_loss, report.tokens_learned
    );
    Ok(())
}

/// `manas predict-next "context" [--max-context 5] [--top-k 10] [--use-transformer]`
fn cmd_predict_next(
    text: &str,
    max_context: usize,
    top_k: usize,
    use_transformer: bool,
    transformer_only: bool,
    brain_path: &Path,
) -> Result<(), ManasError> {
    let brain = ManasBrain::new(brain_path);
    if !brain.path.exists() {
        println!("No brain file found at {}", brain.path.display());
        return Ok(());
    }

    let network = brain.load()?;
    let mut trainer = Trainer::new();
    restore_trainer_from_brain(&mut trainer, &brain);

    let mut tok = trainer.tokenizer.clone();
    let tokens = tok.encode(text);
    for &id in &tokens {
        trainer.embedder.embed_or_init(id);
    }

    // Load sequence memory for hybrid prediction
    let seq_path = seq_memory_path(brain_path);
    let seq_memory = if seq_path.exists() {
        SequenceMemory::load_from_file(&seq_path)?
    } else {
        SequenceMemory::new()
    };

    let results: Vec<(u32, f32)> = if use_transformer {
        let transformer_path = transformer_model_path(brain_path);
        let transformer_predictor = if transformer_path.exists() {
            let model = TransformerLanguageModel::load_from_file(&transformer_path)?;
            TransformerPredictor::from_model(&model, max_context)
        } else {
            let embed_dim = trainer.embedder.dim;
            let hidden_dim = (embed_dim * 2).max(8);
            TransformerPredictor::new(embed_dim, hidden_dim, max_context)
        };
        if transformer_only {
            transformer_predictor.predict_top_k_transformer(&trainer.embedder, &tokens, top_k)
        } else {
            transformer_predictor.predict_top_k_assisted(
                &network,
                &trainer.embedder,
                &seq_memory,
                &tokens,
                top_k,
            )
        }
    } else {
        let predictor = NextTokenPredictor::new(max_context);
        predictor.predict_top_k_with_memory(
            &network,
            &trainer.embedder,
            &seq_memory,
            &tokens,
            top_k,
        )
    };

    if results.is_empty() {
        println!("No predictions available");
        return Ok(());
    }

    println!("Top predictions:");
    for (id, score) in &results {
        let word = trainer.tokenizer.decode(*id).unwrap_or("?");
        println!("  {:<20} score={:.4}", word, score);
    }
    Ok(())
}

/// `manas generate "prompt" [--max-tokens 20] [--max-context 5] [--top-k 1] [--temperature 1.0] [--use-transformer]`
fn cmd_generate(
    prompt: &str,
    max_tokens: usize,
    max_context: usize,
    top_k: usize,
    temperature: f32,
    use_transformer: bool,
    brain_path: &Path,
) -> Result<(), ManasError> {
    let brain = ManasBrain::new(brain_path);
    if !brain.path.exists() {
        println!("No brain file found at {}", brain.path.display());
        return Ok(());
    }

    let network = brain.load()?;
    let mut trainer = Trainer::new();
    restore_trainer_from_brain(&mut trainer, &brain);

    let seq_path = seq_memory_path(brain_path);
    let seq_memory = if seq_path.exists() {
        SequenceMemory::load_from_file(&seq_path)?
    } else {
        SequenceMemory::new()
    };

    let mut tok = trainer.tokenizer.clone();
    let _tokens = tok.encode(prompt);
    for &id in &_tokens {
        trainer.embedder.embed_or_init(id);
    }

    let text = if use_transformer {
        let transformer_path = transformer_model_path(brain_path);
        let transformer_predictor = if transformer_path.exists() {
            let model = TransformerLanguageModel::load_from_file(&transformer_path)?;
            TransformerPredictor::from_model(&model, max_context)
        } else {
            let embed_dim = trainer.embedder.dim;
            let hidden_dim = (embed_dim * 2).max(8);
            TransformerPredictor::new(embed_dim, hidden_dim, max_context)
        };
        generate_text_with_transformer(
            &network,
            &trainer.embedder,
            &trainer.tokenizer,
            &seq_memory,
            &transformer_predictor,
            prompt,
            max_tokens,
            top_k,
        )
    } else {
        generate_text_with_memory(
            &network,
            &trainer.embedder,
            &trainer.tokenizer,
            &seq_memory,
            prompt,
            max_tokens,
            max_context,
            top_k,
            temperature,
        )
    };

    if text.is_empty() {
        println!("No output could be generated for the given prompt.");
    } else {
        println!("Generated:\n{}", text);
    }
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

// ─── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn temp_test_dir(name: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("manas_cli_{}_{}", name, nanos));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn default_teach_options() -> TeachOptions {
        TeachOptions {
            max_context: 5,
            epochs: 2,
            learning_rate: 0.05,
            train_transformer: false,
            transformer_learning_rate: 0.01,
            transformer_max_grad_norm: 5.0,
            transformer_max_loss: 50.0,
            no_transformer_rollback: false,
            dry_run: false,
        }
    }

    fn default_ask_options() -> AskOptions {
        AskOptions::default()
    }

    fn teach_identity_file(dir: &Path, brain: &Path) -> PathBuf {
        let teach_dir = dir.join("teach");
        std::fs::create_dir_all(&teach_dir).unwrap();
        let file = teach_dir.join("identity.md");
        std::fs::write(
            &file,
            "Manas is a local-first AI memory system written in Rust.\n\
             Manas learns from text and files.\n\
             Manas stores persistent memory in a .manas brain file.\n\
             Manas uses custom transformer training.\n\
             Manas is not a ChatGPT clone.\n",
        )
        .unwrap();
        cmd_teach(file.to_str().unwrap(), default_teach_options(), brain).unwrap();
        file
    }

    #[test]
    fn format_file_size_bytes() {
        assert_eq!(format_file_size(0), "0 B");
        assert_eq!(format_file_size(1), "1 B");
        assert_eq!(format_file_size(1023), "1023 B");
    }

    #[test]
    fn format_file_size_kb() {
        let s = format_file_size(1024);
        assert!(s.contains("1.00"), "expected 1.00 KB, got {}", s);
        assert!(s.contains("KB"), "expected KB, got {}", s);

        let s = format_file_size(1536);
        assert!(s.contains("1.50"), "expected 1.50 KB, got {}", s);
    }

    #[test]
    fn format_file_size_mb() {
        let s = format_file_size(1048576);
        assert!(s.contains("1.00"), "expected 1.00 MB, got {}", s);
        assert!(s.contains("MB"), "expected MB, got {}", s);
    }

    #[test]
    fn inspect_attention_formatting_shows_partial_o() {
        assert_eq!(format_inspect_attention_status(false), "no");
        assert_eq!(
            format_attention_projections(false, false, false, false),
            "none"
        );
        assert_eq!(format_inspect_attention_status(true), "partial");
        assert_eq!(format_attention_projections(true, false, false, false), "o");
        assert_eq!(
            format_attention_projections(true, true, false, false),
            "o,v"
        );
        assert_eq!(
            format_attention_projections(true, true, true, true),
            "o,v,q,k"
        );
    }

    #[test]
    fn training_attention_formatting_shows_partial_o() {
        assert_eq!(
            format_training_attention_status(true, false, false, false, false),
            "frozen"
        );
        assert_eq!(
            format_training_attention_status(false, false, false, false, false),
            "trainable"
        );
        assert_eq!(
            format_training_attention_status(false, true, true, true, true),
            "partially trained"
        );
        assert_eq!(
            format_attention_projections(true, true, true, true),
            "o,v,q,k"
        );
    }

    #[test]
    fn file_size_existing_file() {
        let mut tmp = std::env::temp_dir();
        tmp.push("manas_test_inspect_file_size");
        let mut f = std::fs::File::create(&tmp).unwrap();
        f.write_all(b"hello").unwrap();
        drop(f);

        let sz = file_size(&tmp);
        assert_eq!(sz, Some(5));

        std::fs::remove_file(&tmp).unwrap();
    }

    #[test]
    fn file_size_missing_file() {
        let p = Path::new("/tmp/manas_test_nonexistent_xyz123");
        assert_eq!(file_size(p), None);
    }

    #[test]
    fn file_size_zero_length() {
        let mut tmp = std::env::temp_dir();
        tmp.push("manas_test_zero_file");
        std::fs::File::create(&tmp).unwrap();

        let sz = file_size(&tmp);
        assert_eq!(sz, Some(0));

        std::fs::remove_file(&tmp).unwrap();
    }

    #[test]
    fn teach_collects_direct_text_input() {
        let discovery = collect_teach_items("Manas is written in Rust").unwrap();
        assert_eq!(discovery.mode, TeachInputMode::Text);
        assert_eq!(discovery.files_discovered, 0);
        assert_eq!(discovery.files_skipped, 0);
        assert_eq!(discovery.items.len(), 1);
        assert!(matches!(discovery.items[0].source, Source::RawText));
    }

    #[test]
    fn teach_missing_path_like_input_errors() {
        let result = collect_teach_items("/tmp/manas_missing_teach_file.md");
        assert!(matches!(result, Err(ManasError::FileNotFound(_))));
    }

    #[test]
    fn teach_direct_text_with_period_is_not_treated_as_missing_path() {
        let discovery = collect_teach_items("Manas v0.9.6 teaches local files.").unwrap();
        assert_eq!(discovery.mode, TeachInputMode::Text);
        assert_eq!(discovery.items.len(), 1);
    }

    #[test]
    fn teach_file_support_is_md_and_txt_only() {
        assert!(teach_supported_file(Path::new("notes.md")));
        assert!(teach_supported_file(Path::new("notes.TXT")));
        assert!(!teach_supported_file(Path::new("notes.rs")));
        assert!(!teach_supported_file(Path::new("notes.html")));
    }

    #[test]
    fn teach_folder_recurses_and_skips_unsupported_and_empty_files() {
        let dir = temp_test_dir("folder_collect");
        let nested = dir.join("nested");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(dir.join("identity.md"), "Manas is local.").unwrap();
        std::fs::write(nested.join("goals.txt"), "Manas teaches folders.").unwrap();
        std::fs::write(dir.join("ignore.rs"), "fn main() {}").unwrap();
        std::fs::write(dir.join("empty.md"), "").unwrap();

        let discovery = collect_teach_items(dir.to_str().unwrap()).unwrap();
        assert_eq!(discovery.mode, TeachInputMode::Folder);
        assert_eq!(discovery.files_discovered, 4);
        assert_eq!(discovery.items.len(), 2);
        assert_eq!(discovery.files_skipped, 2);

        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn teach_dry_run_does_not_create_brain_sidecars() {
        let dir = temp_test_dir("dry_run");
        let brain = dir.join("brain.manas");
        let mut options = default_teach_options();
        options.train_transformer = true;
        options.dry_run = true;

        cmd_teach("Manas dry run text", options, &brain).unwrap();

        assert!(!brain.exists());
        assert!(!seq_memory_path(&brain).exists());
        assert!(!transformer_model_path(&brain).exists());
        assert!(!language_meta_path(&brain).exists());

        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn teach_direct_text_learns_core_and_sequence_memory() {
        let dir = temp_test_dir("direct_text");
        let brain = dir.join("brain.manas");

        cmd_teach(
            "Manas is a local first memory system",
            default_teach_options(),
            &brain,
        )
        .unwrap();

        assert!(brain.exists());
        assert!(seq_memory_path(&brain).exists());
        assert!(language_meta_path(&brain).exists());

        let stats = ManasBrain::new(brain.clone()).inspect().unwrap();
        assert_eq!(stats.total_texts_learned, 1);
        assert!(stats.vocab_size > 0);

        let seq_memory = SequenceMemory::load_from_file(&seq_memory_path(&brain)).unwrap();
        assert!(!seq_memory.transitions.is_empty());

        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn teach_file_preserves_source_metadata() {
        let dir = temp_test_dir("file_source");
        let brain = dir.join("brain.manas");
        let file = dir.join("identity.md");
        std::fs::write(&file, "Manas is a source aware memory system.").unwrap();

        cmd_teach(file.to_str().unwrap(), default_teach_options(), &brain).unwrap();

        let network = ManasBrain::new(brain.clone()).load().unwrap();
        let file_path = file.display().to_string();
        assert!(
            network
                .all_neurons()
                .into_iter()
                .any(|(_, neuron)| matches!(&neuron.source, Source::LocalFile { path } if path == &file_path)),
            "expected at least one neuron to preserve the file source path"
        );

        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn teach_txt_file_trains_sequence_memory() {
        let dir = temp_test_dir("txt_file");
        let brain = dir.join("brain.manas");
        let file = dir.join("goals.txt");
        std::fs::write(&file, "Manas focuses on local learning and memory.").unwrap();

        cmd_teach(file.to_str().unwrap(), default_teach_options(), &brain).unwrap();

        let seq_memory = SequenceMemory::load_from_file(&seq_memory_path(&brain)).unwrap();
        assert!(!seq_memory.transitions.is_empty());

        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn teach_folder_teaches_multiple_supported_files() {
        let dir = temp_test_dir("folder_teach");
        let brain = dir.join("brain.manas");
        let teach_dir = dir.join("teach");
        let nested = teach_dir.join("nested");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(
            teach_dir.join("identity.md"),
            "Manas learns from markdown files.",
        )
        .unwrap();
        std::fs::write(nested.join("goals.txt"), "Manas learns from text files.").unwrap();
        std::fs::write(teach_dir.join("skip.rs"), "let ignored = true;").unwrap();

        cmd_teach(teach_dir.to_str().unwrap(), default_teach_options(), &brain).unwrap();

        let stats = ManasBrain::new(brain.clone()).inspect().unwrap();
        assert_eq!(stats.total_texts_learned, 2);

        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn teach_with_transformer_creates_trained_transformer_sidecar() {
        let dir = temp_test_dir("transformer");
        let brain = dir.join("brain.manas");
        let mut options = default_teach_options();
        options.train_transformer = true;
        options.transformer_learning_rate = 0.05;

        cmd_teach(
            "Rust is a systems programming language focused on safety and performance",
            options,
            &brain,
        )
        .unwrap();

        let transformer_path = transformer_model_path(&brain);
        assert!(transformer_path.exists());
        let model = TransformerLanguageModel::load_from_file(&transformer_path).unwrap();
        assert!(model.ffn_trained);
        assert!(model.attention_trained);
        assert!(model.attention_projection_o_trained());
        assert!(model.attention_projection_v_trained());
        assert!(model.attention_projection_q_trained());
        assert!(model.attention_projection_k_trained());

        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn prediction_works_after_teaching_file() {
        let dir = temp_test_dir("predict_after_teach");
        let brain = dir.join("brain.manas");
        let file = dir.join("identity.md");
        std::fs::write(&file, "Manas is a local first AI memory system.").unwrap();

        cmd_teach(file.to_str().unwrap(), default_teach_options(), &brain).unwrap();

        let network = ManasBrain::new(brain.clone()).load().unwrap();
        let mut trainer = Trainer::new();
        restore_trainer_from_brain(&mut trainer, &ManasBrain::new(brain.clone()));
        let seq_memory = SequenceMemory::load_from_file(&seq_memory_path(&brain)).unwrap();
        let tokens = trainer.tokenizer.encode("Manas is");
        let predictor = NextTokenPredictor::new(5);
        let predictions = predictor.predict_top_k_with_memory(
            &network,
            &trainer.embedder,
            &seq_memory,
            &tokens,
            3,
        );

        assert!(!predictions.is_empty());
        let top = trainer.tokenizer.decode(predictions[0].0).unwrap();
        assert_eq!(top, "a");

        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn generation_works_after_teaching_file() {
        let dir = temp_test_dir("generate_after_teach");
        let brain = dir.join("brain.manas");
        let file = dir.join("identity.md");
        std::fs::write(&file, "Manas is a local first AI memory system.").unwrap();

        cmd_teach(file.to_str().unwrap(), default_teach_options(), &brain).unwrap();

        let network = ManasBrain::new(brain.clone()).load().unwrap();
        let mut trainer = Trainer::new();
        restore_trainer_from_brain(&mut trainer, &ManasBrain::new(brain.clone()));
        let seq_memory = SequenceMemory::load_from_file(&seq_memory_path(&brain)).unwrap();
        let generated = generate_text_with_memory(
            &network,
            &trainer.embedder,
            &trainer.tokenizer,
            &seq_memory,
            "Manas is",
            5,
            5,
            1,
            1.0,
        );

        assert!(
            generated.starts_with("manas is a"),
            "unexpected generated text: {}",
            generated
        );

        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn ask_answers_from_taught_md_file() {
        let dir = temp_test_dir("ask_md");
        let brain = dir.join("brain.manas");
        let file = teach_identity_file(&dir, &brain);

        let report =
            answer_local_question("What is Manas?", default_ask_options(), &brain).unwrap();

        assert_eq!(report.kind, LocalAnswerKind::Answer);
        assert_eq!(
            report.answer.as_deref(),
            Some("Manas is a local-first AI memory system written in Rust.")
        );
        assert_eq!(report.sources, vec![file.display().to_string()]);

        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn ask_answer_includes_source_path() {
        let dir = temp_test_dir("ask_source");
        let brain = dir.join("brain.manas");
        let file = teach_identity_file(&dir, &brain);

        let report =
            answer_local_question("What is Manas?", default_ask_options(), &brain).unwrap();
        let formatted = format_local_answer_report(&report, true);

        assert!(formatted.contains("Sources"));
        assert!(formatted.contains(&file.display().to_string()));

        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn ask_not_enough_memory_without_evidence() {
        let dir = temp_test_dir("ask_no_evidence");
        let brain = dir.join("brain.manas");
        teach_identity_file(&dir, &brain);

        let report =
            answer_local_question("What is Kubernetes?", default_ask_options(), &brain).unwrap();
        let formatted = format_local_answer_report(&report, true);

        assert_eq!(report.kind, LocalAnswerKind::NoEvidence);
        assert_eq!(formatted, "Not enough local memory to answer this yet.");

        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn ask_empty_brain_no_panic() {
        let dir = temp_test_dir("ask_empty_brain");
        let brain = dir.join("brain.manas");
        ManasBrain::new(brain.clone())
            .save(&Network::new())
            .unwrap();

        let report =
            answer_local_question("What is Manas?", default_ask_options(), &brain).unwrap();

        assert_eq!(report.kind, LocalAnswerKind::NoEvidence);

        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn ask_missing_sidecars_no_panic() {
        let dir = temp_test_dir("ask_missing_sidecars");
        let brain = dir.join("brain.manas");
        teach_identity_file(&dir, &brain);
        let _ = std::fs::remove_file(seq_memory_path(&brain));
        let _ = std::fs::remove_file(transformer_model_path(&brain));
        let _ = std::fs::remove_file(language_meta_path(&brain));

        let report =
            answer_local_question("What is Manas?", default_ask_options(), &brain).unwrap();

        assert_eq!(report.kind, LocalAnswerKind::Answer);
        assert_eq!(
            report.answer.as_deref(),
            Some("Manas is a local-first AI memory system written in Rust.")
        );

        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn ask_without_transformer_uses_extractive_memory() {
        let dir = temp_test_dir("ask_no_transformer");
        let brain = dir.join("brain.manas");
        teach_identity_file(&dir, &brain);

        assert!(!transformer_model_path(&brain).exists());
        let report =
            answer_local_question("Is Manas a ChatGPT clone?", default_ask_options(), &brain)
                .unwrap();

        assert_eq!(report.kind, LocalAnswerKind::Answer);
        assert_eq!(
            report.answer.as_deref(),
            Some("Manas is not a ChatGPT clone.")
        );

        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn ask_unrelated_question_does_not_use_wrong_memory() {
        let dir = temp_test_dir("ask_unrelated");
        let brain = dir.join("brain.manas");
        teach_identity_file(&dir, &brain);

        let report =
            answer_local_question("What is Kubernetes?", default_ask_options(), &brain).unwrap();

        assert_eq!(report.kind, LocalAnswerKind::NoEvidence);
        assert!(report.answer.is_none());
        assert!(report.sources.is_empty());

        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn ask_answer_remains_short() {
        let dir = temp_test_dir("ask_short");
        let brain = dir.join("brain.manas");
        let file = dir.join("identity.md");
        std::fs::write(
            &file,
            "Manas is a local-first AI memory system written in Rust with custom local learning, source-aware memory, sequence training, and transformer-assisted prediction.",
        )
        .unwrap();
        cmd_teach(file.to_str().unwrap(), default_teach_options(), &brain).unwrap();

        let mut options = default_ask_options();
        options.max_answer_tokens = 5;
        let report = answer_local_question("What is Manas?", options, &brain).unwrap();

        assert_eq!(report.kind, LocalAnswerKind::Answer);
        assert!(report.answer.as_ref().unwrap().split_whitespace().count() <= 5);

        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn query_without_answer_keeps_existing_mode() {
        let cli = Cli::try_parse_from(["manas", "query", "local-first"]).unwrap();
        match cli.command {
            Commands::Query { answer, .. } => assert!(!answer),
            _ => panic!("expected query command"),
        }
    }

    #[test]
    fn query_answer_uses_local_answer_path() {
        let cli = Cli::try_parse_from(["manas", "query", "What is Manas?", "--answer"]).unwrap();
        match cli.command {
            Commands::Query { text, answer } => {
                assert!(answer);
                assert_eq!(text, "What is Manas?");
            }
            _ => panic!("expected query command"),
        }
    }

    #[test]
    fn teach_then_ask_works_together() {
        let dir = temp_test_dir("teach_ask");
        let brain = dir.join("brain.manas");
        teach_identity_file(&dir, &brain);

        let report =
            answer_local_question("What is Manas?", default_ask_options(), &brain).unwrap();

        assert_eq!(report.kind, LocalAnswerKind::Answer);

        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn ask_local_answer_path_does_not_need_agent_pipeline() {
        let dir = temp_test_dir("ask_local_only");
        let brain = dir.join("brain.manas");
        teach_identity_file(&dir, &brain);

        let mut options = default_ask_options();
        options.use_transformer = true;
        let report = answer_local_question("What is Manas?", options, &brain).unwrap();

        assert_eq!(report.kind, LocalAnswerKind::Answer);

        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn source_metadata_preserved_for_answer_sources() {
        let dir = temp_test_dir("ask_source_metadata");
        let brain = dir.join("brain.manas");
        let file = teach_identity_file(&dir, &brain);

        let network = ManasBrain::new(brain.clone()).load().unwrap();
        let candidates = collect_local_source_candidates(&network);
        let report =
            answer_local_question("What is Manas?", default_ask_options(), &brain).unwrap();

        assert!(
            candidates
                .iter()
                .any(|candidate| candidate.path == file.display().to_string())
        );
        assert_eq!(report.sources, vec![file.display().to_string()]);

        std::fs::remove_dir_all(dir).unwrap();
    }
}
