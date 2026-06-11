use std::hint::black_box;
use std::time::{Duration, Instant};

fn main() {
    let results = run_benches();
    println!("\n{:=<60}", "");
    println!(" Manas Benchmarks");
    println!("{:=<60}", "");
    for (name, duration) in &results {
        let us = duration.as_secs_f64() * 1_000_000.0;
        println!(" {:<44} {:>8.2} µs", name, us);
    }
    println!("{:=<60}", "");
}

fn run_benches() -> Vec<(&'static str, Duration)> {
    let mut results = Vec::new();

    results.push(("tokenize (short text)", bench_tokenize_short()));
    results.push(("tokenize (long text)", bench_tokenize_long()));
    results.push(("embed average (10 tokens, dim=64)", bench_embed_average()));
    results.push((
        "forward pass (2 layers, 16+64 neurons)",
        bench_forward_small(),
    ));
    results.push(("forward pass (3 layers, 256+128+64)", bench_forward_large()));
    results.push(("backprop (2 layers, 16+64)", bench_backprop_small()));
    results.push(("learn (short text, repeat 50x)", bench_learn_short()));
    results.push(("save to .manas", bench_save()));
    results.push(("load from .manas", bench_load()));
    results.push(("importance scoring (16 neurons)", bench_importance()));
    results.push(("compress (16 neurons)", bench_compress()));

    results
}

fn bench_tokenize_short() -> Duration {
    let mut trainer = manas_learn::Trainer::new();
    let text = "rust is a systems programming language";
    let start = Instant::now();
    for _ in 0..1000 {
        let _ = black_box(trainer.tokenizer.encode(text));
    }
    start.elapsed() / 1000
}

fn bench_tokenize_long() -> Duration {
    let mut trainer = manas_learn::Trainer::new();
    let text = "The Rust programming language helps you write faster, more reliable software. \
                High-level ergonomics and low-level control are often at odds; \
                Rust challenges that conflict by balancing powerful technical capacity \
                and a great developer experience.";
    let start = Instant::now();
    for _ in 0..500 {
        let _ = black_box(trainer.tokenizer.encode(text));
    }
    start.elapsed() / 500
}

fn bench_embed_average() -> Duration {
    let mut embedder = manas_learn::Embedder::new(64);
    let ids: Vec<u32> = (0..10).collect();
    for &id in &ids {
        embedder.embed_or_init(id);
    }
    let start = Instant::now();
    for _ in 0..5000 {
        let _ = black_box(embedder.average_embed(&ids));
    }
    start.elapsed() / 5000
}

fn make_small_network() -> manas_core::Network {
    let mut net = manas_core::Network::new();
    net.grow_layer(16, 64);
    net.grow_layer(64, 16);
    net
}

fn make_large_network() -> manas_core::Network {
    let mut net = manas_core::Network::new();
    net.grow_layer(256, 64);
    net.grow_layer(128, 256);
    net.grow_layer(64, 128);
    net
}

fn bench_forward_small() -> Duration {
    let net = make_small_network();
    let input = vec![0.5; 64];
    let start = Instant::now();
    for _ in 0..5000 {
        let _ = black_box(net.forward(&input));
    }
    start.elapsed() / 5000
}

fn bench_forward_large() -> Duration {
    let net = make_large_network();
    let input = vec![0.5; 64];
    let start = Instant::now();
    for _ in 0..1000 {
        let _ = black_box(net.forward(&input));
    }
    start.elapsed() / 1000
}

fn bench_backprop_small() -> Duration {
    let net = make_small_network();
    let input = vec![0.5; 64];
    let target = vec![0.5; 64];
    let start = Instant::now();
    for _ in 0..500 {
        let _ = black_box(manas_learn::compute_gradients(&net, &input, &target));
    }
    start.elapsed() / 500
}

fn bench_learn_short() -> Duration {
    let mut trainer = manas_learn::Trainer::new();
    let mut net = manas_core::Network::new();
    let text = "rust is fast and safe";
    let start = Instant::now();
    for _ in 0..50 {
        let _ = black_box(trainer.learn(&mut net, text).unwrap());
    }
    start.elapsed() / 50
}

fn bench_save() -> Duration {
    let net = make_small_network();
    let path = std::path::Path::new("/tmp/manas_bench_save.manas");
    let brain = manas_store::ManasBrain::new(path);
    let start = Instant::now();
    for _ in 0..100 {
        brain.save(&net).unwrap();
    }
    let elapsed = start.elapsed() / 100;
    std::fs::remove_file(path).ok();
    elapsed
}

fn bench_load() -> Duration {
    let net = make_small_network();
    let path = std::path::Path::new("/tmp/manas_bench_load.manas");
    let brain = manas_store::ManasBrain::new(path);
    brain.save(&net).unwrap();
    let start = Instant::now();
    for _ in 0..100 {
        let _ = black_box(brain.load().unwrap());
    }
    let elapsed = start.elapsed() / 100;
    std::fs::remove_file(path).ok();
    elapsed
}

fn bench_importance() -> Duration {
    let net = make_small_network();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let start = Instant::now();
    for _ in 0..5000 {
        for neuron in &net.layers[0].neurons {
            let _ = black_box(manas_memory::importance_for_neuron(neuron, now));
        }
    }
    start.elapsed() / 5000
}

fn bench_compress() -> Duration {
    let mut net = make_small_network();
    let start = Instant::now();
    for _ in 0..200 {
        let _ = black_box(manas_memory::compress(&mut net, 0.1, 0.8));
    }
    start.elapsed() / 200
}
