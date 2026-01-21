use cs336_native::pretokenize::pretokenize_bytes_impl;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn expand_home(path_str: &str) -> PathBuf {
    if path_str.starts_with("~/")
        && let Ok(home) = std::env::var("HOME") {
            return Path::new(&home).join(&path_str[2..]);
        }
    PathBuf::from(path_str)
}

fn main() {
    let path_str = "~/datasets/cs336/owt_train.txt";
    let path = expand_home(path_str);

    println!("Reading file: {:?}", path);

    let mut file = File::open(&path).expect("Failed to open file");
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).expect("Failed to read file");

    println!("File size: {:.2} MB", buffer.len() as f64 / 1024.0 / 1024.0);

    let guard = pprof::ProfilerGuardBuilder::default()
        .frequency(1000)
        .blocklist(&["libc", "libgcc", "pthread", "vdso"])
        .build()
        .unwrap();

    let start = Instant::now();
    let counts = pretokenize_bytes_impl(&buffer, vec!["<|endoftext|>".to_string()])
        .expect("Pretokenization failed");
    let duration = start.elapsed();

    println!("Pretokenization took: {:.4?}", duration);
    println!("Unique tokens: {}", counts.len());
    println!("Total tokens: {}", counts.values().sum::<u64>());

    if let Ok(report) = guard.report().build() {
        let file = File::create("flamegraph.svg").unwrap();
        report.flamegraph(file).unwrap();
        println!("Flamegraph saved to flamegraph.svg");
    }
}
