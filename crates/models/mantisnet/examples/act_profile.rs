//! Ad-hoc native timing for the ACT prefix builder and collator.
//!
//! Run the direct packed hot path with `--packed`, or two simultaneous
//! 64-position calls (the fitloop prefetch shape) with `--concurrent`.
//! Add `--action-relevant` to either mode to switch the window scope.

use hexo_model_mantisnet::act_encoder::{
    self, ActBuilderConfig, CellScope, D6RelationMode, WindowScope,
};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

fn full_config() -> ActBuilderConfig {
    ActBuilderConfig {
        window_scope: WindowScope::Nonempty,
        cell_scope: CellScope::WindowAndLegal,
        d6_relation_mode: D6RelationMode::Orbit48,
        d_max: 12,
        occupied_radius: 12,
        use_cell_adjacency: true,
        use_occupied_radius_edges: true,
        use_global_numeric_features: true,
        use_window_numeric_features: true,
        use_action_tactical_features: true,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments: Vec<_> = std::env::args().collect();
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../python/mantisnet/scratch/real_games.json");
    let raw: Vec<Vec<[i16; 2]>> = serde_json::from_slice(&fs::read(path)?)?;
    let count = 64;
    let max_ply = 200;
    let mut games = Vec::with_capacity(count);
    let mut plies = Vec::with_capacity(count);
    for index in 0..count {
        let game = &raw[(17 * index) % raw.len()];
        let ply = 1 + index * (max_ply - 1) / (count - 1);
        games.push(game.iter().map(|&[q, r]| (q, r)).collect());
        plies.push(ply);
    }

    let mut cfg = full_config();
    if arguments
        .iter()
        .any(|argument| argument == "--action-relevant")
    {
        cfg.window_scope = WindowScope::ActionRelevant;
    }
    if arguments.iter().any(|argument| argument == "--concurrent") {
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(3));
        let (completed, results) = std::sync::mpsc::channel();
        std::thread::scope(|scope| -> Result<(), Box<dyn std::error::Error>> {
            let mut requests = Vec::with_capacity(2);
            for _ in 0..2 {
                let (start, requested) = std::sync::mpsc::channel();
                requests.push(start);
                let barrier = barrier.clone();
                let completed = completed.clone();
                let games = &games;
                let plies = &plies;
                let cfg = &cfg;
                scope.spawn(move || {
                    while requested.recv().is_ok() {
                        barrier.wait();
                        let batch = act_encoder::build_packed_batch_prefixes(games, plies, cfg);
                        if completed.send(batch).is_err() {
                            break;
                        }
                    }
                });
            }
            drop(completed);
            for iteration in 0..5 {
                for request in &requests {
                    request.send(())?;
                }
                let started = Instant::now();
                barrier.wait();
                let batches = (results.recv()??, results.recv()??);
                let elapsed = started.elapsed();
                println!(
                    "iteration={iteration} concurrent_ms={:.3} aggregate_pos_s={:.1}",
                    elapsed.as_secs_f64() * 1e3,
                    (2 * count) as f64 / elapsed.as_secs_f64(),
                );
                drop(batches);
            }
            drop(requests);
            Ok(())
        })?;
        return Ok(());
    }
    if arguments.iter().any(|argument| argument == "--packed") {
        for iteration in 0..5 {
            let started = Instant::now();
            let batch = act_encoder::build_packed_batch_prefixes(&games, &plies, &cfg)?;
            let elapsed = started.elapsed();
            println!(
                "iteration={iteration} packed_ms={:.3} pos_s={:.1} radius={} ",
                elapsed.as_secs_f64() * 1e3,
                count as f64 / elapsed.as_secs_f64(),
                batch.radius_src.len(),
            );
            drop(batch);
        }
        return Ok(());
    }
    for iteration in 0..5 {
        let started = Instant::now();
        let graphs = act_encoder::build_batch_prefixes(&games, &plies, &cfg)?;
        let built = started.elapsed();
        let radius_rows: usize = graphs.iter().map(|graph| graph.radius_src.len()).sum();
        let radius_max = graphs
            .iter()
            .map(|graph| graph.radius_src.len())
            .max()
            .unwrap_or(0);
        let cells: usize = graphs.iter().map(act_encoder::ActGraph::n_cells).sum();
        let windows: usize = graphs.iter().map(act_encoder::ActGraph::n_windows).sum();
        let legal: usize = graphs.iter().map(act_encoder::ActGraph::n_legal).sum();
        let started = Instant::now();
        let packed = act_encoder::collate(graphs)?;
        let collated = started.elapsed();
        println!(
            "iteration={iteration} build_ms={:.3} collate_ms={:.3} cells={cells} windows={windows} legal={legal} radius={radius_rows} radius_max={radius_max} orbit_bound={}",
            built.as_secs_f64() * 1e3,
            collated.as_secs_f64() * 1e3,
            packed.radius_orbit_bound,
        );
    }
    Ok(())
}
