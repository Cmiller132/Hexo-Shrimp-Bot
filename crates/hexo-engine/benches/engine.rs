//! Criterion suite for the `hexo-engine` hot paths.

mod common;

use std::hint::black_box;
use std::time::Duration;

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput};
use hexo_engine::{Action, HexCoord, Position, Search};

use common::{PLIES, game, inflated, interior_and_edge, position_at};

/// Plies of `+q` excursion behind the inflated-arena fixture.
const EXCURSION_STEPS: usize = 32;

/// The ply the inflated-arena fixture is taken from: a mid-game root, which is where a
/// search that reaches far out and unwinds actually happens.
const INFLATED_PLY: usize = 96;

/// One game stage, with the two placements the timing is split over.
struct Stage {
    /// Ply the position sits at.
    ply: usize,
    /// The position itself.
    pos: Position,
    /// Legal placement nearest the centroid of the stones.
    interior: Action,
    /// Legal placement furthest from it.
    edge: Action,
}

/// Every fixture the suite measures, built once per run.
struct Fixtures {
    /// One entry per ply in [`PLIES`].
    stages: Vec<Stage>,
    /// The longest fixture game as a move list, for `replay`.
    record: Vec<Action>,
    /// [`INFLATED_PLY`] after a search excursion grew the arena and unwound.
    inflated: Position,
}

impl Fixtures {
    fn build() -> Self {
        let stages: Vec<Stage> = PLIES
            .iter()
            .map(|&ply| {
                let pos = position_at(ply);
                let (interior, edge) = interior_and_edge(&pos);
                assert!(pos.is_legal(interior) && pos.is_legal(edge));
                Stage {
                    ply,
                    pos,
                    interior,
                    edge,
                }
            })
            .collect();
        let inflated = inflated(&position_at(INFLATED_PLY), EXCURSION_STEPS);
        Self {
            stages,
            record: game(PLIES[PLIES.len() - 1]),
            inflated,
        }
    }
}

/// `Position::advance` â€” the runner's forward path, on a placement that does not grow
/// the arena.
fn advance(c: &mut Criterion, f: &Fixtures) {
    let mut g = c.benchmark_group("advance");
    for s in &f.stages {
        g.bench_function(BenchmarkId::from_parameter(s.ply), |b| {
            b.iter_batched_ref(
                || s.pos.clone(),
                |p| black_box(p.advance(s.interior)),
                BatchSize::LargeInput,
            );
        });
    }
    g.finish();
}

/// `Search::apply` + `Search::undo` as a pair â€” the search hot path.
fn apply_undo(c: &mut Criterion, f: &Fixtures) {
    let mut g = c.benchmark_group("apply_undo");
    for s in &f.stages {
        for (label, action) in [("interior", s.interior), ("edge", s.edge)] {
            g.bench_function(BenchmarkId::new(label, s.ply), |b| {
                let mut pos = s.pos.clone();
                let mut search = Search::new(&mut pos);
                b.iter(|| {
                    let _ = black_box(search.apply(black_box(action)));
                    black_box(search.undo());
                });
            });
        }
    }
    g.finish();
}

/// `Position::clone` and drop â€” paid once per position handed to a player mirror, and
/// once per game slot reset.
fn clone_drop(c: &mut Criterion, f: &Fixtures) {
    let mut g = c.benchmark_group("clone");
    for s in &f.stages {
        g.bench_function(BenchmarkId::from_parameter(s.ply), |b| {
            b.iter(|| black_box(&s.pos).clone());
        });
    }
    g.finish();
}

/// Full enumeration of the frontier and occupancy planes.
fn enumerate(c: &mut Criterion, f: &Fixtures) {
    let mut g = c.benchmark_group("enumerate");
    for s in &f.stages {
        g.throughput(Throughput::Elements(s.pos.legal_count() as u64));
        g.bench_function(BenchmarkId::new("legal_actions", s.ply), |b| {
            b.iter(|| {
                black_box(&s.pos)
                    .legal_actions()
                    .fold(0i32, |acc, a| acc.wrapping_add(i32::from(a.coord().q)))
            });
        });
        g.throughput(Throughput::Elements(u64::from(s.pos.stone_count())));
        g.bench_function(BenchmarkId::new("stones", s.ply), |b| {
            b.iter(|| {
                black_box(&s.pos).stones().fold(0i32, |acc, (c, p)| {
                    acc.wrapping_add(i32::from(c.q) + p.index() as i32)
                })
            });
        });
    }

    let inflated = &f.inflated;
    g.throughput(Throughput::Elements(inflated.legal_count() as u64));
    g.bench_function(
        BenchmarkId::new("legal_actions_inflated", INFLATED_PLY),
        |b| {
            b.iter(|| {
                black_box(inflated)
                    .legal_actions()
                    .fold(0i32, |acc, a| acc.wrapping_add(i32::from(a.coord().q)))
            });
        },
    );
    g.throughput(Throughput::Elements(u64::from(inflated.stone_count())));
    g.bench_function(BenchmarkId::new("stones_inflated", INFLATED_PLY), |b| {
        b.iter(|| {
            black_box(inflated).stones().fold(0i32, |acc, (c, p)| {
                acc.wrapping_add(i32::from(c.q) + p.index() as i32)
            })
        });
    });
    g.finish();
}

/// The two directions of the canonical ordering against the naive walk they replaced.
fn ordering(c: &mut Criterion, f: &Fixtures) {
    let mut g = c.benchmark_group("ordering");
    let mut subjects: Vec<(String, &Position)> = f
        .stages
        .iter()
        .map(|s| (format!("ply{}", s.ply), &s.pos))
        .collect();
    subjects.push((format!("ply{INFLATED_PLY}_inflated"), &f.inflated));

    for (stage, pos) in &subjects {
        let n = pos.legal_count();
        for (label, k) in [("first", 0), ("middle", n / 2), ("last", n - 1)] {
            let action = pos.nth_legal(k).expect("a rank below legal_count");
            let id = format!("{stage}/{label}");

            g.bench_function(BenchmarkId::new("rank_prefix", &id), |b| {
                b.iter(|| black_box(*pos).legal_rank(black_box(action)));
            });
            g.bench_function(BenchmarkId::new("rank_walk", &id), |b| {
                b.iter(|| {
                    let target = black_box(action);
                    black_box(*pos).legal_actions().position(|a| a == target)
                });
            });
            g.bench_function(BenchmarkId::new("select_scan", &id), |b| {
                b.iter(|| black_box(*pos).nth_legal(black_box(k)));
            });
            g.bench_function(BenchmarkId::new("select_walk", &id), |b| {
                b.iter(|| black_box(*pos).legal_actions().nth(black_box(k)));
            });
        }
    }
    g.finish();
}

/// `Position::windows_through` â€” the 18-window gather a feature encoder reads.
fn windows(c: &mut Criterion, f: &Fixtures) {
    let mut g = c.benchmark_group("windows");
    for s in &f.stages {
        let coord = s.interior.coord();
        g.bench_function(BenchmarkId::from_parameter(s.ply), |b| {
            b.iter(|| black_box(&s.pos).windows_through(black_box(coord)));
        });
    }
    g.finish();
}

/// `Position::replay` of a whole game â€” the record-loading path.
fn replay(c: &mut Criterion, f: &Fixtures) {
    let mut g = c.benchmark_group("replay");
    g.throughput(Throughput::Elements(f.record.len() as u64));
    g.bench_function(BenchmarkId::from_parameter(f.record.len()), |b| {
        b.iter(|| Position::replay(black_box(&f.record)));
    });
    g.finish();
}

/// `Position::new()` plus the opening placement â€” a game slot reset, including the
/// first arena allocation.
fn new_game(c: &mut Criterion) {
    let mut g = c.benchmark_group("new_game");
    g.bench_function("new_plus_opening", |b| {
        b.iter(|| {
            let mut pos = Position::new();
            let _ = black_box(pos.advance(Action::new(HexCoord::ORIGIN)));
            pos
        });
    });
    g.finish();
}

/// Written out rather than assembled by `criterion_group!` + `criterion_main!`: those
/// expand to exactly these five lines, plus a `pub fn` with no doc comment that the
/// workspace `missing_docs` lint then rejects.
fn main() {
    let mut criterion = Criterion::default()
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2))
        .configure_from_args();

    let f = Fixtures::build();
    advance(&mut criterion, &f);
    apply_undo(&mut criterion, &f);
    clone_drop(&mut criterion, &f);
    enumerate(&mut criterion, &f);
    ordering(&mut criterion, &f);
    windows(&mut criterion, &f);
    replay(&mut criterion, &f);
    new_game(&mut criterion);

    criterion.final_summary();
}
