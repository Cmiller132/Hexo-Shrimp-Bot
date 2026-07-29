//! Probe-set and probe-hash contracts.
//!
//! The test encoder and evaluator isolate probe behavior from model behavior.

use hexo_engine::Position;
use hexo_model::{probe_hash, probe_positions};
use hexo_search::{EncodedBatch, Encoder, Evaluation, Evaluator};

/// Zobrist and legal count, which is everything an evaluator below needs.
struct Bytes;

impl Encoder for Bytes {
    fn encode(&self, position: &Position, out: &mut Vec<u8>) {
        out.extend_from_slice(&position.zobrist().to_le_bytes());
        let count = u32::try_from(position.legal_count()).expect("a legal count fits u32");
        out.extend_from_slice(&count.to_le_bytes());
    }
}

/// An evaluator whose complete weight state is `knob`.
struct Knob {
    knob: u64,
}

impl Knob {
    /// A well-behaved answer that depends on the knob and on the position.
    fn answer(&self, zobrist: u64, legal: usize) -> Evaluation {
        let priors = (0..legal)
            .map(|i| {
                let raw = self
                    .knob
                    .wrapping_mul(zobrist | 1)
                    .wrapping_add(i as u64)
                    .rotate_left(17);
                (raw >> 40) as f32 + 1.0
            })
            .collect::<Vec<f32>>();
        let total: f32 = priors.iter().sum();
        Evaluation {
            priors: priors.into_iter().map(|p| p / total).collect(),
            // 24 bits mapped onto `[-1, 1)`, which is the seam's convention.
            value: ((self.knob ^ zobrist) >> 40) as f32 / 8_388_608.0 - 1.0,
        }
    }
}

impl Evaluator for Knob {
    fn evaluate(&mut self, batch: &EncodedBatch, out: &mut Vec<Evaluation>) {
        for item in batch.iter() {
            let zobrist = u64::from_le_bytes(item[0..8].try_into().expect("eight bytes"));
            let legal = u32::from_le_bytes(item[8..12].try_into().expect("four bytes")) as usize;
            out.push(self.answer(zobrist, legal));
        }
    }
}

#[test]
fn every_probe_position_is_a_legal_live_game() {
    for (index, position) in probe_positions().into_iter().enumerate() {
        position.audit().unwrap_or_else(|e| {
            panic!("probe position {index} fails its own integrity check: {e}")
        });
        assert!(!position.is_terminal(), "probe position {index} is decided");
        assert!(
            position.legal_count() > 0,
            "probe position {index} has no action to carry a prior"
        );
    }
}

#[test]
fn the_probe_set_covers_ten_distinct_plies_from_the_opening_to_a_deep_midgame() {
    let plies: Vec<u32> = probe_positions()
        .iter()
        .map(Position::stone_count)
        .collect();
    assert_eq!(plies, vec![0, 1, 2, 5, 9, 10, 11, 12, 13, 21]);
}

#[test]
fn the_probe_set_covers_both_movers_and_both_stones_of_a_turn() {
    // The set includes a mid-turn state with the same mover as the prior ply.
    let phases: Vec<_> = probe_positions().iter().map(Position::phase).collect();
    assert!(
        phases.contains(&hexo_engine::TurnPhase::Opening),
        "{phases:?}"
    );
    assert!(
        phases.contains(&hexo_engine::TurnPhase::FirstStone),
        "{phases:?}"
    );
    assert!(
        phases.contains(&hexo_engine::TurnPhase::SecondStone),
        "{phases:?}"
    );

    let movers: Vec<_> = probe_positions()
        .iter()
        .map(Position::current_player)
        .collect();
    assert!(movers.contains(&hexo_engine::Player::P0), "{movers:?}");
    assert!(movers.contains(&hexo_engine::Player::P1), "{movers:?}");
}

#[test]
fn the_probe_set_spans_frontier_widths_from_one_action_to_hundreds() {
    // Legal counts vary across the set to exercise ragged encodings.
    let counts: Vec<usize> = probe_positions()
        .iter()
        .map(Position::legal_count)
        .collect();
    assert_eq!(counts[0], 1, "the opening has one legal cell: {counts:?}");
    let largest = *counts.iter().max().expect("ten positions");
    assert!(largest >= 3 * counts[1], "{counts:?}");
}

#[test]
fn the_same_weights_answer_the_probe_with_the_same_hash_every_time() {
    let first = probe_hash(&Bytes, &mut Knob { knob: 0xfeed });
    let second = probe_hash(&Bytes, &mut Knob { knob: 0xfeed });
    assert_eq!(first, second);
}

#[test]
fn different_weights_answer_the_probe_with_a_different_hash() {
    let first = probe_hash(&Bytes, &mut Knob { knob: 0xfeed });
    let second = probe_hash(&Bytes, &mut Knob { knob: 0xfeee });
    assert_ne!(first, second);
}

#[test]
fn the_hash_moves_when_a_single_value_moves_in_its_last_bit() {
    // The hash covers exact output bytes.
    struct Nudged {
        inner: Knob,
        nudge: bool,
    }
    impl Evaluator for Nudged {
        fn evaluate(&mut self, batch: &EncodedBatch, out: &mut Vec<Evaluation>) {
            let before = out.len();
            self.inner.evaluate(batch, out);
            if self.nudge {
                let first = &mut out[before];
                first.value = f32::from_bits(first.value.to_bits() ^ 1);
            }
        }
    }

    let plain = probe_hash(
        &Bytes,
        &mut Nudged {
            inner: Knob { knob: 0xfeed },
            nudge: false,
        },
    );
    let nudged = probe_hash(
        &Bytes,
        &mut Nudged {
            inner: Knob { knob: 0xfeed },
            nudge: true,
        },
    );
    assert_ne!(plain, nudged);
}

#[test]
#[should_panic(expected = "the evaluator answered 9 of 10 probe positions")]
fn an_evaluator_that_drops_an_answer_is_caught_rather_than_hashed_misaligned() {
    struct Short(Knob);
    impl Evaluator for Short {
        fn evaluate(&mut self, batch: &EncodedBatch, out: &mut Vec<Evaluation>) {
            self.0.evaluate(batch, out);
            out.pop();
        }
    }
    let _ = probe_hash(&Bytes, &mut Short(Knob { knob: 1 }));
}

#[test]
#[should_panic(expected = "priors are indexed by the engine's canonical legal order")]
fn an_evaluator_whose_priors_do_not_match_the_action_set_is_caught() {
    struct Ragged(Knob);
    impl Evaluator for Ragged {
        fn evaluate(&mut self, batch: &EncodedBatch, out: &mut Vec<Evaluation>) {
            let before = out.len();
            self.0.evaluate(batch, out);
            let mut priors = out[before].priors.to_vec();
            priors.push(0.0);
            out[before].priors = priors.into();
        }
    }
    let _ = probe_hash(&Bytes, &mut Ragged(Knob { knob: 1 }));
}
