//! The command line: two subcommands, and the configuration each of them parses
//! into.
//!
//! `docs/CONTAINER_SPEC.md` §3 ships only the subcommands that work. `serve` and
//! `play` are not here, not even as stubs that print "not implemented": both are
//! entirely wire protocol, there is no wire protocol, and a stub would publish a
//! command line before the thing behind it is designed — the flags it guessed
//! would become the constraint the real implementation has to argue its way out
//! of.
//!
//! Nothing is defaulted that decides how a run behaves. The defaults that do
//! exist — batch size, ply cap, the flush window, the worker count — are
//! operational tuning knobs whose value changes how fast the same run goes, not
//! what it is.

use crate::error::BotError;
use std::num::{NonZeroU32, NonZeroUsize};
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

/// How to invoke the binary, printed with every usage error.
pub const USAGE: &str = "\
usage:
  hexo-bot train --run-dir <dir> --run-id <id> --package <name> --epochs <n> --games <n>
                 [--package-config <string>] [--batch <n>] [--threads <n>]
                 [--batch-wait-ms <n>] [--ply-cap <n>] [--eval-every <n>]
                 [--eval-games <n>] [--resume]

  hexo-bot match --games <n> --seat <spec> --seat <spec>
                 [--batch <n>] [--threads <n>] [--batch-wait-ms <n>]
                 [--ply-cap <n>] [--report <path>]

  a seat spec is `;`-separated: package=<name>;checkpoint=<dir>[;config=<string>][;variant=<name>]";

/// Default batch size: how many encoded leaves one `Evaluator::evaluate` call
/// answers. `docs/CONTAINER_SPEC.md` §7.2 — B comes from what the device wants
/// and is a different number from the game count.
const DEFAULT_BATCH: usize = 64;

/// Default ply cap, matching `hexo_runner::GameSpec::default`.
const DEFAULT_PLY_CAP: u32 = 512;

/// Default number of evaluation games per pairing.
const DEFAULT_EVAL_GAMES: usize = 32;

/// Default flush window for a partially filled batch, in milliseconds.
///
/// Purely a latency/throughput trade: the batcher waits this long for more
/// leaves before crossing with what it has. Too small and the device sees narrow
/// batches; too large and every lane in the sweep idles for it. Two milliseconds
/// is a starting point, not a measured optimum.
const DEFAULT_BATCH_WAIT_MS: u64 = 2;

/// How many hardware threads the worker pool leaves for everything else.
///
/// `docs/CONTAINER_SPEC.md` §7.1 and §13: the batcher, the record writer, and —
/// once a Python-backed package exists — the interpreter's own pools all need
/// somewhere to run, and a pool sized to the whole machine takes it from them.
const RESERVED_THREADS: usize = 3;

/// A parsed command line.
#[derive(Debug)]
pub enum Command {
    /// A whole training run: self-play, fit, checkpoint, and evaluation, in one
    /// long-lived process.
    Train(TrainConfig),
    /// One head-to-head between two seats over fixed weights.
    Match(MatchConfig),
}

impl Command {
    /// The stop flag this command will watch, for the signal handler to set.
    #[must_use]
    pub fn stop(&self) -> &Arc<AtomicBool> {
        match self {
            Self::Train(config) => &config.stop,
            Self::Match(config) => &config.stop,
        }
    }
}

/// Everything `train` was told.
///
/// Every field is what an operator typed or what a documented default supplied,
/// and the run manifest is written from it. There is no seed field, deliberately
/// — see [`crate::train`].
#[derive(Debug)]
pub struct TrainConfig {
    /// Root of the accumulated state; the run lives at `<run-dir>/runs/<run-id>`.
    pub run_dir: PathBuf,
    /// The run's name, and a single path component.
    pub run_id: String,
    /// The registry name of the model package to train.
    pub package: String,
    /// The package's own configuration string, handed over verbatim.
    pub package_config: String,
    /// How many epochs the run is for. A resume may raise it and may not lower
    /// it.
    pub epochs: NonZeroU32,
    /// Concurrent games per self-play phase, which is also how many games one
    /// epoch produces.
    pub games: NonZeroUsize,
    /// How many encoded leaves one evaluator call answers.
    pub batch: NonZeroUsize,
    /// Worker threads in the sweep, over and above the batcher and the writer.
    pub threads: NonZeroUsize,
    /// How long the batcher waits for a partial batch to fill.
    pub batch_wait: Duration,
    /// The placement cap every game in the run is played under.
    pub ply_cap: NonZeroU32,
    /// Run an evaluation round every this many epochs; zero never runs one.
    pub eval_every: u32,
    /// Games per evaluation pairing.
    pub eval_games: NonZeroUsize,
    /// Continue an existing run rather than refusing its directory.
    pub resume: bool,
    /// Set to ask the run to stop at its next checkpointable moment.
    pub stop: Arc<AtomicBool>,
}

/// Everything `match` was told.
#[derive(Debug)]
pub struct MatchConfig {
    /// How many games to play. Also the lane count: every game is in flight at
    /// once, which is what makes the batches wide (`CONTAINER_SPEC.md` §7.2).
    pub games: NonZeroUsize,
    /// How many encoded leaves one evaluator call answers.
    pub batch: NonZeroUsize,
    /// Worker threads in the sweep.
    pub threads: NonZeroUsize,
    /// How long the batcher waits for a partial batch to fill.
    pub batch_wait: Duration,
    /// The placement cap every game is played under.
    pub ply_cap: NonZeroU32,
    /// The two competitors, in the order the flags were given. The first is
    /// `P0` in even-numbered lanes and `P1` in odd ones.
    pub seats: [SeatSpec; 2],
    /// Where to also write the JSON report, beyond stdout.
    pub report: Option<PathBuf>,
    /// Set to ask the match to stop.
    pub stop: Arc<AtomicBool>,
}

/// One competitor in a `match`: which package, which weights, which search.
///
/// `;`-separated rather than `,`-separated because a package configuration
/// string and a session variant name both contain commas —
/// `mcts:visits=64,inflight=8,cpuct=1.5` is one value, not three.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SeatSpec {
    /// The registry name of the package filling this seat.
    pub package: String,
    /// The checkpoint directory its weights are loaded from.
    pub checkpoint: PathBuf,
    /// The package's configuration string; empty when the spec omits it.
    pub config: String,
    /// A session variant name, or `None` to use the package's evaluation mode.
    pub variant: Option<String>,
}

/// Read a command line.
///
/// The arguments are the ones after the binary's own name.
///
/// # Errors
///
/// [`BotError::Usage`] for anything that is not a command line this binary
/// takes: no subcommand, an unknown one, an unknown flag, a flag stated twice, a
/// flag missing its value, a value that is not the kind of number the flag
/// wants, a required flag that was not given, or a `--seat` spec that does not
/// parse.
pub fn parse<I, S>(args: I) -> Result<Command, BotError>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let args: Vec<String> = args.into_iter().map(Into::into).collect();
    let Some((subcommand, rest)) = args.split_first() else {
        return Err(BotError::usage(format!(
            "no subcommand; this binary has `train` and `match`.\n{USAGE}"
        )));
    };
    let mut flags = Flags::split(rest)?;
    match subcommand.as_str() {
        "train" => Ok(Command::Train(train(&mut flags)?)),
        "match" => Ok(Command::Match(matches(&mut flags)?)),
        other => Err(BotError::usage(format!(
            "unknown subcommand {other:?}; this binary has `train` and `match`. `serve` and \
             `play` are not built: they are entirely wire protocol, and there is no wire \
             protocol yet (docs/CONTAINER_SPEC.md §3).\n{USAGE}"
        ))),
    }
}

/// Read `train`'s flags.
fn train(flags: &mut Flags) -> Result<TrainConfig, BotError> {
    let config = TrainConfig {
        run_dir: PathBuf::from(flags.required("--run-dir")?),
        run_id: run_id(&flags.required("--run-id")?)?,
        package: flags.required("--package")?,
        package_config: flags.optional("--package-config")?.unwrap_or_default(),
        epochs: flags.number("--epochs")?.ok_or_else(missing("--epochs"))?,
        games: flags.number("--games")?.ok_or_else(missing("--games"))?,
        batch: flags
            .number("--batch")?
            .unwrap_or(nonzero_usize(DEFAULT_BATCH)),
        threads: match flags.number("--threads")? {
            Some(threads) => threads,
            None => default_threads()?,
        },
        batch_wait: batch_wait(flags)?,
        ply_cap: flags
            .number("--ply-cap")?
            .unwrap_or(nonzero_u32(DEFAULT_PLY_CAP)),
        eval_every: flags.number::<u32>("--eval-every")?.unwrap_or(0),
        eval_games: flags
            .number("--eval-games")?
            .unwrap_or(nonzero_usize(DEFAULT_EVAL_GAMES)),
        resume: flags.switch("--resume"),
        stop: Arc::new(AtomicBool::new(false)),
    };
    flags.finish("train")?;
    Ok(config)
}

/// Read `match`'s flags.
fn matches(flags: &mut Flags) -> Result<MatchConfig, BotError> {
    let seats = flags.repeated("--seat");
    let [first, second] = <[String; 2]>::try_from(seats).map_err(|given: Vec<String>| {
        BotError::usage(format!(
            "a match has exactly two seats, and --seat was given {} time(s).\n{USAGE}",
            given.len(),
        ))
    })?;
    let config = MatchConfig {
        games: flags.number("--games")?.ok_or_else(missing("--games"))?,
        batch: flags
            .number("--batch")?
            .unwrap_or(nonzero_usize(DEFAULT_BATCH)),
        threads: match flags.number("--threads")? {
            Some(threads) => threads,
            None => default_threads()?,
        },
        batch_wait: batch_wait(flags)?,
        ply_cap: flags
            .number("--ply-cap")?
            .unwrap_or(nonzero_u32(DEFAULT_PLY_CAP)),
        seats: [seat(&first)?, seat(&second)?],
        report: flags.optional("--report")?.map(PathBuf::from),
        stop: Arc::new(AtomicBool::new(false)),
    };
    flags.finish("match")?;
    Ok(config)
}

/// Read one `--seat` spec.
///
/// # Errors
///
/// [`BotError::Usage`] for a segment that is not `key=value`, a key this build
/// does not have, a key stated twice, or a missing `package` or `checkpoint`.
pub fn seat(spec: &str) -> Result<SeatSpec, BotError> {
    let bad = |problem: String| {
        BotError::usage(format!(
            "--seat {spec:?}: {problem}; the form is \
             `package=<name>;checkpoint=<dir>[;config=<string>][;variant=<name>]`"
        ))
    };

    let mut package = None;
    let mut checkpoint = None;
    let mut config = None;
    let mut variant = None;
    for segment in spec.split(';') {
        let Some((key, value)) = segment.split_once('=') else {
            return Err(bad(format!("{segment:?} is not a `key=value` pair")));
        };
        let slot = match key {
            "package" => &mut package,
            "checkpoint" => &mut checkpoint,
            "config" => &mut config,
            "variant" => &mut variant,
            _ => {
                return Err(bad(format!(
                    "unknown key {key:?}; a seat takes `package`, `checkpoint`, `config`, and \
                     `variant`"
                )));
            }
        };
        if slot.is_some() {
            return Err(bad(format!("{key:?} is stated twice")));
        }
        *slot = Some(value.to_owned());
    }

    Ok(SeatSpec {
        package: package.ok_or_else(|| bad("no `package`".to_owned()))?,
        checkpoint: PathBuf::from(checkpoint.ok_or_else(|| bad("no `checkpoint`".to_owned()))?),
        // Absent is not the same as guessed: an empty configuration string is
        // handed to the package, and the package decides whether it can use one.
        config: config.unwrap_or_default(),
        variant,
    })
}

/// Hold `--run-id` to being one path component.
///
/// The run id names a directory and is written into every shard header, so a
/// value carrying a separator or a `..` would place a run somewhere the operator
/// did not ask for and label its records with a name no path could be built
/// from.
fn run_id(id: &str) -> Result<String, BotError> {
    let ok = !id.is_empty()
        && id != "."
        && id != ".."
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'));
    if !ok {
        return Err(BotError::usage(format!(
            "--run-id {id:?} is not a usable run id; it names a directory and is written into \
             every shard header, so it must be a non-empty string of ASCII letters, digits, `-`, \
             `_`, and `.`, and not `.` or `..`"
        )));
    }
    Ok(id.to_owned())
}

/// The flush window, read as whole milliseconds.
fn batch_wait(flags: &mut Flags) -> Result<Duration, BotError> {
    let ms = flags
        .number::<u64>("--batch-wait-ms")?
        .unwrap_or(DEFAULT_BATCH_WAIT_MS);
    Ok(Duration::from_millis(ms))
}

/// The worker count when `--threads` was not given.
///
/// # Errors
///
/// [`BotError::Usage`] if the host's parallelism cannot be read. Falling back to
/// one thread would be a silently substituted default for a value that decides
/// how a multi-day run performs, so the operator is asked instead.
fn default_threads() -> Result<NonZeroUsize, BotError> {
    let available = std::thread::available_parallelism().map_err(|source| {
        BotError::usage(format!(
            "--threads was not given and this host's parallelism cannot be read ({source}); pass \
             --threads"
        ))
    })?;
    Ok(nonzero_usize(
        available.get().saturating_sub(RESERVED_THREADS).max(1),
    ))
}

/// A required flag that was not given.
fn missing(name: &'static str) -> impl Fn() -> BotError {
    move || BotError::usage(format!("{name} is required and has no default.\n{USAGE}"))
}

/// A constant that is not zero.
fn nonzero_usize(value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).expect("the compiled-in defaults are not zero")
}

/// A constant that is not zero.
fn nonzero_u32(value: u32) -> NonZeroU32 {
    NonZeroU32::new(value).expect("the compiled-in defaults are not zero")
}

/// One pass over the argument list, and what is left of it.
///
/// Kept as a list rather than a map so that a flag stated twice is visible: a
/// map would silently keep one of the two, which is the last-one-wins lenience
/// this workspace refuses.
#[derive(Debug)]
struct Flags {
    /// `(name, value)` in the order given; `None` for a switch.
    given: Vec<(String, Option<String>)>,
    /// Whether each entry has been read, so `finish` can name what was not.
    used: Vec<bool>,
}

impl Flags {
    /// Cut the arguments into flags and their values.
    fn split(args: &[String]) -> Result<Self, BotError> {
        let mut given = Vec::new();
        let mut rest = args.iter();
        while let Some(argument) = rest.next() {
            if !argument.starts_with("--") {
                return Err(BotError::usage(format!(
                    "{argument:?} is not a flag; every argument after the subcommand is a \
                     `--flag` or its value.\n{USAGE}"
                )));
            }
            if argument == "--resume" {
                given.push((argument.clone(), None));
                continue;
            }
            let Some(value) = rest.next() else {
                return Err(BotError::usage(format!(
                    "{argument} needs a value.\n{USAGE}"
                )));
            };
            if value.starts_with("--") {
                return Err(BotError::usage(format!(
                    "{argument} needs a value, and {value:?} is another flag.\n{USAGE}"
                )));
            }
            given.push((argument.clone(), Some(value.clone())));
        }
        let used = vec![false; given.len()];
        Ok(Self { given, used })
    }

    /// The value of a flag stated at most once.
    fn optional(&mut self, name: &str) -> Result<Option<String>, BotError> {
        let mut found: Option<String> = None;
        for (index, (flag, value)) in self.given.iter().enumerate() {
            if flag != name {
                continue;
            }
            if found.is_some() {
                return Err(BotError::usage(format!(
                    "{name} is stated more than once; a repeated flag is a mistake, not a \
                     last-one-wins"
                )));
            }
            let value = value
                .clone()
                .ok_or_else(|| BotError::usage(format!("{name} needs a value.\n{USAGE}")))?;
            found = Some(value);
            self.used[index] = true;
        }
        Ok(found)
    }

    /// The value of a flag that has to be there.
    fn required(&mut self, name: &'static str) -> Result<String, BotError> {
        self.optional(name)?.ok_or_else(missing(name))
    }

    /// The parsed value of a flag stated at most once.
    fn number<T: FromStr>(&mut self, name: &str) -> Result<Option<T>, BotError> {
        let Some(text) = self.optional(name)? else {
            return Ok(None);
        };
        let parsed = text.parse::<T>().map_err(|_| {
            BotError::usage(format!(
                "{name} is {text:?}, which is not a {} this flag can use",
                core::any::type_name::<T>(),
            ))
        })?;
        Ok(Some(parsed))
    }

    /// Every value of a flag that may be stated more than once.
    fn repeated(&mut self, name: &str) -> Vec<String> {
        let mut found = Vec::new();
        for (index, (flag, value)) in self.given.iter().enumerate() {
            if flag == name {
                self.used[index] = true;
                if let Some(value) = value {
                    found.push(value.clone());
                }
            }
        }
        found
    }

    /// Whether a valueless flag was given.
    fn switch(&mut self, name: &str) -> bool {
        let mut present = false;
        for (index, (flag, _)) in self.given.iter().enumerate() {
            if flag == name {
                self.used[index] = true;
                present = true;
            }
        }
        present
    }

    /// Refuse anything the subcommand did not ask for.
    fn finish(&self, subcommand: &str) -> Result<(), BotError> {
        let unread: Vec<&str> = self
            .given
            .iter()
            .zip(&self.used)
            .filter(|(_, used)| !**used)
            .map(|((flag, _), _)| flag.as_str())
            .collect();
        if unread.is_empty() {
            return Ok(());
        }
        Err(BotError::usage(format!(
            "`{subcommand}` does not take {unread:?}.\n{USAGE}"
        )))
    }
}
