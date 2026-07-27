//! The `hexo-bot` binary: parse, arm the stop flag, run, and report.
//!
//! Everything below the command line is `hexo_bot`'s, so the loop this binary
//! exists to run is exercised in-process by the test suite instead of by
//! spawning a child and parsing its output.

use hexo_bot::{BotError, Command, Outcome};
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::Ordering;

/// Exit **0** completed, **2** stopped by signal after finishing cleanly, **1**
/// failed — `docs/CONTAINER_SPEC.md` §8.1, and the reason it is pinned there is
/// that a run that ended is not a run that broke.
fn main() -> ExitCode {
    match run() {
        Ok(outcome) => ExitCode::from(outcome.exit_code()),
        Err(error) => {
            report(&error);
            ExitCode::FAILURE
        }
    }
}

/// Everything the binary does, with one place for failure to come out.
fn run() -> Result<Outcome, BotError> {
    let command = hexo_bot::parse(std::env::args().skip(1))?;

    // `unsafe_code = "forbid"` rules out a hand-rolled handler, so the flag is
    // set by the one small crate that owns that unsafety. With the `termination`
    // feature this is `SIGTERM` in a container and Ctrl-C in a shell, which are
    // the same request.
    let stop = Arc::clone(command.stop());
    ctrlc::set_handler(move || stop.store(true, Ordering::Relaxed)).map_err(BotError::Signal)?;

    match command {
        Command::Train(config) => hexo_bot::train(&config),
        Command::Match(config) => {
            let played = hexo_bot::play_match(&config)?;
            println!("{}", played.report.to_json());
            Ok(played.outcome)
        }
    }
}

/// Print the failure and everything under it.
///
/// The chain matters here: the interesting half of a `hexo-bot` failure is
/// usually a package's or the record format's own error, which carries the
/// version pair or the file offset that locates the problem.
fn report(error: &BotError) {
    eprintln!("hexo-bot: {error}");
    let mut source = std::error::Error::source(error);
    while let Some(cause) = source {
        eprintln!("  caused by: {cause}");
        source = cause.source();
    }
}
