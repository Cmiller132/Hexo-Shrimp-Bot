//! The executable-only PyO3 crossing for MantisNet.
//!
//! The model package owns encoding and evaluation semantics. This module owns
//! the one operation that cannot live in a logic crate: loading and calling the
//! production Python/Torch module. A `LiveForward` holds no Rust-side model
//! opinion; it converts the package's public raw batch to CPU tensors and hands
//! the two raw cell heads back unchanged.

use hexo_model_mantisnet::encoder::RawBatch;
use hexo_model_mantisnet::{BoxError, Forward, ForwardLoader, RawOutputs};
use pyo3::exceptions::{PyOverflowError, PyRuntimeError, PyValueError};
use pyo3::types::{
    PyAnyMethods, PyByteArray, PyBytes, PyBytesMethods, PyDict, PyDictMethods, PyList,
    PyListMethods, PyModule, PyTuple,
};
use pyo3::{Bound, Py, PyAny, PyErr, PyResult, Python};
use serde_json::Value;
use std::fmt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, OnceLock};

const PYTHON_ENV: &str = "HEXO_PYTHON";
const DISCOVERY_MARKER: &str = "__HEXO_PYTHON_PATH__=";
const DISCOVERY_SCRIPT: &str = concat!(
    "import json, sys\n",
    "print(",
    "\"__HEXO_PYTHON_PATH__=\" + ",
    "json.dumps({\"major\": sys.version_info.major, ",
    "\"minor\": sys.version_info.minor, \"path\": sys.path}))\n",
);

static PYTHON_SETUP: OnceLock<Result<(), SetupError>> = OnceLock::new();

/// Stateless constructor for a live, CPU-only MantisNet forward.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct PythonForwardLoader;

impl ForwardLoader for PythonForwardLoader {
    fn load(&self, weights: &Path) -> Result<Box<dyn Forward>, BoxError> {
        ensure_python()?;

        let loaded = Python::attach(|py| -> PyResult<LiveForward> {
            let run = PyModule::import(py, "mantisnet.klent.run")?;
            let model = run
                .getattr("load_model")?
                // `&Path` converts through pathlib.Path. This is deliberately the
                // production loader, including its checkpoint version refusal.
                .call1((weights, "cpu"))?
                .unbind();
            let torch = PyModule::import(py, "torch")?.unbind();
            // The constructor helper, not the bare dataclass: the builder
            // derives the relay tables from the decoder incidence at
            // construction, and that derivation lives in one place.
            let batch_type = PyModule::import(py, "mantisnet.builder")?
                .getattr("batch_from_arrays")?
                .unbind();
            Ok(LiveForward {
                model,
                torch,
                batch_type,
            })
        });

        loaded
            .map(|forward| Box::new(forward) as Box<dyn Forward>)
            .map_err(|source| {
                Box::new(PythonCallError {
                    doing: "loading a CPU MantisNet through mantisnet.klent.run.load_model",
                    source,
                }) as BoxError
            })
    }
}

struct LiveForward {
    model: Py<PyAny>,
    torch: Py<PyModule>,
    batch_type: Py<PyAny>,
}

impl Forward for LiveForward {
    fn forward(&mut self, batch: &RawBatch) -> Result<RawOutputs, BoxError> {
        // This is the only interpreter attachment in a forward. Tensor
        // construction, the model call, and extraction of both heads all happen
        // while this one attachment is held.
        Python::attach(|py| {
            let torch = self.torch.bind(py);
            let py_batch = build_batch(py, torch, self.batch_type.bind(py), batch)?;
            let output = call_inference(py, torch, self.model.bind(py), &py_batch)?;
            Ok(RawOutputs {
                policy_logits: tensor_f32_vec(py, torch, &output, "policy_logits")?,
                q_values: tensor_f32_vec(py, torch, &output, "q_values")?,
            })
        })
        .map_err(|source| {
            Box::new(PythonCallError {
                doing: "running the live CPU MantisNet forward",
                source,
            }) as BoxError
        })
    }
}

fn call_inference<'py>(
    py: Python<'py>,
    torch: &Bound<'py, PyModule>,
    model: &Bound<'py, PyAny>,
    batch: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let inference = torch.call_method0("inference_mode")?;
    inference.call_method0("__enter__")?;
    let output = model.call1((batch,));
    let exited = inference.call_method1("__exit__", (py.None(), py.None(), py.None()));
    match (output, exited) {
        (Err(source), _) => Err(source),
        (Ok(_), Err(source)) => Err(source),
        (Ok(output), Ok(_)) => Ok(output),
    }
}

fn build_batch<'py>(
    py: Python<'py>,
    torch: &Bound<'py, PyModule>,
    batch_type: &Bound<'py, PyAny>,
    batch: &RawBatch,
) -> PyResult<Bound<'py, PyAny>> {
    let legal_offset_rows = batch
        .n_pos
        .checked_add(1)
        .ok_or_else(|| PyOverflowError::new_err("MantisNet batch position count overflowed"))?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("n_pos", batch.n_pos)?;
    kwargs.set_item("max_t", batch.max_t)?;
    kwargs.set_item("max_w", batch.max_w)?;
    kwargs.set_item("n_cells", batch.cell_pos.len())?;

    set_tensor(
        py,
        torch,
        &kwargs,
        "stone_own",
        &batch.stone_own,
        &[batch.stone_own.len()],
    )?;
    set_tensor(
        py,
        torch,
        &kwargs,
        "window_feat",
        &batch.window_feat,
        &[batch.window_feat.len()],
    )?;
    set_tensor(
        py,
        torch,
        &kwargs,
        "window_id",
        &batch.window_id,
        &[batch.window_feat.len(), 3],
    )?;
    set_tensor(
        py,
        torch,
        &kwargs,
        "moves_idx",
        &batch.moves_idx,
        &[batch.n_pos],
    )?;
    set_tensor(
        py,
        torch,
        &kwargs,
        "inc_stone",
        &batch.inc_stone,
        &[batch.inc_stone.len()],
    )?;
    set_tensor(
        py,
        torch,
        &kwargs,
        "inc_window",
        &batch.inc_window,
        &[batch.inc_window.len()],
    )?;
    set_tensor(
        py,
        torch,
        &kwargs,
        "inc_class",
        &batch.inc_class,
        &[batch.inc_class.len()],
    )?;
    set_tensor(
        py,
        torch,
        &kwargs,
        "stone_slot",
        &batch.stone_slot,
        &[batch.stone_slot.len()],
    )?;
    set_tensor(
        py,
        torch,
        &kwargs,
        "coords",
        &batch.coords,
        &[batch.n_pos, batch.max_t, 2],
    )?;
    set_tensor(
        py,
        torch,
        &kwargs,
        "attn_valid",
        &batch.attn_valid,
        &[batch.n_pos, batch.max_t],
    )?;
    set_tensor(
        py,
        torch,
        &kwargs,
        "window_slot",
        &batch.window_slot,
        &[batch.window_slot.len()],
    )?;
    set_tensor(
        py,
        torch,
        &kwargs,
        "value_valid",
        &batch.value_valid,
        &[batch.n_pos, batch.max_w],
    )?;
    set_tensor(
        py,
        torch,
        &kwargs,
        "legal_offsets",
        &batch.legal_offsets,
        &[legal_offset_rows],
    )?;
    set_tensor(
        py,
        torch,
        &kwargs,
        "cell_pos",
        &batch.cell_pos,
        &[batch.cell_pos.len()],
    )?;
    set_tensor(
        py,
        torch,
        &kwargs,
        "dec_cell",
        &batch.dec_cell,
        &[batch.dec_cell.len()],
    )?;
    set_tensor(
        py,
        torch,
        &kwargs,
        "dec_window",
        &batch.dec_window,
        &[batch.dec_window.len()],
    )?;
    set_tensor(
        py,
        torch,
        &kwargs,
        "dec_class",
        &batch.dec_class,
        &[batch.dec_class.len()],
    )?;
    set_tensor(
        py,
        torch,
        &kwargs,
        "act_class",
        &batch.act_class,
        &[batch.act_class.len()],
    )?;
    set_tensor(
        py,
        torch,
        &kwargs,
        "act_rev",
        &batch.act_rev,
        &[batch.act_rev.len()],
    )?;
    set_tensor(
        py,
        torch,
        &kwargs,
        "act_empty",
        &batch.act_empty,
        &[batch.cell_pos.len(), 3],
    )?;

    batch_type.call((), Some(&kwargs))
}

fn set_tensor<'py, T: TensorElement>(
    py: Python<'py>,
    torch: &Bound<'py, PyModule>,
    kwargs: &Bound<'py, PyDict>,
    name: &'static str,
    values: &[T],
    shape: &[usize],
) -> PyResult<()> {
    kwargs.set_item(name, cpu_tensor(py, torch, name, values, shape)?)
}

trait TensorElement: Copy {
    const DTYPE: &'static str;

    fn append_bytes(self, out: &mut Vec<u8>);
}

impl TensorElement for i64 {
    const DTYPE: &'static str = "int64";

    fn append_bytes(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_ne_bytes());
    }
}

impl TensorElement for i32 {
    const DTYPE: &'static str = "int32";

    fn append_bytes(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_ne_bytes());
    }
}

impl TensorElement for bool {
    const DTYPE: &'static str = "bool";

    fn append_bytes(self, out: &mut Vec<u8>) {
        out.push(u8::from(self));
    }
}

fn cpu_tensor<'py, T: TensorElement>(
    py: Python<'py>,
    torch: &Bound<'py, PyModule>,
    name: &'static str,
    values: &[T],
    shape: &[usize],
) -> PyResult<Bound<'py, PyAny>> {
    let expected = shape.iter().try_fold(1usize, |len, &dim| {
        len.checked_mul(dim).ok_or_else(|| {
            PyOverflowError::new_err(format!(
                "MantisNet tensor {name} shape {shape:?} overflowed"
            ))
        })
    })?;
    if expected != values.len() {
        return Err(PyValueError::new_err(format!(
            "MantisNet tensor {name} has {} values, but shape {shape:?} requires {expected}",
            values.len(),
        )));
    }

    let shape = PyTuple::new(py, shape.iter().copied())?;
    let options = PyDict::new(py);
    options.set_item("dtype", torch.getattr(T::DTYPE)?)?;
    options.set_item("device", "cpu")?;
    if values.is_empty() {
        return torch.call_method("empty", (&shape,), Some(&options));
    }

    let capacity = values
        .len()
        .checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| {
            PyOverflowError::new_err(format!("MantisNet tensor {name} byte size overflowed"))
        })?;
    let mut bytes = Vec::with_capacity(capacity);
    for &value in values {
        value.append_bytes(&mut bytes);
    }

    // `frombuffer` is the one bulk transfer into Python. Clone immediately so
    // the tensor owns its storage rather than retaining this temporary
    // bytearray, then restore the model's declared shape.
    options.del_item("device")?;
    let buffer = PyByteArray::new(py, &bytes);
    torch
        .call_method("frombuffer", (&buffer,), Some(&options))?
        .call_method0("clone")?
        .call_method1("reshape", (&shape,))
}

fn tensor_f32_vec(
    py: Python<'_>,
    torch: &Bound<'_, PyModule>,
    output: &Bound<'_, PyAny>,
    field: &'static str,
) -> PyResult<Vec<f32>> {
    let options = PyDict::new(py);
    options.set_item("dtype", torch.getattr("float32")?)?;
    options.set_item("device", "cpu")?;
    let bytes = output
        .getattr(field)?
        .call_method0("detach")?
        .call_method("to", (), Some(&options))?
        .call_method0("contiguous")?
        .call_method1("reshape", (-1,))?
        .call_method0("numpy")?
        .call_method0("tobytes")?;
    let raw = bytes.cast::<PyBytes>()?.as_bytes();
    if !raw.len().is_multiple_of(std::mem::size_of::<f32>()) {
        return Err(PyValueError::new_err(format!(
            "MantisNet output {field} returned {} bytes, not whole float32 values",
            raw.len(),
        )));
    }
    Ok(raw
        .chunks_exact(4)
        .map(|chunk| f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn ensure_python() -> Result<(), BoxError> {
    PYTHON_SETUP
        .get_or_init(configure_python)
        .clone()
        .map_err(|source| Box::new(source) as BoxError)
}

fn configure_python() -> Result<(), SetupError> {
    let discovery = std::env::var_os(PYTHON_ENV)
        .map(|interpreter| discover_python(PathBuf::from(interpreter)))
        .transpose()?;

    // Explicit rather than PyO3's optional auto-initialize feature: setup is a
    // named, one-shot operation and is complete before any live model exists.
    Python::initialize();
    if let Some(discovery) = discovery {
        Python::attach(|py| apply_discovery(py, &discovery)).map_err(|source| {
            SetupError::Python {
                doing: "matching HEXO_PYTHON to the embedded interpreter and setting sys.path",
                source: Arc::new(source),
            }
        })?;
    }
    Ok(())
}

fn discover_python(interpreter: PathBuf) -> Result<Discovery, SetupError> {
    if interpreter.as_os_str().is_empty() {
        return Err(SetupError::MalformedDiscovery {
            interpreter,
            problem: format!("{PYTHON_ENV} is set but empty"),
        });
    }
    let output = Command::new(&interpreter)
        .args(["-c", DISCOVERY_SCRIPT])
        .output()
        .map_err(|source| SetupError::Spawn {
            interpreter: interpreter.clone(),
            source: Arc::new(source),
        })?;
    if !output.status.success() {
        return Err(SetupError::InterpreterFailed {
            interpreter,
            status: output.status.to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).trim().to_owned(),
        });
    }
    parse_discovery(&interpreter, output.stdout)
}

fn parse_discovery(interpreter: &Path, stdout: Vec<u8>) -> Result<Discovery, SetupError> {
    let stdout = String::from_utf8(stdout).map_err(|source| SetupError::DiscoveryUtf8 {
        interpreter: interpreter.to_path_buf(),
        source: Arc::new(source),
    })?;
    let json = stdout
        .lines()
        .rev()
        .find_map(|line| line.strip_prefix(DISCOVERY_MARKER))
        .ok_or_else(|| SetupError::MalformedDiscovery {
            interpreter: interpreter.to_path_buf(),
            problem: format!(
                "discovery output had no {DISCOVERY_MARKER:?} record; stdout was {stdout:?}"
            ),
        })?;
    let document: Value =
        serde_json::from_str(json).map_err(|source| SetupError::DiscoveryJson {
            interpreter: interpreter.to_path_buf(),
            source: Arc::new(source),
        })?;
    let malformed = |problem: String| SetupError::MalformedDiscovery {
        interpreter: interpreter.to_path_buf(),
        problem,
    };
    let object = document
        .as_object()
        .ok_or_else(|| malformed("discovery record is not a JSON object".to_owned()))?;
    let major = object.get("major").and_then(Value::as_u64).ok_or_else(|| {
        malformed("discovery field `major` is not an unsigned integer".to_owned())
    })?;
    let minor = object.get("minor").and_then(Value::as_u64).ok_or_else(|| {
        malformed("discovery field `minor` is not an unsigned integer".to_owned())
    })?;
    let entries = object
        .get("path")
        .and_then(Value::as_array)
        .ok_or_else(|| malformed("discovery field `path` is not an array".to_owned()))?;
    let mut path = Vec::with_capacity(entries.len());
    for (index, entry) in entries.iter().enumerate() {
        path.push(
            entry
                .as_str()
                .ok_or_else(|| {
                    malformed(format!("discovery field `path[{index}]` is not a string"))
                })?
                .to_owned(),
        );
    }
    Ok(Discovery {
        interpreter: interpreter.to_path_buf(),
        major,
        minor,
        path,
    })
}

fn apply_discovery(py: Python<'_>, discovery: &Discovery) -> PyResult<()> {
    let sys = PyModule::import(py, "sys")?;
    let version = sys.getattr("version_info")?;
    let embedded_major: u64 = version.get_item(0)?.extract()?;
    let embedded_minor: u64 = version.get_item(1)?.extract()?;
    if (embedded_major, embedded_minor) != (discovery.major, discovery.minor) {
        return Err(PyRuntimeError::new_err(format!(
            "{PYTHON_ENV}={} is Python {}.{}, but hexo-bot is linked to Python \
             {embedded_major}.{embedded_minor}; rebuild with PYO3_PYTHON set to the same interpreter",
            discovery.interpreter.display(),
            discovery.major,
            discovery.minor,
        )));
    }

    let current = sys.getattr("path")?;
    let current = current.cast::<PyList>()?;
    let replacement = PyList::new(py, &discovery.path)?;
    current.set_slice(0, current.len(), &replacement)
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Discovery {
    interpreter: PathBuf,
    major: u64,
    minor: u64,
    path: Vec<String>,
}

#[derive(Clone, Debug)]
enum SetupError {
    Spawn {
        interpreter: PathBuf,
        source: Arc<std::io::Error>,
    },
    InterpreterFailed {
        interpreter: PathBuf,
        status: String,
        stderr: String,
    },
    DiscoveryUtf8 {
        interpreter: PathBuf,
        source: Arc<std::string::FromUtf8Error>,
    },
    DiscoveryJson {
        interpreter: PathBuf,
        source: Arc<serde_json::Error>,
    },
    MalformedDiscovery {
        interpreter: PathBuf,
        problem: String,
    },
    Python {
        doing: &'static str,
        source: Arc<PyErr>,
    },
}

impl fmt::Display for SetupError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Spawn {
                interpreter,
                source,
            } => write!(
                f,
                "could not run {PYTHON_ENV}={}: {source}",
                interpreter.display()
            ),
            Self::InterpreterFailed {
                interpreter,
                status,
                stderr,
            } => write!(
                f,
                "{PYTHON_ENV}={} exited {status} while reporting its environment: {stderr}",
                interpreter.display()
            ),
            Self::DiscoveryUtf8 {
                interpreter,
                source,
            } => write!(
                f,
                "{PYTHON_ENV}={} returned non-UTF-8 discovery output: {source}",
                interpreter.display()
            ),
            Self::DiscoveryJson {
                interpreter,
                source,
            } => write!(
                f,
                "{PYTHON_ENV}={} returned malformed discovery JSON: {source}",
                interpreter.display()
            ),
            Self::MalformedDiscovery {
                interpreter,
                problem,
            } => write!(
                f,
                "{PYTHON_ENV}={} returned unusable discovery output: {problem}",
                interpreter.display()
            ),
            Self::Python { doing, source } => write!(f, "Python failed while {doing}: {source}"),
        }
    }
}

impl std::error::Error for SetupError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Spawn { source, .. } => Some(source.as_ref()),
            Self::DiscoveryUtf8 { source, .. } => Some(source.as_ref()),
            Self::DiscoveryJson { source, .. } => Some(source.as_ref()),
            Self::Python { source, .. } => Some(source.as_ref()),
            _ => None,
        }
    }
}

#[derive(Debug)]
struct PythonCallError {
    doing: &'static str,
    source: PyErr,
}

impl fmt::Display for PythonCallError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Python failed while {}: {}", self.doing, self.source)
    }
}

impl std::error::Error for PythonCallError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_marked_discovery_after_unrelated_output() {
        let interpreter = Path::new("C:/venv/Scripts/python.exe");
        let stdout = format!(
            "sitecustomize said hello\n{DISCOVERY_MARKER}{{\"major\":3,\"minor\":13,\
             \"path\":[\"C:/work\", \"C:/venv/Lib/site-packages\"]}}\n"
        )
        .into_bytes();

        assert_eq!(
            parse_discovery(interpreter, stdout).unwrap(),
            Discovery {
                interpreter: interpreter.to_path_buf(),
                major: 3,
                minor: 13,
                path: vec!["C:/work".to_owned(), "C:/venv/Lib/site-packages".to_owned()],
            }
        );
    }

    #[test]
    fn malformed_discovery_names_the_interpreter_and_field() {
        let interpreter = Path::new("/opt/venv/bin/python");
        let stdout = format!("{DISCOVERY_MARKER}{{\"major\":3,\"minor\":13,\"path\":[null]}}\n")
            .into_bytes();

        let error = parse_discovery(interpreter, stdout).unwrap_err();
        let message = error.to_string();
        assert!(message.contains("/opt/venv/bin/python"));
        assert!(message.contains("path[0]"));
    }
}
