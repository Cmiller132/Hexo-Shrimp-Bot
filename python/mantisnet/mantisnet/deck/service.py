"""The non-HTTP seams of the control deck."""

from __future__ import annotations

import json
import os
import re
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from ..builder import collate_prefixes
from ..klent import telemetry
from ..klent.inspect import inspect_position
from ..klent.run import _versions
from ..model import CRITIC_LOGITS, MantisConfig, MantisNet

_RUN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}\Z")


def json_file(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"missing required run artifact: {path}") from None
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read {path}: {exc}") from exc


def telemetry_connection(run_dir: Path) -> sqlite3.Connection:
    path = run_dir / telemetry.DB_NAME
    if not path.is_file():
        raise FileNotFoundError(f"missing required run artifact: {path}")
    conn = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    try:
        row = conn.execute("SELECT version FROM schema_version").fetchone()
    except sqlite3.Error as exc:
        conn.close()
        raise RuntimeError(f"cannot read telemetry database {path}: {exc}") from exc
    if row is None:  # created, version row not yet committed
        conn.close()
        raise RuntimeError(f"telemetry database {path} is still materializing")
    version = row[0]
    if version != telemetry.SCHEMA_VERSION:
        conn.close()
        raise RuntimeError(
            f"telemetry schema {version} != this build {telemetry.SCHEMA_VERSION}"
        )
    return conn


class RunRegistry:
    def __init__(self, root: Path, command_prefix: list[str] | None = None):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.command_prefix = command_prefix or [sys.executable, "-m", "mantisnet.klent.run"]
        self.children: dict[str, subprocess.Popen] = {}
        self.readers: dict[str, threading.Thread] = {}
        self.lock = threading.RLock()

    def path(self, name: str, *, require=True) -> Path:
        if not _RUN_NAME.fullmatch(name):
            raise ValueError("run name must use letters, digits, dot, dash, or underscore")
        path = (self.root / name).resolve()
        if path.parent != self.root:
            raise ValueError("run escapes the runs root")
        if require and not (path / "config.json").is_file():
            raise FileNotFoundError(f"no run named {name}")
        return path

    def _child_alive(self, name: str) -> bool:
        child = self.children.get(name)
        return child is not None and child.poll() is None

    def _heartbeat_fresh(self, path: Path) -> bool:
        status = path / "status.json"
        return status.is_file() and time.time() - status.stat().st_mtime <= 10

    def active_name(self) -> str | None:
        for name, child in self.children.items():
            if child.poll() is None:
                return name
        for item in self.list():
            if item["state"] == "active":
                return item["name"]
        return None

    def describe(self, name: str) -> dict:
        # A child registered by the deck is describable before its artifacts exist.
        path = self.path(name, require=name not in self.children)
        # JSON and telemetry artifacts may be observed before their writes complete.
        config_path = path / "config.json"
        try:
            config = json_file(config_path) if config_path.is_file() else None
        except json.JSONDecodeError:
            config = None
        status_path = path / "status.json"
        status = json_file(status_path) if status_path.is_file() else None
        child = self.children.get(name)
        alive = self._child_alive(name)
        fresh = self._heartbeat_fresh(path)
        # The heartbeat's iteration is null until the first commit.
        done = (status.get("iteration") or 0) if status else 0
        if (path / telemetry.DB_NAME).is_file():
            # Listings tolerate an incomplete or incompatible telemetry database.
            try:
                with telemetry_connection(path) as conn:
                    row = conn.execute(
                        "SELECT MAX(iteration) FROM iterations"
                    ).fetchone()
            except (RuntimeError, sqlite3.OperationalError):
                row = (None,)
            done = max(done, 0 if row[0] is None else int(row[0]) + 1)
        target = int(config["iterations"]) if config is not None else None
        console = path / "deck-console.log"
        tail = console.read_text(encoding="utf-8", errors="replace")[-4096:] if console.exists() else ""
        if alive or (child is None and fresh):
            state = "active"
        elif target is not None and done >= target:
            state = "completed"
        elif "stopping starved:" in tail:
            state = "starved"
        else:
            state = "stopped"
        return {
            "name": name,
            "state": state,
            "controlled": child is not None,
            "pid": child.pid if child is not None else None,
            "exit_code": child.poll() if child is not None else None,
            "iterations": target,
            "iteration": done,
            "heartbeat": status,
            "heartbeat_age": time.time() - status_path.stat().st_mtime
            if status_path.exists()
            else None,
            "working": bool(alive and not fresh),
            "checkpoints": self.checkpoints(name) if config is not None else [],
        }

    def list(self) -> list[dict]:
        named = {
            path.name
            for path in sorted(self.root.iterdir())
            if path.is_dir() and (path / "config.json").is_file()
        } | set(self.children)
        return [self.describe(name) for name in sorted(named)]

    def checkpoints(self, name: str) -> list[dict]:
        path = self.path(name)
        return [
            {
                "name": file.name,
                "path": str(file),
                "bytes": file.stat().st_size,
                "modified": datetime.fromtimestamp(
                    file.stat().st_mtime, timezone.utc
                ).isoformat(),
                "iteration": int(match.group(1)) if (
                    match := re.fullmatch(r"checkpoint_(\d+)\.pt", file.name)
                ) else None,
            }
            for file in sorted(path.glob("*.pt"))
        ]

    def launch(self, request: dict) -> dict:
        with self.lock:
            if active := self.active_name():
                raise RuntimeError(f"training run {active} is already active")
            name = request["name"]
            out = self.path(name, require=False)
            resume = bool(request.get("resume"))
            if resume:
                self.path(name)
            elif out.exists() and any(out.iterdir()):
                raise FileExistsError(f"{out} exists and is not empty")
            args = [
                *self.command_prefix,
                "--out", str(out),
                "--iterations", str(request["iterations"]),
                "--games", str(request.get("games", 64)),
                "--envs", str(request.get("envs", 256)),
                "--seed", str(request.get("seed", 0)),
                "--lam-ret", str(request.get("lam_ret", float(np.exp(-1 / 16)))),
                "--checkpoint-every", str(request.get("checkpoint_every", 25)),
                "--eval-every", str(request.get("eval_every", 0)),
                "--eval-games", str(request.get("eval_games", 64)),
                "--eval-time", str(request.get("eval_time", 0.1)),
                "--eval-sims", str(request.get("eval_sims", 32)),
                "--device", str(request.get("device", os.environ.get("DECK_DEVICE", "cuda"))),
            ]
            if request.get("eval_depth") is not None:
                # An omitted depth preserves the driver's time-limited uncapped search.
                args += ["--eval-depth", str(request["eval_depth"])]
            if request.get("init_from"):
                args += ["--init-from", str(self.resolve_checkpoint(request["init_from"]))]
            if resume:
                args.append("--resume")
            if request.get("device") == "cpu":
                args.append("--no-compile")
            sealbot = request.get("sealbot") or os.environ.get("SEALBOT_ROOT")
            if request.get("eval_every", 0):
                if not sealbot:
                    raise ValueError("eval_every > 0 needs sealbot or SEALBOT_ROOT")
                args += ["--sealbot", sealbot]
            child = subprocess.Popen(
                args,
                cwd=Path(__file__).parents[2],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            self.children[name] = child
            reader = threading.Thread(
                target=self._capture, args=(name, out, child), daemon=True
            )
            self.readers[name] = reader
            reader.start()
        deadline = time.time() + 2
        while time.time() < deadline and not (out / "config.json").exists():
            if child.poll() is not None:
                reader.join(timeout=1)
                tail = (out / "deck-console.log").read_text(
                    encoding="utf-8", errors="replace"
                )[-2000:] if (out / "deck-console.log").exists() else ""
                raise RuntimeError(f"training child exited {child.returncode}: {tail}")
            time.sleep(0.02)
        return {"name": name, "pid": child.pid, "state": "active", "command": args}

    @staticmethod
    def _capture(name: str, out: Path, child: subprocess.Popen) -> None:
        # Delay the console file until the child establishes the run directory.
        while child.poll() is None and not (out / "config.json").exists():
            time.sleep(0.01)
        out.mkdir(parents=True, exist_ok=True)
        with (out / "deck-console.log").open("a", encoding="utf-8") as log:
            assert child.stdout is not None
            for line in child.stdout:
                log.write(line)
                log.flush()

    def sentinel(self, name: str, sentinel: str) -> dict:
        path = self.path(name)
        (path / sentinel).touch(exist_ok=True)
        return {"run": name, "sentinel": sentinel, "accepted": True}

    def kill(self, name: str, confirm: bool) -> dict:
        if not confirm:
            raise ValueError("kill requires confirm: true")
        child = self.children.get(name)
        if child is None:
            raise RuntimeError(f"run {name} was not launched by this deck")
        if child.poll() is not None:
            raise RuntimeError(f"run {name} already exited with {child.returncode}")
        child.send_signal(signal.SIGTERM)
        return {"run": name, "signal": "SIGTERM", "accepted": True}

    def resolve_checkpoint(self, value: str) -> Path:
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = (self.root / candidate).resolve()
        else:
            candidate = candidate.resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"no checkpoint {candidate}")
        return candidate

    def manifest(self, name: str) -> dict:
        path = self.path(name)
        invocations = []
        inv = path / "invocations.jsonl"
        if inv.exists():
            for line_no, line in enumerate(inv.read_text(encoding="utf-8").splitlines(), 1):
                try:
                    invocations.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"{inv}:{line_no}: {exc}") from exc
        return {"config": json_file(path / "config.json"), "invocations": invocations}


def _config_from_checkpoint(raw: dict) -> MantisConfig:
    state = raw["model"]
    # Metadata remains browsable even when a checkpoint head shape is unsupported.
    critic_width = state["mlp_q.out.weight"].shape[0]
    if critic_width != CRITIC_LOGITS:
        raise ValueError(
            f"unsupported critic readout width {critic_width}; this build loads "
            f"the {CRITIC_LOGITS}-row return-mass critic, and a narrower "
            "readout must be converted by mantisnet.klent.graft"
        )
    if "model_config" in raw:
        return MantisConfig(**raw["model_config"])
    blocks = 1 + max(
        [int(k.split(".")[1]) for k in state if k.startswith("blocks.")], default=-1
    )
    h = state["stone_table.weight"].shape[1]
    heads, bias_bins = state["blocks.0.dist_bias"].shape
    return MantisConfig(
        h=h,
        blocks=blocks,
        heads=heads,
        ffn_factor=state["blocks.0.ffn.0.weight"].shape[0] // h,
        d_max=bias_bins - 2,
        value_queries=state["value_queries"].shape[0],
        value_bins=state["mlp_v.2.weight"].shape[0],
        policy_hidden=state["mlp_p.lin_a.weight"].shape[0],
        value_hidden=state["mlp_v.0.weight"].shape[0],
    )


class InferenceCache:
    def __init__(self, registry: RunRegistry, device: str, capacity: int = 2):
        self.registry, self.device, self.capacity = registry, device, capacity
        self.models: OrderedDict[str, MantisNet] = OrderedDict()
        self.lock = threading.RLock()

    def get(self, checkpoint: str) -> tuple[Path, MantisNet]:
        path = self.registry.resolve_checkpoint(checkpoint)
        key = str(path)
        with self.lock:
            if key in self.models:
                model = self.models.pop(key)
                self.models[key] = model
                return path, model
            raw = torch.load(path, map_location="cpu", weights_only=False)
            if raw.get("versions") != _versions():
                raise ValueError(
                    f"checkpoint versions {raw.get('versions')} != this build {_versions()}"
                )
            model = MantisNet(_config_from_checkpoint(raw)).to(self.device)
            model.load_state_dict(raw["model"])
            model.eval()
            self.models[key] = model
            while len(self.models) > self.capacity:
                _old_key, old = self.models.popitem(last=False)
                del old
                if self.device.startswith("cuda"):
                    torch.cuda.empty_cache()
            return path, model

    def parameters(self, path: Path, tau, lam) -> tuple[float, float]:
        if tau is not None and lam is not None:
            return float(tau), float(lam)
        try:
            run = path.parent
            config = json_file(run / "config.json")
            return float(config["klent"]["tau"]), float(config["klent"]["lam"])
        except (FileNotFoundError, KeyError, TypeError):
            raise ValueError(
                "a checkpoint outside a run directory requires tau and lam"
            ) from None

    def inspect(self, checkpoint: str, moves, t=None, tau=None, lam=None) -> dict:
        path, model = self.get(checkpoint)
        tau, lam = self.parameters(path, tau, lam)
        return inspect_position(
            model, moves, len(moves) if t is None else t, tau, lam, self.device
        )


class PlaySessions:
    def __init__(self, inference: InferenceCache):
        self.inference = inference
        self.sessions: dict[str, dict] = {}
        self.lock = threading.RLock()
        self.counter = 0

    def create(self, seats: list[dict], seed=0, ply_cap=512) -> dict:
        if len(seats) != 2 or any(s.get("kind") not in {
            "human", "random", "checkpoint", "sealbot"
        } for s in seats):
            raise ValueError("seats must contain two human/random/checkpoint/sealbot entries")
        with self.lock:
            self.counter += 1
            sid = f"play-{self.counter}"
            self.sessions[sid] = {
                "session_id": sid, "seats": seats, "moves": [], "seed": seed,
                "rng": np.random.default_rng(seed), "ply_cap": ply_cap,
            }
            self._bots(self.sessions[sid])
            return self.view(sid)

    def _position(self, session):
        import hexo_py
        return hexo_py.Position.replay(session["moves"])

    def view(self, sid: str) -> dict:
        if sid not in self.sessions:
            raise KeyError(f"no play session {sid}")
        session = self.sessions[sid]
        pos = self._position(session)
        seat = None if pos.is_terminal else pos.current_player
        return {
            k: v for k, v in session.items() if k != "rng"
        } | {
            "current_player": seat,
            "moves_remaining": 0 if pos.is_terminal else pos.moves_remaining,
            "terminal": pos.is_terminal,
            "winner": pos.winner if pos.is_terminal else None,
            "legal_count": 0 if pos.is_terminal else pos.legal_count,
            "legal_moves": [] if pos.is_terminal else pos.legal_moves(),
            "capped": len(session["moves"]) >= session["ply_cap"] and not pos.is_terminal,
        }

    def move(self, sid: str, move) -> dict:
        session = self.sessions.get(sid)
        if session is None:
            raise KeyError(f"no play session {sid}")
        pos = self._position(session)
        if pos.is_terminal:
            raise ValueError("the game is terminal")
        if session["seats"][pos.current_player]["kind"] != "human":
            raise ValueError(f"seat {pos.current_player} is not human")
        try:
            pos.advance(int(move[0]), int(move[1]))
        except Exception as exc:
            raise ValueError(str(exc)) from exc
        session["moves"].append((int(move[0]), int(move[1])))
        self._bots(session)
        return self.view(sid)

    def _bots(self, session) -> None:
        while True:
            pos = self._position(session)
            if pos.is_terminal or len(session["moves"]) >= session["ply_cap"]:
                return
            seat = session["seats"][pos.current_player]
            if seat["kind"] == "human":
                return
            if seat["kind"] == "random":
                move = pos.nth_legal(int(session["rng"].integers(pos.legal_count)))
            elif seat["kind"] == "checkpoint":
                read = self.inference.inspect(
                    seat["checkpoint"], session["moves"], tau=seat.get("tau"),
                    lam=seat.get("lam"),
                )
                key = {"argmax": "policy", "sample": "policy", "improved": "improved"}.get(
                    seat.get("mode", "argmax"), "policy"
                )
                weights = np.asarray([row[key] for row in read["legal"]])
                index = (
                    int(session["rng"].choice(len(weights), p=weights / weights.sum()))
                    if seat.get("mode") == "sample"
                    else int(weights.argmax())
                )
                move = read["legal"][index]["move"]
            else:
                from ..klent.sealbot import _mirror, load_sealbot
                root = Path(seat.get("root") or os.environ.get("SEALBOT_ROOT", ""))
                game_mod, bot_cls = load_sealbot(root, seat.get("variant", "current"))
                bot = bot_cls(float(seat.get("time", 0.05)))
                if seat.get("depth") is not None:
                    bot.max_depth = int(seat["depth"])
                turn = bot.get_move(_mirror(game_mod, session["moves"]))
                if not turn:
                    raise RuntimeError("SealBot returned no moves for a live position")
                move = turn[0]
            try:
                pos.advance(int(move[0]), int(move[1]))
            except Exception as exc:
                raise RuntimeError(f"bot proposed illegal move {move}: {exc}") from exc
            session["moves"].append((int(move[0]), int(move[1])))
