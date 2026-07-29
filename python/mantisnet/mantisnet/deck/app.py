"""FastAPI surface for the Shrimp Control Deck."""

from __future__ import annotations

import asyncio
import json
import math
import os
import shlex
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from ..builder import collate_prefixes
from ..klent import telemetry
from ..klent.evaluate import argmax_choose, play_match
from ..klent.run import _versions
from ..klent.opponents import _elo, _wilson
from ..klent.sealbot import record_match, sealbot_match
from ..model import MantisConfig
from .service import (
    InferenceCache,
    PlaySessions,
    RunRegistry,
    json_file,
    telemetry_connection,
)
from .state import DeckState


class LaunchRun(BaseModel):
    name: str
    iterations: int = Field(gt=0)
    games: int = Field(default=64, gt=0)
    envs: int = Field(default=256, gt=0)
    seed: int = 0
    lam_ret: float = Field(default=float(np.exp(-1 / 16)), gt=0, le=1)
    eval_every: int = Field(default=0, ge=0)
    eval_games: int = Field(default=64, ge=2, le=64)
    eval_depth: int | None = Field(default=None, gt=0)
    eval_time: float = Field(default=0.1, gt=0)
    eval_sims: int = Field(default=32, ge=0)
    checkpoint_every: int = Field(default=25, gt=0)
    init_from: str | None = None
    resume: bool = False
    sealbot: str | None = None
    device: Literal["cpu", "cuda"] | None = None


class Kill(BaseModel):
    confirm: bool = False


class Review(BaseModel):
    tags: list[str] = []
    note: str = ""


class InspectRequest(BaseModel):
    checkpoint: str
    moves: list[tuple[int, int]] = []
    t: int | None = Field(default=None, ge=0)
    tau: float | None = Field(default=None, gt=0)
    lam: float | None = Field(default=None, ge=0)


class PlayRequest(BaseModel):
    seats: list[dict[str, Any]]
    seed: int = 0
    ply_cap: int = Field(default=512, gt=0)


class MoveRequest(BaseModel):
    move: tuple[int, int]


class ProbeRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    checkpoint: str
    moves: list[tuple[int, int]] = []
    module: str = "policy"


class PresetRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    config: dict[str, Any]


class MatchRequest(BaseModel):
    checkpoint_a: str
    checkpoint_b: str | None = None
    opponent: Literal["checkpoint", "sealbot"] = "sealbot"
    games: int = Field(default=8, ge=2, le=64)
    ply_cap: int = Field(default=200, gt=0)
    seed: int = 0
    sealbot_root: str | None = None
    sealbot_variant: str = "current"
    sealbot_depth: int | None = Field(default=1, gt=0)
    sealbot_time: float = Field(default=0.05, gt=0)


def _error(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status, detail={"code": code, "message": message})


def _range(lo, hi):
    return (lo, hi) if lo is not None or hi is not None else None


def _jsonable(value):
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class MatchRunner:
    def __init__(self, app: FastAPI):
        self.app = app
        self.lock = threading.Lock()
        self.active: int | None = None

    def launch(self, request: MatchRequest) -> dict:
        if request.games % 2:
            raise ValueError("matches require an even game count for seat balance")
        with self.lock:
            if self.active is not None:
                try:
                    status = self.app.state.deck_state.match(self.active)["status"]
                except KeyError:
                    status = "failed"
                if status in {"queued", "running"}:
                    raise RuntimeError(f"match job {self.active} is already active")
            payload = request.model_dump()
            job_id = self.app.state.deck_state.new_match(payload)
            self.active = job_id
            threading.Thread(
                target=self._run, args=(job_id, request), daemon=True
            ).start()
        return {
            "job_id": job_id,
            "status": "queued",
            "training_active": self.app.state.registry.active_name() is not None,
        }

    def _run(self, job_id: int, request: MatchRequest) -> None:
        state: DeckState = self.app.state.deck_state
        state.update_match(job_id, "running")
        try:
            path_a, model_a = self.app.state.inference.get(request.checkpoint_a)
            rng = np.random.default_rng(request.seed)
            if request.opponent == "checkpoint":
                if not request.checkpoint_b:
                    raise ValueError("checkpoint opponent requires checkpoint_b")
                _path_b, model_b = self.app.state.inference.get(request.checkpoint_b)
                started = time.monotonic()
                result = play_match(
                    argmax_choose(model_a, self.app.state.inference.device),
                    argmax_choose(model_b, self.app.state.inference.device),
                    request.games, request.ply_cap, rng,
                )
                ci_lo, ci_hi = _wilson(result["score_a"], result["games"])
                result = result | {
                    "score": result["score_a"],
                    "win_rate": result["score_a"] / result["games"],
                    "ci_lo": ci_lo, "ci_hi": ci_hi,
                    "elo": _elo(result["score_a"] / result["games"]),
                    "elo_lo": _elo(ci_lo), "elo_hi": _elo(ci_hi),
                    "score_as_p0": None, "score_as_p1": None,
                    "avg_plies": None, "seconds": time.monotonic() - started,
                }
                with telemetry.open_telemetry(path_a.parent) as writer:
                    opponent_id = writer.opponent(
                        "checkpoint", {"checkpoint": request.checkpoint_b}
                    )
                    result["match_id"] = writer.write_eval_match(
                        opponent_id, result, [], source="cli",
                        checkpoint=path_a.name,
                    )
            else:
                root = request.sealbot_root or os.environ.get("SEALBOT_ROOT")
                if not root:
                    raise ValueError("SealBot match requires sealbot_root or SEALBOT_ROOT")
                result, games = sealbot_match(
                    model_a, self.app.state.inference.device, request.games,
                    request.ply_cap, rng, request.sealbot_time, Path(root),
                    variant=request.sealbot_variant, max_depth=request.sealbot_depth,
                )
                with telemetry.open_telemetry(path_a.parent) as writer:
                    # The summary's opponent identity includes the variant.
                    match_id = record_match(
                        writer, result, games, source="deck",
                        checkpoint=path_a.name,
                    )
                result["match_id"] = match_id
            state.update_match(job_id, "completed", _jsonable(result))
        except Exception as exc:
            state.update_match(job_id, "failed", error=str(exc))
        finally:
            with self.lock:
                if self.active == job_id:
                    self.active = None


def create_app(
    runs_root: Path | str | None = None,
    frontend_dist: Path | str | None = None,
    device: str | None = None,
    command_prefix: list[str] | None = None,
) -> FastAPI:
    root = Path(runs_root or os.environ.get("DECK_RUNS_ROOT", "runs")).resolve()
    dist = Path(
        frontend_dist
        or os.environ.get(
            "DECK_FRONTEND_DIST", Path(__file__).parents[4] / "frontend" / "dist"
        )
    ).resolve()
    infer_device = device or os.environ.get(
        "DECK_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
    )
    if command_prefix is None and os.environ.get("DECK_RUN_COMMAND"):
        command_prefix = shlex.split(os.environ["DECK_RUN_COMMAND"])

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.registry = RunRegistry(root, command_prefix)
        app.state.deck_state = DeckState(root / "deck.db")
        app.state.inference = InferenceCache(app.state.registry, infer_device)
        app.state.plays = PlaySessions(app.state.inference)
        app.state.matches = MatchRunner(app)
        yield

    app = FastAPI(title="Shrimp Control Deck", version="1", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request, exc):
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "validation", "message": str(exc)}},
        )

    @app.exception_handler(HTTPException)
    async def http_error(_request, exc):
        detail = exc.detail if isinstance(exc.detail, dict) else {
            "code": "http_error", "message": str(exc.detail)
        }
        return JSONResponse(status_code=exc.status_code, content={"error": detail})

    @app.get("/api/health")
    def health():
        return {"ok": True, "device": infer_device}

    @app.get("/api/runs")
    def runs(request: Request):
        return request.app.state.registry.list()

    @app.post("/api/runs", status_code=201)
    def launch(body: LaunchRun, request: Request):
        try:
            if body.resume and body.init_from:
                raise ValueError("resume and init_from are mutually exclusive")
            return request.app.state.registry.launch(
                {k: v for k, v in body.model_dump().items() if v is not None}
            )
        except RuntimeError as exc:
            raise _error(409, "run_active", str(exc)) from exc
        except FileExistsError as exc:
            raise _error(409, "run_exists", str(exc)) from exc
        except (ValueError, FileNotFoundError) as exc:
            raise _error(400, "invalid_run", str(exc)) from exc

    @app.get("/api/runs/{run}")
    def run_detail(run: str, request: Request):
        try:
            return request.app.state.registry.describe(run)
        except (ValueError, FileNotFoundError) as exc:
            raise _error(404, "run_not_found", str(exc)) from exc

    @app.get("/api/runs/{run}/manifest")
    def manifest(run: str, request: Request):
        try:
            return request.app.state.registry.manifest(run)
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            raise _error(404, "run_artifact", str(exc)) from exc

    @app.get("/api/runs/{run}/checkpoints")
    def checkpoints(run: str, request: Request):
        try:
            return request.app.state.registry.checkpoints(run)
        except (ValueError, FileNotFoundError) as exc:
            raise _error(404, "run_not_found", str(exc)) from exc

    @app.post("/api/runs/{run}/stop", status_code=202)
    def stop(run: str, request: Request):
        try:
            return request.app.state.registry.sentinel(run, "STOP")
        except (ValueError, FileNotFoundError) as exc:
            raise _error(404, "run_not_found", str(exc)) from exc

    @app.post("/api/runs/{run}/checkpoint", status_code=202)
    def checkpoint(run: str, request: Request):
        try:
            return request.app.state.registry.sentinel(run, "CHECKPOINT")
        except (ValueError, FileNotFoundError) as exc:
            raise _error(404, "run_not_found", str(exc)) from exc

    @app.post("/api/runs/{run}/kill", status_code=202)
    def kill(run: str, body: Kill, request: Request):
        try:
            return request.app.state.registry.kill(run, body.confirm)
        except ValueError as exc:
            raise _error(400, "confirmation_required", str(exc)) from exc
        except (FileNotFoundError, RuntimeError) as exc:
            raise _error(409, "cannot_kill", str(exc)) from exc

    def conn_for(request: Request, run: str):
        try:
            return telemetry_connection(request.app.state.registry.path(run))
        except (ValueError, FileNotFoundError) as exc:
            raise _error(404, "telemetry_not_found", str(exc)) from exc
        except RuntimeError as exc:
            raise _error(500, "telemetry_error", str(exc)) from exc

    @app.get("/api/runs/{run}/summary")
    def summary(run: str, request: Request):
        with conn_for(request, run) as conn:
            return telemetry.summary(conn)

    @app.get("/api/runs/{run}/iterations")
    def iterations(
        run: str, request: Request, columns: str = Query(...),
        from_iteration: int | None = None, to_iteration: int | None = None,
    ):
        try:
            with conn_for(request, run) as conn:
                return telemetry.iteration_series(
                    conn, [c.strip() for c in columns.split(",") if c.strip()],
                    iterations=_range(from_iteration, to_iteration),
                )
        except ValueError as exc:
            raise _error(400, "invalid_query", str(exc)) from exc

    @app.get("/api/runs/{run}/games")
    def games(
        run: str, request: Request, kind: str | None = None,
        winner: int | None = None, capped: bool | None = None,
        min_length: int | None = None, max_length: int | None = None,
        from_iteration: int | None = None, to_iteration: int | None = None,
        order: str = "recent", limit: int = Query(100, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ):
        try:
            with conn_for(request, run) as conn:
                rows = telemetry.search_games(
                    conn, kind=kind, winner=winner, capped=capped,
                    min_length=min_length, max_length=max_length,
                    iterations=_range(from_iteration, to_iteration), order=order,
                    limit=limit, offset=offset,
                )
                for row in rows:
                    full = conn.execute(
                        "SELECT moves FROM games WHERE game_id=?", (row["game_id"],)
                    ).fetchone()
                    row["opening"] = telemetry.canonical_opening(
                        telemetry.unpack_moves(full[0]), min(4, row["length"])
                    )
                return rows
        except ValueError as exc:
            raise _error(400, "invalid_query", str(exc)) from exc

    @app.get("/api/runs/{run}/games/{game_id}")
    def game(run: str, game_id: int, request: Request):
        try:
            with conn_for(request, run) as conn:
                value = telemetry.fetch_game(conn, game_id)
            value["review"] = request.app.state.deck_state.review(run, game_id)
            return value
        except KeyError as exc:
            raise _error(404, "game_not_found", str(exc)) from exc

    @app.put("/api/runs/{run}/games/{game_id}/review")
    def review(run: str, game_id: int, body: Review, request: Request):
        try:
            with conn_for(request, run) as conn:
                telemetry.fetch_game(conn, game_id)
            return request.app.state.deck_state.put_review(
                run, game_id, body.tags, body.note
            )
        except KeyError as exc:
            raise _error(404, "game_not_found", str(exc)) from exc
        except ValueError as exc:
            raise _error(400, "invalid_review", str(exc)) from exc

    @app.get("/api/runs/{run}/calibration")
    def calibration(
        run: str, request: Request, by: str = "v_hat", bucket: float = 0.1,
        from_iteration: int | None = None, to_iteration: int | None = None,
    ):
        try:
            with conn_for(request, run) as conn:
                return telemetry.calibration(
                    conn, by=by, bucket=bucket,
                    iterations=_range(from_iteration, to_iteration),
                )
        except ValueError as exc:
            raise _error(400, "invalid_query", str(exc)) from exc

    @app.get("/api/runs/{run}/blunders")
    def blunders(
        run: str, request: Request, threshold: float = 0.5,
        from_iteration: int | None = None, to_iteration: int | None = None,
        limit: int = Query(50, ge=1, le=500),
    ):
        with conn_for(request, run) as conn:
            return telemetry.blunders(
                conn, threshold=threshold,
                iterations=_range(from_iteration, to_iteration), limit=limit,
            )

    @app.get("/api/runs/{run}/openings")
    def openings(
        run: str, request: Request, plies: int = Query(4, ge=1, le=32),
        kind: str = "selfplay", from_iteration: int | None = None,
        to_iteration: int | None = None, limit: int = Query(50, ge=1, le=500),
    ):
        with conn_for(request, run) as conn:
            return telemetry.opening_atlas(
                conn, plies=plies, kind=kind,
                iterations=_range(from_iteration, to_iteration), limit=limit,
            )

    @app.get("/api/runs/{run}/strength")
    def strength(run: str, request: Request, opponent_id: int | None = None):
        with conn_for(request, run) as conn:
            return _jsonable(telemetry.strength_curve(conn, opponent_id=opponent_id))

    @app.get("/api/runs/{run}/crossplay")
    def crossplay(run: str, request: Request):
        with conn_for(request, run) as conn:
            return telemetry.crossplay_matrix(conn)

    @app.get("/api/runs/{run}/events")
    async def events(run: str, request: Request, once: bool = False):
        try:
            run_dir = request.app.state.registry.path(run)
        except (ValueError, FileNotFoundError) as exc:
            raise _error(404, "run_not_found", str(exc)) from exc

        async def stream():
            last_iteration, last_status, log_offset = -1, 0.0, 0
            seen_checkpoints: set[str] = set()
            previous_state = None
            while not await request.is_disconnected():
                try:
                    described = request.app.state.registry.describe(run)
                    if described["state"] != previous_state:
                        previous_state = described["state"]
                        yield _sse("lifecycle", {"state": previous_state})
                    status_path = run_dir / "status.json"
                    if status_path.exists() and status_path.stat().st_mtime > last_status:
                        last_status = status_path.stat().st_mtime
                        yield _sse("heartbeat", json_file(status_path))
                    if (run_dir / telemetry.DB_NAME).exists():
                        with telemetry_connection(run_dir) as conn:
                            row = conn.execute(
                                "SELECT MAX(iteration) FROM iterations"
                            ).fetchone()
                            current = -1 if row[0] is None else int(row[0])
                            if current > last_iteration:
                                for iteration in range(last_iteration + 1, current + 1):
                                    yield _sse("iteration", {"iteration": iteration})
                                    eval_row = conn.execute(
                                        "SELECT * FROM eval_matches WHERE iteration=? "
                                        "ORDER BY match_id DESC LIMIT 1", (iteration + 1,)
                                    ).fetchone()
                                    if eval_row:
                                        yield _sse("eval", dict(eval_row))
                                last_iteration = current
                    checkpoint_names = {p.name for p in run_dir.glob("*.pt")}
                    for name in sorted(checkpoint_names - seen_checkpoints):
                        yield _sse("checkpoint", {"name": name})
                    seen_checkpoints = checkpoint_names
                    log = run_dir / "deck-console.log"
                    if log.exists():
                        with log.open("r", encoding="utf-8", errors="replace") as handle:
                            handle.seek(log_offset)
                            for line in handle:
                                line = line.rstrip()
                                kind = "eval" if " | eval " in line else "log"
                                yield _sse(kind, {"message": line})
                            log_offset = handle.tell()
                except Exception as exc:
                    yield _sse("lifecycle", {"state": "error", "message": str(exc)})
                if once:
                    return
                await asyncio.sleep(1)

        return StreamingResponse(
            stream(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/inspect")
    def inspect(body: InspectRequest, request: Request):
        try:
            return request.app.state.inference.inspect(**body.model_dump())
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            raise _error(400, "inspect_failed", str(exc)) from exc

    @app.post("/api/inspect/d6")
    def d6(body: InspectRequest, request: Request):
        try:
            base = request.app.state.inference.inspect(**body.model_dump())
            results = []
            for index, transform in enumerate(telemetry.D6_TRANSFORMS):
                transformed = [transform(move) for move in body.moves]
                read = request.app.state.inference.inspect(
                    body.checkpoint, transformed, body.t, body.tau, body.lam
                )
                mapped = {tuple(row["move"]): row for row in read["legal"]}
                policy_dev = max(
                    abs(row["policy"] - mapped[transform(tuple(row["move"]))]["policy"])
                    for row in base["legal"]
                )
                q_dev = max(
                    abs(row["q"] - mapped[transform(tuple(row["move"]))]["q"])
                    for row in base["legal"]
                )
                results.append(
                    {"transform": index, "policy_max": policy_dev, "q_max": q_dev}
                )
            return {
                "transforms": results,
                "policy_max": max(row["policy_max"] for row in results),
                "q_max": max(row["q_max"] for row in results),
            }
        except (ValueError, FileNotFoundError, RuntimeError, KeyError) as exc:
            raise _error(400, "d6_failed", str(exc)) from exc

    @app.post("/api/inspect/attention")
    def attention(body: InspectRequest, request: Request):
        try:
            _path, model = request.app.state.inference.get(body.checkpoint)
            t = len(body.moves) if body.t is None else body.t
            batch = collate_prefixes([body.moves], [t]).to(
                request.app.state.inference.device
            )
            captures: list[dict[str, torch.Tensor]] = [
                {} for _ in range(model.cfg.blocks)
            ]
            hooks = []
            for index, block in enumerate(model.blocks):
                hooks.append(block.wq.register_forward_hook(
                    lambda _m, _i, out, index=index: captures[index].__setitem__("q", out.detach())
                ))
                hooks.append(block.wk.register_forward_hook(
                    lambda _m, _i, out, index=index: captures[index].__setitem__("k", out.detach())
                ))
            with torch.no_grad():
                model.trunk(batch)
            for hook in hooks:
                hook.remove()
            length = int(batch.attn_valid[0].sum())
            coords = batch.coords[0, :length].cpu()
            layers = []
            for index, (block, captured) in enumerate(zip(model.blocks, captures)):
                h, heads = model.cfg.h, model.cfg.heads
                dim = h // heads
                q = captured["q"][0, :length].view(length, heads, dim).transpose(0, 1)
                k = captured["k"][0, :length].view(length, heads, dim).transpose(0, 1)
                logits = q.float() @ k.float().transpose(-1, -2) / math.sqrt(dim)
                dq = coords[:, None, 0] - coords[None, :, 0]
                dr = coords[:, None, 1] - coords[None, :, 1]
                distance = torch.maximum(torch.maximum(dq.abs(), dr.abs()), (dq + dr).abs())
                buckets = (distance - 1).clamp(0, model.cfg.d_max - 1).long()
                buckets[distance == 0] = model.cfg.self_bucket
                buckets[0, :] = model.cfg.token_bucket
                buckets[:, 0] = model.cfg.token_bucket
                logits += block.dist_bias.float()[:, buckets]
                layers.append({
                    "block": index,
                    "heads": torch.softmax(logits, dim=-1).cpu().tolist(),
                })
            return {"tokens": length, "layers": layers}
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            raise _error(400, "attention_failed", str(exc)) from exc

    @app.get("/api/model")
    def model_manifest():
        return {"config": MantisConfig().__dict__, "versions": _versions()}

    @app.post("/api/play", status_code=201)
    def play(body: PlayRequest, request: Request):
        try:
            return request.app.state.plays.create(
                body.seats, body.seed, body.ply_cap
            )
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            raise _error(400, "play_failed", str(exc)) from exc

    @app.get("/api/play/{session_id}")
    def play_get(session_id: str, request: Request):
        try:
            return request.app.state.plays.view(session_id)
        except KeyError as exc:
            raise _error(404, "session_not_found", str(exc)) from exc

    @app.post("/api/play/{session_id}/moves")
    def play_move(session_id: str, body: MoveRequest, request: Request):
        try:
            return request.app.state.plays.move(session_id, body.move)
        except KeyError as exc:
            raise _error(404, "session_not_found", str(exc)) from exc
        except ValueError as exc:
            raise _error(400, "illegal_move", str(exc)) from exc

    @app.get("/api/play/{session_id}/inspect")
    def play_inspect(session_id: str, checkpoint: str, request: Request):
        try:
            session = request.app.state.plays.view(session_id)
            return request.app.state.inference.inspect(checkpoint, session["moves"])
        except KeyError as exc:
            raise _error(404, "session_not_found", str(exc)) from exc
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            raise _error(400, "inspect_failed", str(exc)) from exc

    @app.get("/api/probes")
    def probes(request: Request):
        return request.app.state.deck_state.probes()

    @app.post("/api/probes", status_code=201)
    def save_probe(body: ProbeRequest, request: Request):
        try:
            request.app.state.registry.resolve_checkpoint(body.checkpoint)
            return request.app.state.deck_state.add_probe(**body.model_dump())
        except (ValueError, FileNotFoundError) as exc:
            raise _error(400, "invalid_probe", str(exc)) from exc

    @app.delete("/api/probes/{probe_id}", status_code=204)
    def delete_probe(probe_id: int, request: Request):
        try:
            request.app.state.deck_state.delete_probe(probe_id)
        except KeyError as exc:
            raise _error(404, "probe_not_found", str(exc)) from exc

    @app.get("/api/presets")
    def presets(request: Request):
        return request.app.state.deck_state.presets()

    @app.put("/api/presets/{name}")
    def save_preset(name: str, body: PresetRequest, request: Request):
        if name != body.name:
            raise _error(400, "invalid_preset", "path and body names differ")
        return request.app.state.deck_state.put_preset(name, body.config)

    @app.get("/api/matches")
    def matches(request: Request):
        return request.app.state.deck_state.matches()

    @app.post("/api/matches", status_code=202)
    def match(body: MatchRequest, request: Request):
        try:
            return request.app.state.matches.launch(body)
        except RuntimeError as exc:
            raise _error(409, "match_active", str(exc)) from exc
        except ValueError as exc:
            raise _error(400, "invalid_match", str(exc)) from exc

    @app.get("/api/matches/{job_id}")
    def match_get(job_id: int, request: Request):
        try:
            return request.app.state.deck_state.match(job_id)
        except KeyError as exc:
            raise _error(404, "match_not_found", str(exc)) from exc

    @app.get("/api/matches/{job_id}/events")
    async def match_events(job_id: int, request: Request):
        try:
            request.app.state.deck_state.match(job_id)
        except KeyError as exc:
            raise _error(404, "match_not_found", str(exc)) from exc

        async def stream():
            previous = None
            while not await request.is_disconnected():
                job = request.app.state.deck_state.match(job_id)
                if job != previous:
                    previous = job
                    yield _sse("match", job)
                if job["status"] in {"completed", "failed"}:
                    return
                await asyncio.sleep(0.5)

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        if not dist.is_dir() or not (dist / "index.html").is_file():
            raise _error(
                503, "frontend_missing",
                f"frontend build is missing at {dist}; run the frontend-build service",
            )
        candidate = (dist / path).resolve()
        if path and candidate.is_file() and dist in candidate.parents:
            return FileResponse(candidate)
        return FileResponse(dist / "index.html")

    return app


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(_jsonable(data), separators=(',', ':'))}\n\n"


app = create_app()
