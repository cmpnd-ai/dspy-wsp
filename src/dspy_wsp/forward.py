"""The forward-pass tool: the only way the writer runs the model.

Two kinds of pass, each costing one pass whatever its size (paper, Experiment 8):

* generate: run the DSPy program on practice examples with the original weights,
  the edited weights, or both (paired, same seeds), and score every answer with
  the metric.
* teacher_forced: read texts without generating and return log-probs, top-k,
  hidden states, MLP activations or attention probabilities.

A request that fails is refused before it runs and does not count.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dspy_wsp import stats
from dspy_wsp.edits import EditError, EditStore, plain_file

logger = logging.getLogger(__name__)


@dataclass
class Limits:
    """Per-pass limits, defaulting to the paper's Experiment 8."""

    max_generations: int = 512
    max_generations_both: int = 256  # per model
    max_tokens: int | None = None  # cap on a request's per-call max_tokens (None: no cap)
    max_texts: int = 1024
    max_text_tokens: int = 32_768
    max_total_tokens: int = 524_288
    max_attention_texts: int = 2
    max_attention_tokens: int = 2_048
    max_topk: int = 50
    max_pass_bytes: int = 6 * 2**30
    pooled_answer_tokens: int = 8192
    max_reads: int = 1024  # decision models: question rows per read pass


class RequestError(ValueError):
    """A forward request refused before it ran; it does not count against the budget."""


def _jsonable(v):
    try:
        json.dumps(v)
        return v
    except TypeError:
        return str(v)


def copy_program(program, lm=None):
    """Deep-copy a DSPy program without deep-copying its LMs.

    A backend LM holds the backend (and its model) through its engine; a deep copy
    would clone the model and edits would never reach the copy.
    """
    preds = [p for _, p in program.named_predictors()]
    saved = [p.lm for p in preds]
    for p in preds:
        p.lm = None
    try:
        new = program.deepcopy()
    finally:
        for p, original in zip(preds, saved):
            p.lm = original
    if lm is not None:
        for _, p in new.named_predictors():
            p.lm = lm
    return new


def item_seed(base: int, item_id: str, sample: int) -> int:
    h = hashlib.sha256(f"{base}:{item_id}:{sample}".encode()).digest()
    return int.from_bytes(h[:8], "big") % (2**31 - 1)


def _aggregate(summary: dict) -> dict:
    """A summary without fields that only describe text generation."""
    return {k: v for k, v in summary.items() if k not in ("mean_output_tokens", "cut_share")}


def _int_list(v, name: str, lo: int, hi: int | None) -> list[int]:
    if v is None:
        return []
    if not isinstance(v, list) or not all(isinstance(x, int) for x in v):
        raise RequestError(f'"{name}" must be a list of integers')
    bad = [x for x in v if x < lo or (hi is not None and x > hi)]
    if bad:
        raise RequestError(f'"{name}" has out-of-range values {bad}; allowed {lo}..{hi}')
    return v


METRIC_FAILED = "metric failed: "


class ForwardTool:
    def __init__(
        self,
        *,
        backend,
        program,
        metric: Callable,
        practice: Mapping,  # id -> dspy.Example
        edit_store: EditStore,
        work_dir: str | Path,
        limits: Limits | None = None,
        seed: int = 0,
        validation: Mapping | None = None,  # id -> dspy.Example; decision runs only, aggregate results only
    ):
        self.backend = backend
        self.program = program
        self.metric = metric
        self.practice = dict(practice)
        self.edit_store = edit_store
        self.work_dir = Path(work_dir)
        self.limits = limits or Limits()
        self.seed = seed
        first = next(iter(self.practice.values()), None)
        self.input_keys = sorted(first.inputs().keys()) if first is not None else []
        self._metric_arity = self._arity(metric)
        self.layers = backend.dims().get("layers")
        self.decision = backend.kind == "decision"
        self.validation = dict(validation or {})
        self.validation_log: list[dict] = []
        self.allow_validation = True  # a population search turns this off: it scores candidates on validation itself

    # --- entry point ----------------------------------------------------------------
    def handle(self, body: str, k: int) -> tuple[str, bool]:
        """Run one request. Returns (message for the writer, whether the pass counted)."""
        try:
            req = json.loads(body)
            if not isinstance(req, dict):
                raise RequestError("the forward block must contain one JSON object")
            if set(req) == {"file"}:
                req = self._request_from_file(req["file"])
        except json.JSONDecodeError as e:
            return f"FORWARD REQUEST FAILED (not counted): invalid JSON: {e}", False
        except RequestError as e:
            return f"FORWARD REQUEST FAILED (not counted): {e}", False
        t0 = time.time()
        try:
            kind = req.get("type")
            if self.decision:
                if kind == "decide":
                    summary = self.decide(req, k)
                elif kind == "read":
                    summary = self.read(req, k)
                else:
                    raise RequestError('this is a decision model: "type" must be "decide" or "read"')
            elif kind == "generate":
                summary = self.generate(req, k)
            elif kind == "teacher_forced":
                summary = self.teacher_forced(req, k)
            else:
                raise RequestError('"type" must be "generate" or "teacher_forced"')
        except (RequestError, EditError) as e:
            return f"FORWARD REQUEST FAILED (not counted): {e}", False
        except Exception as e:  # a backend failure is not the writer's fault
            logger.exception("forward pass %s failed", k)
            return f"FORWARD REQUEST FAILED (not counted): {type(e).__name__}: {e}", False
        summary["seconds"] = round(time.time() - t0, 1)
        return f"RESULT of forward pass {k}:\n```\n{json.dumps(summary, indent=1)}\n```", True

    def _request_from_file(self, name) -> dict:
        """A request the writer saved as JSON in its work folder, so long texts can be built in code."""
        if not isinstance(name, str):
            raise RequestError('"file" must be a path in the work folder')
        work = self.work_dir.resolve()
        path = (work / name).resolve()
        if not path.is_relative_to(work):
            raise RequestError(f'"file" must be inside {work}')
        if not path.is_file():
            raise RequestError(f"no such file: {path}")
        req = json.loads(path.read_text())
        if not isinstance(req, dict) or "file" in req:
            raise RequestError("the request file must contain one JSON request object")
        return req

    # --- generation -----------------------------------------------------------------
    def generate(self, req: dict, k: int) -> dict:
        model = req.get("model", "edited")
        if model not in ("original", "edited", "both"):
            raise RequestError('"model" must be "original", "edited" or "both"')
        ids = req.get("problems") or []
        custom = req.get("inputs") or []
        if not isinstance(ids, list) or not isinstance(custom, list):
            raise RequestError('"problems" and "inputs" must be lists')
        unknown = [i for i in ids if i not in self.practice]
        if unknown:
            raise RequestError(f"unknown practice ids: {unknown[:5]}{' ...' if len(unknown) > 5 else ''}")
        for c in custom:
            if not isinstance(c, dict) or set(c) != set(self.input_keys):
                raise RequestError(f'each entry of "inputs" must be an object with exactly the input fields {self.input_keys}')
        n = req.get("n", 1)
        if not isinstance(n, int) or n < 1:
            raise RequestError('"n" must be a positive integer')
        items = [(i, self.practice[i]) for i in ids] + [(f"input_{j}", c) for j, c in enumerate(custom)]
        total = len(items) * n
        if total == 0:
            raise RequestError('give "problems" or "inputs"')
        cap = self.limits.max_generations_both if model == "both" else self.limits.max_generations
        if total > cap:
            raise RequestError(f"{total} generations requested; at most {cap}" + (" per model with \"both\"" if model == "both" else ""))
        max_tokens = req.get("max_tokens")
        if max_tokens is not None:
            if not isinstance(max_tokens, int) or max_tokens < 1:
                raise RequestError('"max_tokens" must be a positive integer')
            if self.limits.max_tokens and max_tokens > self.limits.max_tokens:
                raise RequestError(f'"max_tokens" is at most {self.limits.max_tokens}')
        acts = req.get("activations")
        hidden_layers = mlp_layers = []
        if acts is not None:
            if not isinstance(acts, dict):
                raise RequestError('"activations" must be an object')
            L = self.layers
            hidden_layers = _int_list(acts.get("layers"), "activations.layers", 0, L)
            mlp_layers = _int_list(acts.get("mlp_layers"), "activations.mlp_layers", 0, L - 1 if L else None)

        edits = self.edit_store.load_all() if model in ("edited", "both") else None
        program = self._program(max_tokens)
        summary: dict = {"type": "generate", "model": model, "generations": total}
        if model in ("edited", "both") and not edits:
            summary["note"] = "no edits are saved, so the edited model equals the original"
        runs = ["original", "edited"] if model == "both" else [model]
        results = {}
        for which in runs:
            with self.backend.using(edits if which == "edited" else None):
                rows = self.run_items(program, items, n, which)
                if hidden_layers or mlp_layers:
                    folder = f"gen_{k}_acts" + (f"_{which}" if model == "both" else "")
                    summary.setdefault("activations", {})[which] = self._pooled(rows, hidden_layers, mlp_layers, self.work_dir / folder)
            name = f"gen_{k}.jsonl" if model != "both" else f"gen_{k}_{which}.jsonl"
            self._write_rows(self.work_dir / name, rows)
            summary.setdefault("files", []).append(str(self.work_dir / name))
            results[which] = rows
        if model == "both":
            summary["original"] = self.summarize(results["original"])
            summary["edited"] = self.summarize(results["edited"])
            summary["paired"] = stats.paired(self._scores(results["original"]), self._scores(results["edited"]))
        else:
            summary.update(self.summarize(results[model]))
        return summary

    # --- decision models ------------------------------------------------------------
    def decide(self, req: dict, k: int) -> dict:
        """Run the program on practice items (as `generate`), or on the whole validation set with aggregate results only."""
        extra = {"max_tokens", "activations"} & set(req)
        if extra:
            raise RequestError(f"decide takes no {sorted(extra)}")
        if not req.get("validation"):
            summary = self.generate(req, k)
            summary["type"] = "decide"
            return summary
        if not self.allow_validation:
            raise RequestError("validation passes are not available in this run: the search scores your edit on the validation set after the round")
        if not self.validation:
            raise RequestError("this run has no validation set")
        if req.get("problems") or req.get("inputs") or req.get("n", 1) != 1:
            raise RequestError('a validation run takes no "problems", "inputs" or "n": it runs the whole validation set once')
        model = req.get("model", "edited")
        if model not in ("original", "edited", "both"):
            raise RequestError('"model" must be "original", "edited" or "both"')
        edits = self.edit_store.load_all() if model in ("edited", "both") else None
        program = self._program(None)
        items = list(self.validation.items())
        runs = ["original", "edited"] if model == "both" else [model]
        results = {}
        for which in runs:
            with self.backend.using(edits if which == "edited" else None):
                results[which] = self.run_items(program, items, 1, which)
        summary: dict = {"type": "decide", "validation": True, "model": model, "items": len(items),
                         "note": "validation runs report aggregate scores only"}
        if model == "both":
            summary["original"] = _aggregate(self.summarize(results["original"]))
            summary["edited"] = _aggregate(self.summarize(results["edited"]))
            summary["paired"] = stats.paired(self._scores(results["original"]), self._scores(results["edited"]))
        else:
            summary.update(_aggregate(self.summarize(results[model])))
        if model in ("edited", "both") and not edits:
            summary["note"] += "; no edits are saved, so the edited model equals the original"
        self.validation_log.append({"pass": k, "model": model, "edited_tensors": sorted(edits or {}),
                                    **{key: summary[key] for key in ("score", "original", "edited", "paired") if key in summary}})
        return summary

    def read(self, req: dict, k: int) -> dict:
        """What the decision model computes for given System One requests: logits, probabilities and head inputs."""
        model = req.get("model", "original")
        if model not in ("original", "edited"):
            raise RequestError('"model" must be "original" or "edited"')
        if ("requests" in req) == ("from" in req):
            raise RequestError('give exactly one of "requests" or "from"')
        requests = req["requests"] if "requests" in req else self._recorded_requests(req["from"])
        if not isinstance(requests, list) or not requests:
            raise RequestError('"requests" must be a non-empty list of {"state", "questions"} objects')
        for r in requests:
            if not isinstance(r, dict) or not isinstance(r.get("questions"), dict) or not r["questions"]:
                raise RequestError('each request must be an object with "state" and a non-empty "questions" object')
        n_rows = sum(len(r["questions"]) for r in requests)
        if n_rows > self.limits.max_reads:
            raise RequestError(f"{n_rows} questions; at most {self.limits.max_reads} per read pass")
        layers = _int_list(req.get("layers"), "layers", 0, self.layers)
        edits = self.edit_store.load_all() if model == "edited" else None
        with self.backend.using(edits):
            arrays, meta = self.backend.read(requests, layers)
        folder = self.work_dir / f"fwd_{k}"
        folder.mkdir(parents=True, exist_ok=True)
        files = {}
        for name, a in arrays.items():
            np.save(folder / f"{name}.npy", a)
            files[f"{name}.npy"] = list(a.shape)
        meta.update(model=model, files=files)
        (folder / "meta.json").write_text(json.dumps(meta, indent=1))
        (folder / "requests.json").write_text(json.dumps(requests, default=str))
        return {"type": "read", "model": model, "requests": len(requests), "questions": n_rows, "dir": str(folder), "files": files}

    def _recorded_requests(self, src) -> list:
        """The System One requests recorded for practice ids in an earlier decide pass."""
        if not isinstance(src, dict) or not isinstance(src.get("pass"), int):
            raise RequestError('"from" must be {"pass": <k>, "ids": [...], "model": "original" or "edited" (for a "both" pass)}')
        k, which = src["pass"], src.get("model")
        path = self.work_dir / (f"gen_{k}_{which}.jsonl" if which else f"gen_{k}.jsonl")
        if not plain_file(path):
            raise RequestError(f"no records for pass {k}" + ("" if which else ' (a "both" pass needs "model")') + "; validation runs keep no records")
        wanted = src.get("ids")
        rows = [json.loads(line) for line in path.open()]
        if wanted is not None:
            if not isinstance(wanted, list):
                raise RequestError('"ids" must be a list')
            rows = [r for r in rows if r["id"] in set(wanted)]
        return [json.loads(c["prompt"]) for r in rows for c in r.get("calls", [])]

    def _program(self, max_tokens: int | None):
        lm = self.backend.lm()
        if max_tokens is not None:
            lm = lm.copy(max_tokens=max_tokens)
        return copy_program(self.program, lm)

    def run_items(self, program, items, n: int, which: str) -> list[dict]:
        """Run the program on every (item, sample). Rows come back in request order.

        Runs go `backend.concurrency` at a time, so a batching backend can serve
        their LM calls together. Each run's seed depends only on its item and sample.
        """
        jobs = [(item_id, ex, j) for item_id, ex in items for j in range(n)]
        workers = max(1, int(self.backend.concurrency or 1))
        if workers == 1 or len(jobs) <= 1:
            return [self._run_one(program, *job, which) for job in jobs]
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(workers, len(jobs)), thread_name_prefix="wsp-gen") as pool:
            return list(pool.map(lambda job: self._run_one(program, *job, which), jobs))

    def _run_one(self, program, item_id, ex, j: int, which: str) -> dict:
        graded = not isinstance(ex, dict)
        inputs = ex.inputs().toDict() if graded else dict(ex)
        self.backend.seed(item_seed(self.seed, item_id, j))
        row = {"id": item_id, "sample": j, "model": which, "inputs": {k: _jsonable(v) for k, v in inputs.items()}}
        with self.backend.record_calls() as calls:
            try:
                pred = program(**inputs)
                row["outputs"] = {k: _jsonable(v) for k, v in pred.items()}
            except Exception as e:
                pred = None
                row["error"] = f"{type(e).__name__}: {str(e)[:500]}"
        if graded:
            row["score"], feedback = (0.0, None) if pred is None else self._score(ex, pred)
            if feedback:
                row["feedback"] = feedback
            if feedback and feedback.startswith(METRIC_FAILED):
                row["metric_error"] = True
        row["calls"] = [dataclasses.asdict(c) for c in calls]
        row["cut"] = any(c.finish_reason == "length" for c in calls)
        row["output_tokens"] = sum(c.output_tokens for c in calls)
        return row

    @staticmethod
    def _arity(metric) -> int:
        try:
            params = inspect.signature(metric).parameters.values()
        except (TypeError, ValueError):
            return 3
        if any(p.kind == p.VAR_POSITIONAL for p in params):
            return 3
        return sum(p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) for p in params)

    def _score(self, example, pred) -> tuple[float, str | None]:
        try:
            r = self.metric(example, pred, None) if self._metric_arity >= 3 else self.metric(example, pred)
            if isinstance(r, Mapping) and "score" in r:
                return float(r["score"]), r.get("feedback")
            if not isinstance(r, (bool, int, float)) and hasattr(r, "score"):  # GEPA-style dspy.Prediction(score=, feedback=)
                return float(r.score), getattr(r, "feedback", None)
            return float(r), None
        except Exception as e:
            return 0.0, f"{METRIC_FAILED}{type(e).__name__}: {e}"

    @staticmethod
    def _scores(rows) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        for r in rows:
            if "score" in r:
                out.setdefault(r["id"], []).append(r["score"])
        return out

    @staticmethod
    def summarize(rows) -> dict:
        graded = [r for r in rows if "score" in r]
        per_item = {i: stats.mean(v) for i, v in ForwardTool._scores(rows).items()}
        cut = sum(r["cut"] for r in rows)
        errors = sum(("error" in r) and not r["cut"] for r in rows)
        metric_errors = sum(bool(r.get("metric_error")) for r in graded)
        imperfect = sum(r.get("score", 1.0) < 1.0 and not r["cut"] and "error" not in r and not r.get("metric_error") for r in graded)
        return {
            "graded": len(graded),
            "score": stats.mean([r["score"] for r in graded]),
            "score_se_over_items": stats.standard_error(list(per_item.values())),
            "mean_output_tokens": stats.mean([r["output_tokens"] for r in rows]),
            "cut_share": cut / len(rows) if rows else float("nan"),
            "failures": {"cut_at_max_tokens": cut, "program_error": errors, "metric_error": metric_errors, "scored_below_1": imperfect},
        }

    def _pooled(self, rows, hidden_layers, mlp_layers, folder: Path) -> dict:
        pairs, index = [], []
        for r in rows:
            for c_i, call in enumerate(r["calls"]):
                pairs.append((call["prompt"], call["completion"]))
                index.append({"id": r["id"], "sample": r["sample"], "call": c_i, "score": r.get("score")})
        arrays = self.backend.pooled_activations(pairs, layers=hidden_layers, mlp_layers=mlp_layers, max_answer_tokens=self.limits.pooled_answer_tokens)
        folder.mkdir(parents=True, exist_ok=True)
        files = {}
        for name, a in arrays.items():
            np.save(folder / f"{name}.npy", a)
            files[f"{name}.npy"] = list(a.shape)
        (folder / "index.json").write_text(json.dumps({"rows": index, "hidden_layers": hidden_layers, "mlp_layers": mlp_layers, "files": files}))
        return {"dir": str(folder), "files": files}

    @staticmethod
    def _write_rows(path: Path, rows) -> None:
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    @staticmethod
    def _saved_rows(path: Path, items, n: int) -> list[dict] | None:
        """Rows saved at `path` if they cover exactly these items and samples with the same inputs."""
        if not path.exists():
            return None
        try:
            rows = [json.loads(line) for line in path.open()]
        except (OSError, json.JSONDecodeError):
            return None
        expected = [(item_id, dict(ex.inputs()), j) for item_id, ex in items for j in range(n)]
        if [(r.get("id"), r.get("inputs"), r.get("sample")) for r in rows] != expected:
            return None
        logger.info("reusing %d saved rows from %s", len(rows), path)
        return rows

    def paired_eval(self, examples, edits, *, n: int = 1, out_dir: str | Path) -> dict:
        """Host-only paired evaluation (never shown to the writer, never counted)."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        items = [(f"val_{i}", ex) for i, ex in enumerate(examples)]
        program = self._program(None)
        results = {}
        for which in ("original", "edited"):
            path = out / f"{which}.jsonl"
            # The original weights never change, so a rerun reuses saved original rows for the same items.
            rows = self._saved_rows(path, items, n) if which == "original" else None
            if rows is None:
                with self.backend.using(edits if which == "edited" else None):
                    rows = self.run_items(program, items, n, which)
                self._write_rows(path, rows)
            results[which] = rows
        return {
            "original": self.summarize(results["original"]),
            "edited": self.summarize(results["edited"]),
            "paired": stats.paired(self._scores(results["original"]), self._scores(results["edited"])),
        }

    # --- teacher forcing ------------------------------------------------------------
    def teacher_forced(self, req: dict, k: int) -> dict:
        lim = self.limits
        model = req.get("model", "edited")
        if model not in ("original", "edited"):
            raise RequestError('"model" must be "original" or "edited" for teacher_forced')
        texts, chats = req.get("texts"), req.get("chats")
        if (texts is None) == (chats is None):
            raise RequestError('give exactly one of "texts" or "chats"')
        if chats is not None:
            if not isinstance(chats, list) or not all(isinstance(c, list) for c in chats):
                raise RequestError('"chats" must be a list of message lists')
            kw = {"enable_thinking": req["enable_thinking"]} if "enable_thinking" in req else {}
            texts = [self.backend.render_chat(c, add_generation_prompt=False, **kw) for c in chats]
        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts) or not texts:
            raise RequestError('"texts" must be a non-empty list of strings')
        if len(texts) > lim.max_texts:
            raise RequestError(f"{len(texts)} texts; at most {lim.max_texts}")
        ret = req.get("return") or {}
        if not isinstance(ret, dict):
            raise RequestError('"return" must be an object')
        L = self.layers
        logprobs = bool(ret.get("token_logprobs", False))
        topk = ret.get("topk", 0) or 0
        if not isinstance(topk, int) or not 0 <= topk <= lim.max_topk:
            raise RequestError(f'"topk" must be an integer from 0 to {lim.max_topk}')
        hidden = _int_list(ret.get("hidden_states"), "hidden_states", 0, L)
        mlp = _int_list(ret.get("mlp_act"), "mlp_act", 0, L - 1 if L else None)
        attn = _int_list(ret.get("attn_layers"), "attn_layers", 0, L - 1 if L else None)
        lengths = [self.backend.num_tokens(t) for t in texts]
        if max(lengths) > lim.max_text_tokens:
            raise RequestError(f"a text has {max(lengths):,} tokens; at most {lim.max_text_tokens:,}")
        if sum(lengths) > lim.max_total_tokens:
            raise RequestError(f"{sum(lengths):,} tokens in total; at most {lim.max_total_tokens:,}")
        if attn and (len(texts) > lim.max_attention_texts or max(lengths) > lim.max_attention_tokens):
            raise RequestError(f"attention needs at most {lim.max_attention_texts} texts of at most {lim.max_attention_tokens:,} tokens")
        d = self.backend.dims()
        n, T = len(texts), max(lengths)
        est = n * T * (4 + 4) + n * T * topk * 8
        est += n * len(hidden) * T * (d.get("hidden") or 0) * 2 + n * len(mlp) * T * (d.get("intermediate") or 0) * 2
        est += n * len(attn) * (d.get("heads") or 0) * T * T * 2
        if est > lim.max_pass_bytes:
            raise RequestError(f"the results would take about {est / 2**30:.1f} GB; at most {lim.max_pass_bytes / 2**30:.1f} GB per pass")
        edits = self.edit_store.load_all() if model == "edited" else None
        with self.backend.using(edits):
            arrays, meta = self.backend.teacher_forced(texts, token_logprobs=logprobs, topk=topk, hidden_states=hidden, mlp_act=mlp, attn_layers=attn)
        folder = self.work_dir / f"fwd_{k}"
        folder.mkdir(parents=True, exist_ok=True)
        files = {}
        for name, a in arrays.items():
            np.save(folder / f"{name}.npy", a)
            files[f"{name}.npy"] = list(a.shape)
        meta.update(model=model, files=files)
        (folder / "meta.json").write_text(json.dumps(meta, indent=1))
        summary = {"type": "teacher_forced", "model": model, "texts": n, "tokens": int(sum(lengths)), "dir": str(folder), "files": files}
        if logprobs:
            lp = arrays["token_logprobs"]
            summary["mean_token_logprob"] = float(np.nanmean(lp)) if np.isfinite(lp).any() else None
        return summary
