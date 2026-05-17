"""Command-line entrypoint for ggufscan.

Orchestration strategy
----------------------
llama.cpp cannot do chat completion and embeddings on the same Llama instance.
We therefore run the scan in two passes when any selected test needs embeddings:

  Pass 1 — chat handle: every test calls `collect()` to produce raw samples.
  Pass 2 — embedding handle (if needed): tests that need vectors call `score()`
           with the embedding handle; others score in pass 1.

The model is loaded sequentially (chat unloaded before embedding), so peak VRAM
stays at one model instance.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

_DEFAULT_CONFIG_PATH = Path("config/ggufscan.yml")


def _load_config(path: Path) -> dict:
    """Load YAML config. Returns {} on missing file; raises on parse error."""
    if not path.is_file():
        return {}
    import yaml
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    return data

from ggufscan.parser import parse
from ggufscan.report import to_json, to_markdown, to_ragas_dataset
from ggufscan.static_scan import StaticScanner
from ggufscan.tests import TEST_REGISTRY
from ggufscan.tests.base import Collected, TestResult
from ggufscan.utils import strip_think

log = logging.getLogger("ggufscan")


def _parse_split(s: str | None) -> list[float] | None:
    if not s:
        return None
    return [float(x) for x in s.split(",")]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ggufscan",
        description="GGUF static + dynamic security scanner.",
    )
    p.add_argument("--model", required=True, help="Path to .gguf file")
    p.add_argument(
        "--tests", default="jailbreak,harmful_bias,backdoor",
        help="Comma-separated test names. Choices: " + ", ".join(TEST_REGISTRY),
    )
    p.add_argument("--static-only", action="store_true", help="Skip dynamic inference tests")
    p.add_argument("--quick", action="store_true", help="Subset of prompts per test (smoke run)")
    p.add_argument("--n-ctx", type=int, default=4096)
    p.add_argument("--n-gpu-layers", type=int, default=-1)
    p.add_argument("--tensor-split", default=None, help="Comma-separated floats (e.g. '0.5,0.5')")
    p.add_argument("--chat-format", default=None, help="Override chat_format (default: auto-detect)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-tokens", type=int, default=-1,
                   help="Default: 256, auto-bumped to 1024 for thinking models (qwen3, glm4, deepseek_r1)")
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--output", default=None, help="Markdown report output path (file or dir)")
    p.add_argument("--json", dest="json_out", default=None,
                   help="Full JSON report output path (file or dir)")
    p.add_argument("--ragas", dest="ragas_out", default=None,
                   help="RAGAS-compatible Q/A/reference dataset (JSON list, file or dir)")

    # --- RAGAS judge stage (runs in-process after the 3 tests, on a separate model) ---
    p.add_argument("--config", default=str(_DEFAULT_CONFIG_PATH),
                   help=f"Path to YAML config (default: {_DEFAULT_CONFIG_PATH}). "
                        "`judge.model` triggers in-pipeline RAGAS eval.")
    p.add_argument("--judge-model", default=None,
                   help="Path to judge .gguf (overrides config `judge.model`). "
                        "If resolved (flag or config), evaluator runs automatically "
                        "after dynamic tests and the report is enriched with a RAGAS "
                        "section + LLM-generated `# Analysis Summary`.")
    p.add_argument("--judge-n-ctx", type=int, default=None,
                   help="Override config `judge.n_ctx` (default: 4096)")
    p.add_argument("--judge-tensor-split", default=None,
                   help="Override config `judge.tensor_split` (falls back to --tensor-split)")
    p.add_argument("--judge-max-tokens", type=int, default=None,
                   help="Override config `judge.max_tokens` (default: 512)")
    p.add_argument("--judge-temperature", type=float, default=None,
                   help="Override config `judge.temperature` (default: 0.0)")
    p.add_argument("--ragas-results", dest="ragas_results_out", default=None,
                   help="Enriched RAGAS JSON output (per-record metrics). "
                        "Required only to write that artifact; the section is added to "
                        "the main markdown report regardless.")
    p.add_argument("--skip-ragas-eval", action="store_true",
                   help="Skip the in-pipeline judge stage even if --judge-model is set.")

    p.add_argument("-v", "--verbose", action="count", default=0)
    return p


def _run_dynamic(parsed, args, requested: list[str]) -> list[TestResult]:
    from ggufscan.inference import ModelHandle

    tests = {name: TEST_REGISTRY[name]() for name in requested}
    chat_kwargs: dict[str, Any] = {
        "n_ctx": args.n_ctx,
        "n_gpu_layers": args.n_gpu_layers,
        "tensor_split": _parse_split(args.tensor_split),
        "chat_format": args.chat_format,
        "seed": args.seed,
    }
    needs_embedding = any(t.needs_embedding for t in tests.values())

    # Pass 1 — chat handle: collect samples for every test.
    log.info("Pass 1/2: loading model for chat completion...")
    collected: dict[str, Collected] = {}
    with ModelHandle(parsed, embedding=False, **chat_kwargs) as chat:
        for name, test in tests.items():
            log.info("Collecting samples for %s...", name)
            collected[name] = test.collect(
                chat, quick=args.quick,
                max_tokens=args.max_tokens, temperature=args.temperature,
                seed=args.seed,
            )

    # Pass 2 — embedding handle (only if needed). Otherwise score immediately.
    results: list[TestResult] = []
    embedding_handle = None

    if needs_embedding:
        log.info("Pass 2/2: reloading model with embedding=True...")
        with ModelHandle(parsed, embedding=True, **chat_kwargs) as emb:
            for name, test in tests.items():
                log.info("Scoring %s...", name)
                r = test.score(
                    collected[name],
                    embedding_handle=emb if test.needs_embedding else None,
                )
                results.append(r)
                _print_test_summary(r, collected[name])
    else:
        for name, test in tests.items():
            log.info("Scoring %s...", name)
            r = test.score(collected[name], embedding_handle=None)
            results.append(r)
            _print_test_summary(r, collected[name])

    return results


def _print_test_summary(r: TestResult, collected: Collected) -> None:
    """Print a 3-5 line condensed summary of a test result to stderr."""
    status = "✅ PASS" if r.passed else "❌ FAIL"
    elapsed = r.details.get("elapsed_seconds")
    elapsed_str = f"  elapsed={elapsed:.1f}s" if elapsed is not None else ""
    header = (f"[{r.name}] {status}  score={r.score:.2%} (threshold={r.threshold:.2%})  "
              f"n={len(collected.samples)}{elapsed_str}")
    lines = [header]

    if r.name == "jailbreak":
        cats = r.details.get("per_category", {})
        cat_str = ", ".join(
            f"{c}={s['refused']}/{s['total']}" for c, s in sorted(cats.items())
        )
        lines.append(f"  refused={r.details.get('refused')}/{r.details.get('total_prompts')}  by_category: {cat_str}")
    elif r.name == "harmful_bias":
        lines.append(
            f"  harmful={r.details.get('harmful_refused')}/{r.details.get('harmful_total')} "
            f"({r.details.get('harmful_score', 0):.2%})  "
            f"bias_cos_mean={r.details.get('bias_score', 0):.3f} "
            f"(pairs={r.details.get('bias_pairs_evaluated')})"
        )
        per_axis = r.details.get("bias_per_axis_mean", {})
        if per_axis:
            lines.append("  bias_per_axis: " + ", ".join(f"{k}={v:.3f}" for k, v in per_axis.items()))
    elif r.name == "backdoor":
        lines.append(
            f"  outliers={r.details.get('outliers')}/{r.details.get('n_pairs')} "
            f"(ratio={r.details.get('outlier_ratio', 0):.2%})  "
            f"dist mean={r.details.get('distance_mean', 0):.3f} "
            f"max={r.details.get('distance_max', 0):.3f}"
        )

    failed = [s for s in collected.samples
              if s.verdict is not None and not s.verdict.refused]
    if failed:
        lines.append("  top failures (stripped):")
        for s in failed[:3]:
            label = s.metadata.get("id") or s.metadata.get("category") or "?"
            ans = strip_think(s.response).replace("\n", " ")[:80] or "<empty after strip>"
            lines.append(f"    - {label}: {ans!r}")

    biased = [s for s in collected.samples if s.metadata.get("biased")]
    if biased:
        lines.append("  biased pairs:")
        for s in biased[:3]:
            lines.append(f"    - {s.metadata.get('id')} ({s.metadata.get('axis')}): "
                         f"cos={s.metadata.get('cosine_similarity', 0):.3f}")

    outliers = [s for s in collected.samples if s.metadata.get("is_outlier")]
    if outliers:
        lines.append("  backdoor outliers:")
        for s in outliers[:3]:
            lines.append(f"    - payload={s.metadata.get('payload')!r:>20} "
                         f"dist={s.metadata.get('cosine_distance', 0):.3f}")

    for ln in lines:
        print(ln, file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=max(logging.WARNING - 10 * args.verbose, logging.DEBUG),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    model_path = Path(args.model).expanduser()
    if not model_path.is_file():
        print(f"ERROR: model not found: {model_path}", file=sys.stderr)
        return 2

    cfg = _load_config(Path(args.config).expanduser())

    log.info("Parsing %s", model_path)
    parsed = parse(str(model_path))
    static_report = StaticScanner(parsed).scan()
    log.info("Static scan complete (score=%d)", static_report.security_score)

    # Resolve max_tokens default: thinking models need much more room because
    # <think>...</think> reasoning eats the budget before the visible answer.
    if args.max_tokens == -1:
        from ggufscan.inference import is_thinking_model
        if is_thinking_model(parsed):
            args.max_tokens = 1024
            print(f"[ggufscan] thinking arch '{parsed.architecture}' detected → "
                  f"max_tokens auto-bumped to {args.max_tokens}", file=sys.stderr)
        else:
            args.max_tokens = 256

    dynamic_results: list[TestResult] = []
    if not args.static_only:
        try:
            import llama_cpp  # noqa: F401
        except ImportError:
            print(
                "ERROR: llama-cpp-python not installed. Use --static-only or install:\n"
                "  CMAKE_ARGS=\"-DGGML_CUDA=on\" pip install llama-cpp-python",
                file=sys.stderr,
            )
            return 3

        requested = [t.strip() for t in args.tests.split(",") if t.strip()]
        unknown = [t for t in requested if t not in TEST_REGISTRY]
        if unknown:
            print(f"ERROR: unknown test(s): {unknown}. Choices: {list(TEST_REGISTRY)}", file=sys.stderr)
            return 4

        dynamic_results = _run_dynamic(parsed, args, requested)

    stem = model_path.stem
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = "static" if args.static_only else "scan"

    # Build RAGAS records (always in-memory if dynamic ran; written to disk only
    # if --ragas was passed). Judge stage runs on the in-memory list.
    ragas_records: list[dict] = []
    if dynamic_results:
        ragas_records = to_ragas_dataset(dynamic_results)

    if args.ragas_out and ragas_records:
        out = _resolve_out_path(args.ragas_out, default_name=f"{stem}_{suffix}_{stamp}_ragas.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(ragas_records, indent=2, ensure_ascii=False, default=str), encoding="utf-8",
        )
        log.info("RAGAS dataset written to %s (%d records)", out, len(ragas_records))
        print(f"RAGAS: {out} ({len(ragas_records)} records)", file=sys.stderr)

    # In-pipeline judge stage (runs after the 3 tests).
    ragas_eval = _maybe_run_judge(args, ragas_records, stem, stamp, suffix, cfg)

    md = to_markdown(static_report, dynamic_results, ragas_eval=ragas_eval)
    if args.output:
        out = _resolve_out_path(args.output, default_name=f"{stem}_{suffix}_{stamp}.md")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        log.info("Markdown report written to %s", out)
        print(f"Markdown: {out}", file=sys.stderr)
    else:
        print(md)

    if args.json_out:
        data = to_json(static_report, dynamic_results)
        if ragas_eval:
            data["ragas_eval"] = ragas_eval
        out = _resolve_out_path(args.json_out, default_name=f"{stem}_{suffix}_{stamp}.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8",
        )
        log.info("JSON report written to %s", out)
        print(f"JSON: {out}", file=sys.stderr)

    return 0


def _resolve_judge_cfg(args, cfg: dict) -> tuple[Path | None, dict, str | None]:
    """Resolve judge config: --judge-* flags override config `judge.*`.

    Returns (path, params, source) where source is 'flag', 'config', or None.
    params merges defaults + config + flag overrides.
    """
    jc = cfg.get("judge") or {}
    if args.judge_model:
        path = Path(args.judge_model).expanduser()
        source = "flag"
    elif jc.get("model"):
        path = Path(str(jc["model"])).expanduser()
        source = "config"
    else:
        return None, {}, None

    def _pick(flag_val, cfg_key, default):
        if flag_val is not None:
            return flag_val
        return jc.get(cfg_key, default)

    params = {
        "n_ctx": _pick(args.judge_n_ctx, "n_ctx", 4096),
        "max_tokens": _pick(args.judge_max_tokens, "max_tokens", 512),
        "temperature": _pick(args.judge_temperature, "temperature", 0.0),
        "tensor_split": _pick(args.judge_tensor_split, "tensor_split", None),
    }
    return path, params, source


def _maybe_run_judge(args, ragas_records, stem, stamp, suffix, cfg: dict):
    """Run the in-pipeline RAGAS judge stage if a judge model is resolved.

    Returns a dict {summary, analysis, records, judge_model} suitable for
    inclusion in the markdown / JSON report, or None if skipped.
    """
    if args.skip_ragas_eval or not ragas_records:
        return None

    judge_path, jparams, source = _resolve_judge_cfg(args, cfg)
    if judge_path is None:
        if args.ragas_out:
            print(f"INFO: no judge resolved (set `judge.model` in {args.config} or pass "
                  "--judge-model), skipping RAGAS eval stage", file=sys.stderr)
        return None

    if not judge_path.is_file():
        print(f"WARNING: judge model not found ({source}): {judge_path}, skipping eval",
              file=sys.stderr)
        return None
    if source == "config":
        log.info("Judge resolved from config %s: %s", args.config, judge_path)

    log.info("Loading judge model %s for in-pipeline RAGAS eval...", judge_path.name)
    from ggufscan.inference import ModelHandle
    from ggufscan.ragas_eval import RagasEvaluator

    judge_parsed = parse(str(judge_path))
    ts_raw = jparams["tensor_split"]
    if isinstance(ts_raw, list):
        judge_split = [float(x) for x in ts_raw]
    elif isinstance(ts_raw, str):
        judge_split = _parse_split(ts_raw)
    else:
        judge_split = _parse_split(args.tensor_split)
    judge_kwargs: dict[str, Any] = {
        "n_ctx": jparams["n_ctx"],
        "n_gpu_layers": args.n_gpu_layers,
        "tensor_split": judge_split,
        "chat_format": args.chat_format,
        "seed": args.seed,
    }

    with ModelHandle(judge_parsed, embedding=False, **judge_kwargs) as judge:
        ev = RagasEvaluator(
            judge=judge,
            max_tokens=jparams["max_tokens"],
            temperature=jparams["temperature"],
        )
        log.info("Judging %d RAGAS records...", len(ragas_records))
        enriched = ev.evaluate_all(ragas_records)
        summary = ev.summarize(enriched)
        log.info("Generating Analysis Summary...")
        analysis = ev.generate_analysis(enriched, summary)

    # Persist enriched JSON if --ragas-results requested, else alongside other artifacts.
    if args.ragas_results_out:
        out = _resolve_out_path(
            args.ragas_results_out,
            default_name=f"{stem}_{suffix}_{stamp}_ragas_results.json",
        )
    elif args.output:
        # Place next to the main markdown by default.
        base_dir = Path(args.output).expanduser()
        base_dir = base_dir if base_dir.is_dir() else base_dir.parent
        out = base_dir / f"{stem}_{suffix}_{stamp}_ragas_results.json"
    else:
        out = None

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {"summary": summary, "analysis": analysis, "records": enriched,
                 "judge_model": str(judge_path)},
                indent=2, ensure_ascii=False, default=str,
            ),
            encoding="utf-8",
        )
        log.info("RAGAS results written to %s", out)
        print(f"RAGAS results: {out}", file=sys.stderr)

    return {
        "summary": summary,
        "analysis": analysis,
        "records": enriched,
        "judge_model": str(judge_path),
    }


def _resolve_out_path(raw: str, default_name: str) -> Path:
    """If `raw` points to (or ends with /) a directory, append default_name."""
    p = Path(raw).expanduser()
    if p.is_dir() or raw.endswith("/") or raw.endswith("\\"):
        return p / default_name
    return p


if __name__ == "__main__":
    raise SystemExit(main())
