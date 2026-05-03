#!/usr/bin/env python3
"""Benchmark harness for sast-agent against AICGSecEval V1 dataset.

Clones each test case repo at the specified commit, runs sast-agent focused
on the known vulnerable file, and checks whether the tool identifies the
vulnerability at the correct location.

Usage:
    python benchmark/run_benchmark.py --dataset /path/to/data_v1.json \
        --languages javascript python --max-cases 10 --output results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Add parent dir to path so we can import sast_agent.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sast_agent.config import BedrockConfig, ScanConfig, ToolConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
)
logger = logging.getLogger(__name__)


def _setup_logging(log_file: str = "") -> None:
    """Add a debug-level file handler to the root logger."""
    if not log_file:
        return
    root = logging.getLogger()
    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(name)s — %(message)s"))
    root.addHandler(fh)
    root.setLevel(logging.INFO)

CLONE_DIR = "/tmp/benchmark-repos"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class TestCase:
    instance_id: str
    repo: str
    base_commit: str
    vuln_file: str
    vuln_lines: list[int]  # [start, end]
    language: str
    vuln_type: str
    vuln_source: str  # CVE ID
    cwe_id: str


@dataclass
class BenchmarkResult:
    instance_id: str
    repo: str
    vuln_file: str
    vuln_lines: list[int]
    vuln_type: str
    cwe_id: str
    language: str

    # Outcome
    detected: bool = False  # Did we find a vuln in the right file?
    line_overlap: bool = False  # Did our finding overlap the known vuln lines?
    correct_cwe: bool = False  # Did we identify the right CWE category?
    findings_count: int = 0
    finding_titles: list[str] = field(default_factory=list)
    finding_files: list[str] = field(default_factory=list)
    confidence: float = 0.0
    elapsed_seconds: float = 0.0
    error: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clone_repo(repo: str, commit: str, dest: str) -> bool:
    """Clone a GitHub repo and checkout a specific commit.

    Handles ``commit^`` syntax (parent of commit) by resolving it with
    ``git rev-parse`` after a full clone.
    """
    if os.path.exists(dest):
        shutil.rmtree(dest)

    repo_url = f"https://github.com/{repo}.git"
    has_caret = commit.endswith("^")
    logger.info("Cloning %s @ %s", repo_url, commit)

    try:
        # If commit has ^, we need the full history to resolve the parent.
        # Otherwise try a shallow clone first for speed.
        if not has_caret:
            result = subprocess.run(
                ["git", "clone", "--no-checkout", "--filter=blob:none", repo_url, dest],
                capture_output=True, text=True, timeout=300,
            )
        else:
            result = subprocess.run(
                ["git", "clone", "--no-checkout", "--filter=blob:none", repo_url, dest],
                capture_output=True, text=True, timeout=600,
            )

        if result.returncode != 0:
            if os.path.exists(dest):
                shutil.rmtree(dest)
            result = subprocess.run(
                ["git", "clone", "--no-checkout", repo_url, dest],
                capture_output=True, text=True, timeout=600,
            )
            if result.returncode != 0:
                logger.error("Clone failed: %s", result.stderr[:200])
                return False

        # Resolve the commit ref (handles ^ syntax).
        resolve = subprocess.run(
            ["git", "rev-parse", commit],
            cwd=dest, capture_output=True, text=True, timeout=30,
        )
        if resolve.returncode != 0:
            logger.error("Could not resolve %s: %s", commit, resolve.stderr[:200])
            return False

        resolved_sha = resolve.stdout.strip()

        # Checkout — this fetches the tree/blobs on demand with partial clone.
        result = subprocess.run(
            ["git", "checkout", resolved_sha],
            cwd=dest, capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            logger.error("Checkout failed for %s: %s", resolved_sha, result.stderr[:200])
            return False

        return True
    except subprocess.TimeoutExpired:
        logger.error("Clone/checkout timed out for %s", repo)
        return False
    except Exception as exc:
        logger.error("Clone error for %s: %s", repo, exc)
        return False


def map_vuln_type_to_risk_area(vuln_type: str) -> str:
    """Map AICGSecEval vuln types to our risk_area categories."""
    mapping = {
        "SQLI": "sqli",
        "XSS": "xss",
        "Command Injection": "rce",
        "Path Traversal": "path_traversal",
    }
    return mapping.get(vuln_type, vuln_type.lower())


def check_line_overlap(finding_files: list[str], vuln_file: str,
                       report: Any, vuln_lines: list[int]) -> bool:
    """Check if any finding's evidence overlaps the known vulnerable lines."""
    if len(vuln_lines) < 2:
        return False
    vuln_start, vuln_end = vuln_lines[0], vuln_lines[1]

    for finding in report.findings:
        for af in finding.affected_files:
            if vuln_file in af or af in vuln_file:
                # Check evidence line ranges.
                for ev in finding.evidence:
                    if ev.line_range:
                        try:
                            parts = ev.line_range.replace(" ", "").split("-")
                            ev_start = int(parts[0])
                            ev_end = int(parts[1]) if len(parts) > 1 else ev_start
                            # Check overlap.
                            if ev_start <= vuln_end and ev_end >= vuln_start:
                                return True
                        except (ValueError, IndexError):
                            continue
    return False


def check_cwe_match(report: Any, expected_cwe: str) -> bool:
    """Check if any finding matches the expected CWE category."""
    expected = expected_cwe.lower().replace("-", "")
    risk_area_to_cwe = {
        "sqli": "cwe89",
        "xss": "cwe79",
        "rce": "cwe78",
        "command_injection": "cwe78",
        "path_traversal": "cwe22",
        "ssrf": "cwe918",
    }
    for finding in report.findings:
        # Check by title/description keywords.
        title_lower = finding.title.lower()
        desc_lower = finding.description.lower()
        combined = title_lower + " " + desc_lower

        if expected in combined.replace("-", "").replace(" ", ""):
            return True

        # Check by risk area mapping.
        for risk_area, cwe in risk_area_to_cwe.items():
            if cwe == expected and risk_area in combined:
                return True

    return False


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

def run_single_case(tc: TestCase, max_hypotheses: int = 5) -> BenchmarkResult:
    """Run planner + analyzer on a single test case and evaluate the result.

    Skips the verifier — we score against the raw planner hypotheses and
    analyzer evidence so that "uncertain" findings still count as detections
    if they point to the right file/lines.
    """
    from sast_agent.agents.planner import run_planner
    from sast_agent.agents.analyzer import run_analyzer
    from sast_agent.providers import create_llm_from_config
    from sast_agent.tool_registry import ToolHandler

    result = BenchmarkResult(
        instance_id=tc.instance_id,
        repo=tc.repo,
        vuln_file=tc.vuln_file,
        vuln_lines=tc.vuln_lines,
        vuln_type=tc.vuln_type,
        cwe_id=tc.cwe_id,
        language=tc.language,
    )

    # Clone the repo.
    dest = os.path.join(CLONE_DIR, tc.instance_id)
    if not clone_repo(tc.repo, tc.base_commit, dest):
        result.error = "clone_failed"
        return result

    # Check the vuln file exists.
    vuln_path = os.path.join(dest, tc.vuln_file)
    if not os.path.exists(vuln_path):
        result.error = f"vuln_file_not_found: {tc.vuln_file}"
        logger.warning("Vuln file not found: %s in %s", tc.vuln_file, dest)
        return result

    t0 = time.time()
    try:
        config = ScanConfig(
            bedrock=BedrockConfig(),
            tools=ToolConfig(
                codeql_enabled=True,
                semantic_search_enabled=True,
                poc_generation_enabled=True,
                semantic_index_path=f"/tmp/bench-index-{tc.instance_id}",
            ),
            max_hypotheses=max_hypotheses,
            repo_path=dest,
            target_files=[tc.vuln_file],
        )

        bedrock = create_llm_from_config(config.llm)
        tool_handler = ToolHandler(config)

        # Phase 1: Planner.
        planner_output = run_planner(config, bedrock)

        # Check if any hypothesis targets the right file.
        for h in planner_output.hypotheses:
            for f in h.files:
                if tc.vuln_file in f or f in tc.vuln_file:
                    result.detected = True
                    break

        # Check if any hypothesis matches the right vuln type.
        expected_risk = map_vuln_type_to_risk_area(tc.vuln_type)
        for h in planner_output.hypotheses:
            if h.risk_area == expected_risk or expected_risk in h.hypothesis.lower():
                result.correct_cwe = True
                break

        # Phase 2: Analyzers.
        all_evidence_files: list[str] = []
        for h in planner_output.hypotheses:
            try:
                ao = run_analyzer(h, config, bedrock, tool_handler)

                # Check analyzer evidence for file match.
                for ev in ao.evidence:
                    all_evidence_files.append(ev.file)
                    if tc.vuln_file in ev.file or ev.file in tc.vuln_file:
                        result.detected = True
                        result.confidence = max(result.confidence, ao.confidence)

                        # Check line overlap from evidence.
                        if ev.line_range and len(tc.vuln_lines) >= 2:
                            try:
                                parts = ev.line_range.replace(" ", "").split("-")
                                ev_start = int(parts[0])
                                ev_end = int(parts[1]) if len(parts) > 1 else ev_start
                                if ev_start <= tc.vuln_lines[1] and ev_end >= tc.vuln_lines[0]:
                                    result.line_overlap = True
                            except (ValueError, IndexError):
                                pass

                # Also check analyzer reasoning for vuln type match.
                combined = (ao.reasoning + " " + ao.exploitability).lower()
                if expected_risk in combined or tc.vuln_type.lower() in combined:
                    result.correct_cwe = True

                result.finding_titles.append(h.hypothesis)
                result.findings_count += 1

            except Exception as exc:
                logger.warning("Analyzer failed for hypothesis %s: %s", h.id, exc)

        result.finding_files = all_evidence_files
        result.elapsed_seconds = time.time() - t0

    except Exception as exc:
        result.elapsed_seconds = time.time() - t0
        result.error = str(exc)[:200]
        logger.error("Scan failed for %s: %s", tc.instance_id, exc)

    # Cleanup index cache.
    for ext in (".faiss", ".meta.json"):
        p = f"/tmp/bench-index-{tc.instance_id}{ext}"
        if os.path.exists(p):
            os.remove(p)

    return result


def _result_to_dict(r: BenchmarkResult) -> dict:
    return {
        "instance_id": r.instance_id,
        "repo": r.repo,
        "vuln_file": r.vuln_file,
        "vuln_lines": r.vuln_lines,
        "vuln_type": r.vuln_type,
        "cwe_id": r.cwe_id,
        "language": r.language,
        "detected": r.detected,
        "line_overlap": r.line_overlap,
        "correct_cwe": r.correct_cwe,
        "findings_count": r.findings_count,
        "finding_titles": r.finding_titles,
        "confidence": r.confidence,
        "elapsed_seconds": r.elapsed_seconds,
        "error": r.error,
    }


def _save_results(results: list[BenchmarkResult], output_path: str) -> None:
    """Write current results to disk (called after every case)."""
    with open(output_path, "w") as f:
        json.dump([_result_to_dict(r) for r in results], f, indent=2)


def run_benchmark(
    dataset_path: str,
    languages: list[str] | None = None,
    vuln_types: list[str] | None = None,
    max_cases: int = 0,
    max_hypotheses: int = 5,
    output_path: str = "benchmark-results.json",
) -> list[BenchmarkResult]:
    """Run the full benchmark suite."""
    with open(dataset_path) as f:
        data = json.load(f)

    # Parse test cases.
    cases: list[TestCase] = []
    for entry in data:
        tc = TestCase(
            instance_id=entry["instance_id"],
            repo=entry["repo"],
            base_commit=entry["base_commit"],
            vuln_file=entry["vuln_file"],
            vuln_lines=entry.get("vuln_lines", []),
            language=entry["language"],
            vuln_type=entry["vuln_type"],
            vuln_source=entry.get("vuln_source", ""),
            cwe_id=entry.get("cwe_id", ""),
        )
        cases.append(tc)

    # Filter.
    if languages:
        langs_lower = [l.lower() for l in languages]
        cases = [c for c in cases if c.language.lower() in langs_lower]
    if vuln_types:
        vt_lower = [v.lower() for v in vuln_types]
        cases = [c for c in cases if c.vuln_type.lower() in vt_lower]
    if max_cases > 0:
        cases = cases[:max_cases]

    logger.info("Running benchmark: %d test cases", len(cases))

    # Load existing results if resuming.
    results: list[BenchmarkResult] = []
    completed_ids: set[str] = set()
    if os.path.exists(output_path):
        try:
            with open(output_path) as f:
                existing = json.load(f)
            for entry in existing:
                # Only skip cases that completed successfully — retry errors.
                if entry.get("error"):
                    continue
                completed_ids.add(entry["instance_id"])
                results.append(BenchmarkResult(
                    instance_id=entry["instance_id"],
                    repo=entry["repo"],
                    vuln_file=entry["vuln_file"],
                    vuln_lines=entry["vuln_lines"],
                    vuln_type=entry["vuln_type"],
                    cwe_id=entry["cwe_id"],
                    language=entry["language"],
                    detected=entry["detected"],
                    line_overlap=entry["line_overlap"],
                    correct_cwe=entry["correct_cwe"],
                    findings_count=entry["findings_count"],
                    finding_titles=entry["finding_titles"],
                    confidence=entry["confidence"],
                    elapsed_seconds=entry["elapsed_seconds"],
                    error=entry["error"],
                ))
            if completed_ids:
                logger.info("Resuming: %d cases already completed", len(completed_ids))
        except (json.JSONDecodeError, KeyError):
            pass

    for i, tc in enumerate(cases):
        if tc.instance_id in completed_ids:
            logger.info("=== [%d/%d] %s — SKIPPED (already done) ===", i + 1, len(cases), tc.instance_id)
            continue

        logger.info(
            "=== [%d/%d] %s — %s (%s) — %s ===",
            i + 1, len(cases), tc.instance_id, tc.vuln_type, tc.language, tc.repo,
        )
        result = run_single_case(tc, max_hypotheses=max_hypotheses)
        results.append(result)

        status = "✅ DETECTED" if result.detected else "❌ MISSED"
        if result.error:
            status = f"⚠️  ERROR: {result.error[:60]}"
        logger.info(
            "  Result: %s | lines_overlap=%s | cwe_match=%s | %.0fs",
            status, result.line_overlap, result.correct_cwe, result.elapsed_seconds,
        )

        # Save after every case so progress isn't lost.
        _save_results(results, output_path)

    return results


def print_summary(results: list[BenchmarkResult]) -> None:
    """Print a summary scorecard."""
    total = len(results)
    errors = sum(1 for r in results if r.error)
    evaluated = total - errors
    detected = sum(1 for r in results if r.detected)
    line_hits = sum(1 for r in results if r.line_overlap)
    cwe_hits = sum(1 for r in results if r.correct_cwe)

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Total test cases:    {total}")
    print(f"Errors (skip):       {errors}")
    print(f"Evaluated:           {evaluated}")
    print(f"Detected (file):     {detected}/{evaluated} ({detected/max(evaluated,1)*100:.1f}%)")
    print(f"Line overlap:        {line_hits}/{evaluated} ({line_hits/max(evaluated,1)*100:.1f}%)")
    print(f"CWE match:           {cwe_hits}/{evaluated} ({cwe_hits/max(evaluated,1)*100:.1f}%)")
    print(f"Avg time per case:   {sum(r.elapsed_seconds for r in results)/max(total,1):.1f}s")

    # Breakdown by language.
    print("\nBy language:")
    langs = sorted(set(r.language for r in results))
    for lang in langs:
        lr = [r for r in results if r.language == lang and not r.error]
        ld = sum(1 for r in lr if r.detected)
        print(f"  {lang:12s}: {ld}/{len(lr)} detected ({ld/max(len(lr),1)*100:.0f}%)")

    # Breakdown by vuln type.
    print("\nBy vulnerability type:")
    vtypes = sorted(set(r.vuln_type for r in results))
    for vt in vtypes:
        vr = [r for r in results if r.vuln_type == vt and not r.error]
        vd = sum(1 for r in vr if r.detected)
        print(f"  {vt:20s}: {vd}/{len(vr)} detected ({vd/max(len(vr),1)*100:.0f}%)")

    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark sast-agent against AICGSecEval")
    parser.add_argument("--dataset", required=True, help="Path to data_v1.json")
    parser.add_argument("--languages", nargs="*", help="Filter by language (e.g. javascript python)")
    parser.add_argument("--vuln-types", nargs="*", help="Filter by vuln type (e.g. SQLI XSS)")
    parser.add_argument("--max-cases", type=int, default=0, help="Max cases to run (0=all)")
    parser.add_argument("--max-hypotheses", type=int, default=5, help="Max hypotheses per scan")
    parser.add_argument("--output", "-o", default="benchmark-results.json", help="Output JSON file")
    parser.add_argument("--log-file", default="benchmark-debug.log", help="Debug log file (default: benchmark-debug.log)")
    args = parser.parse_args()

    _setup_logging(args.log_file)

    results = run_benchmark(
        dataset_path=args.dataset,
        languages=args.languages,
        vuln_types=args.vuln_types,
        max_cases=args.max_cases,
        max_hypotheses=args.max_hypotheses,
        output_path=args.output,
    )

    print(f"\nResults written to {args.output}")
    print_summary(results)


if __name__ == "__main__":
    main()
