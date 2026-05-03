"""CodeQL integration for deep static analysis."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field

from sast_agent.tools.shell import ShellResult, run_shell


@dataclass
class CodeQLResult:
    query: str
    results: list[dict]
    raw_output: str
    success: bool
    error: str = ""


@dataclass
class CodeQLDatabase:
    path: str
    language: str
    created: bool = False


# ---------------------------------------------------------------------------
# Built-in query templates for common vulnerability classes
# ---------------------------------------------------------------------------

BUILTIN_QUERIES: dict[str, dict[str, str]] = {
    "sqli": {
        "javascript": "codeql/javascript-queries:Security/CWE-089/SqlInjection.ql",
        "python": "codeql/python-queries:Security/CWE-089/SqlInjection.ql",
        "java": "codeql/java-queries:Security/CWE/CWE-089/SqlTainted.ql",
    },
    "xss": {
        "javascript": "codeql/javascript-queries:Security/CWE-079/Xss.ql",
        "python": "codeql/python-queries:Security/CWE-079/ReflectedXss.ql",
        "java": "codeql/java-queries:Security/CWE/CWE-079/XSS.ql",
    },
    "ssrf": {
        "javascript": "codeql/javascript-queries:Security/CWE-918/RequestForgery.ql",
        "python": "codeql/python-queries:Security/CWE-918/FullServerSideRequestForgery.ql",
        "java": "codeql/java-queries:Security/CWE/CWE-918/RequestForgery.ql",
    },
    "path_traversal": {
        "javascript": "codeql/javascript-queries:Security/CWE-022/TaintedPath.ql",
        "python": "codeql/python-queries:Security/CWE-022/PathInjection.ql",
        "java": "codeql/java-queries:Security/CWE/CWE-022/TaintedPath.ql",
    },
    "command_injection": {
        "javascript": "codeql/javascript-queries:Security/CWE-078/CommandInjection.ql",
        "python": "codeql/python-queries:Security/CWE-078/CommandInjection.ql",
        "java": "codeql/java-queries:Security/CWE/CWE-078/ExecTainted.ql",
    },
    "insecure_deserialization": {
        "javascript": "codeql/javascript-queries:Security/CWE-502/UnsafeDeserialization.ql",
        "python": "codeql/python-queries:Security/CWE-502/UnsafeDeserialization.ql",
        "java": "codeql/java-queries:Security/CWE/CWE-502/UnsafeDeserialization.ql",
    },
    "hardcoded_credentials": {
        "javascript": "codeql/javascript-queries:Security/CWE-798/HardcodedCredentials.ql",
    },
}


def detect_language(repo_path: str) -> str:
    """Detect the primary language of the repository by file count.

    Counts source files by extension (excluding vendor/node_modules/test dirs)
    and returns the language with the most files.
    """
    ext_to_lang = {
        ".php": "php",
        ".py": "python",
        ".js": "javascript",
        ".ts": "javascript",
        ".jsx": "javascript",
        ".tsx": "javascript",
        ".java": "java",
        ".go": "go",
        ".rb": "ruby",
        ".cs": "csharp",
        ".c": "cpp",
        ".cpp": "cpp",
        ".rs": "rust",
        ".swift": "swift",
    }

    # Count files per language, excluding common vendor directories.
    result = run_shell(
        "find . -type f "
        "-not -path './.git/*' "
        "-not -path '*/node_modules/*' "
        "-not -path '*/vendor/*' "
        "-not -path '*/dist/*' "
        "-not -path '*/build/*' "
        "-not -path '*/js/jquery/*' "
        "-not -path '*/js/vendor/*' "
        "| head -2000",
        cwd=repo_path,
    )

    lang_counts: dict[str, int] = {}
    for line in result.stdout.strip().splitlines():
        ext = ""
        parts = line.rsplit(".", 1)
        if len(parts) == 2:
            ext = "." + parts[1].lower()
        lang = ext_to_lang.get(ext)
        if lang:
            lang_counts[lang] = lang_counts.get(lang, 0) + 1

    if not lang_counts:
        # Fallback: check for config files.
        config_checks = [
            ("python", ["setup.py", "pyproject.toml", "requirements.txt"]),
            ("javascript", ["package.json"]),
            ("java", ["pom.xml", "build.gradle"]),
            ("go", ["go.mod"]),
            ("ruby", ["Gemfile"]),
            ("php", ["composer.json"]),
        ]
        for lang, files in config_checks:
            for fname in files:
                check = run_shell(f"test -f {fname} && echo yes", cwd=repo_path)
                if check.stdout.strip() == "yes":
                    return lang
        return "javascript"

    # Return the language with the most source files.
    return max(lang_counts, key=lang_counts.get)


def create_database(repo_path: str, language: str, db_path: str = "") -> CodeQLDatabase:
    """Create a CodeQL database for the repository."""
    if not db_path:
        db_path = tempfile.mkdtemp(prefix=f"codeql-db-{language}-")

    result = run_shell(
        f"codeql database create {db_path} --language={language} --overwrite",
        cwd=repo_path,
        timeout=300,
    )

    return CodeQLDatabase(
        path=db_path,
        language=language,
        created=result.returncode == 0,
    )


def run_query(db_path: str, query: str, timeout: int = 120) -> CodeQLResult:
    """Run a CodeQL query against a database.

    ``query`` can be:
    - A path to a .ql file
    - A builtin key like "sqli" (resolved via BUILTIN_QUERIES)
    - A CodeQL suite name
    """
    output_path = tempfile.mktemp(prefix="codeql-results-", suffix=".sarif")

    result = run_shell(
        f"codeql database analyze {db_path} {query} "
        f"--format=sarifv2.1.0 --output={output_path} "
        f"--threads=0",
        timeout=timeout,
    )

    if result.returncode != 0:
        return CodeQLResult(
            query=query,
            results=[],
            raw_output=result.stderr,
            success=False,
            error=result.stderr,
        )

    # Parse SARIF output.
    results = _parse_sarif(output_path)

    return CodeQLResult(
        query=query,
        results=results,
        raw_output=json.dumps(results, indent=2)[:20_000] if results else "0 results",
        success=True,
    )


def run_builtin_query(db_path: str, risk_area: str, language: str) -> CodeQLResult:
    """Run a built-in query for a known risk area."""
    queries = BUILTIN_QUERIES.get(risk_area, {})
    query_path = queries.get(language)
    if not query_path:
        return CodeQLResult(
            query=risk_area,
            results=[],
            raw_output="",
            success=False,
            error=f"No built-in query for {risk_area}/{language}",
        )
    return run_query(db_path, query_path)


def list_available_queries(language: str) -> list[str]:
    """List available built-in queries for a language."""
    available = []
    for risk_area, lang_queries in BUILTIN_QUERIES.items():
        if language in lang_queries:
            available.append(f"{risk_area}: {lang_queries[language]}")
    return available


def _parse_sarif(sarif_path: str) -> list[dict]:
    """Parse a SARIF file and return a list of finding dicts."""
    sarif_result = run_shell(f"cat {sarif_path}", check_safety=True)
    results: list[dict] = []
    try:
        sarif = json.loads(sarif_result.stdout)
        for run_obj in sarif.get("runs", []):
            for finding in run_obj.get("results", []):
                results.append({
                    "ruleId": finding.get("ruleId", ""),
                    "message": finding.get("message", {}).get("text", ""),
                    "level": finding.get("level", ""),
                    "locations": [
                        {
                            "file": loc.get("physicalLocation", {})
                            .get("artifactLocation", {})
                            .get("uri", ""),
                            "startLine": loc.get("physicalLocation", {})
                            .get("region", {})
                            .get("startLine", 0),
                            "endLine": loc.get("physicalLocation", {})
                            .get("region", {})
                            .get("endLine", 0),
                        }
                        for loc in finding.get("locations", [])
                    ],
                })
    except (json.JSONDecodeError, KeyError):
        pass
    return results


def run_raw_query(db_path: str, ql_source: str, timeout: int = 180) -> CodeQLResult:
    """Write arbitrary QL source to a temp file and execute it.

    This lets the agent author custom taint-tracking queries, data-flow
    queries, or any ad-hoc CodeQL analysis on the fly.
    """
    # Write the QL source to a temp file.
    tmp_dir = tempfile.mkdtemp(prefix="codeql-raw-")
    ql_path = os.path.join(tmp_dir, "custom-query.ql")
    with open(ql_path, "w") as f:
        f.write(ql_source)

    # Also write a qlpack.yml so CodeQL can resolve the query.
    qlpack_path = os.path.join(tmp_dir, "qlpack.yml")

    # Detect the language from the database.
    db_info = run_shell(f"cat {db_path}/codeql-database.yml", check_safety=True)
    lang = "javascript"  # default
    for line in db_info.stdout.splitlines():
        if line.strip().startswith("primaryLanguage:"):
            lang = line.split(":", 1)[1].strip().strip("'\"")
            break

    # Map language to the correct CodeQL standard library pack.
    lang_pack_map = {
        "javascript": "codeql/javascript-all",
        "python": "codeql/python-all",
        "java": "codeql/java-all",
        "go": "codeql/go-all",
        "ruby": "codeql/ruby-all",
        "csharp": "codeql/csharp-all",
        "cpp": "codeql/cpp-all",
        "c": "codeql/cpp-all",
        "swift": "codeql/swift-all",
        "rust": "codeql/rust-all",
    }
    dep_pack = lang_pack_map.get(lang, f"codeql/{lang}-all")

    qlpack_content = (
        f"name: sast-agent/custom-query\n"
        f"version: 0.0.1\n"
        f"dependencies:\n"
        f"  {dep_pack}: \"*\"\n"
    )
    with open(qlpack_path, "w") as f:
        f.write(qlpack_content)

    # Install pack dependencies.
    install_result = run_shell(
        f"codeql pack install {tmp_dir}",
        timeout=120,
    )
    if install_result.returncode != 0:
        return CodeQLResult(
            query=ql_source[:200],
            results=[],
            raw_output=install_result.stderr,
            success=False,
            error=f"Pack install failed: {install_result.stderr[:500]}",
        )

    # Run the query.
    output_path = os.path.join(tmp_dir, "results.sarif")
    result = run_shell(
        f"codeql database analyze {db_path} {ql_path} "
        f"--format=sarifv2.1.0 --output={output_path} "
        f"--threads=0",
        timeout=timeout,
    )

    if result.returncode != 0:
        # Try running as a raw query with bqrs output as fallback.
        bqrs_path = os.path.join(tmp_dir, "results.bqrs")
        result2 = run_shell(
            f"codeql query run --database={db_path} {ql_path} --output={bqrs_path}",
            timeout=timeout,
        )
        if result2.returncode != 0:
            return CodeQLResult(
                query=ql_source[:200],
                results=[],
                raw_output=result.stderr + "\n---\n" + result2.stderr,
                success=False,
                error=result2.stderr[:500],
            )

        # Decode bqrs to text.
        decode_result = run_shell(
            f"codeql bqrs decode --format=csv {bqrs_path}",
            timeout=60,
        )
        return CodeQLResult(
            query=ql_source[:200],
            results=[{"csv_output": decode_result.stdout[:20_000]}],
            raw_output=decode_result.stdout[:20_000],
            success=True,
        )

    # Parse SARIF.
    results = _parse_sarif(output_path)
    return CodeQLResult(
        query=ql_source[:200],
        results=results,
        raw_output=json.dumps(results, indent=2)[:20_000] if results else "0 results",
        success=True,
    )
