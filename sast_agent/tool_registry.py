"""Tool registry: maps Bedrock tool definitions to implementations."""

from __future__ import annotations

import json
from typing import Any

from sast_agent.config import ScanConfig
from sast_agent.tools.shell import run_shell
from sast_agent.tools.git_tools import (
    get_changed_files,
    get_diff,
    get_file_content,
    get_repo_structure,
    grep_codebase,
    get_git_log,
)
from sast_agent.tools.codeql import (
    create_database,
    detect_language,
    run_builtin_query,
    run_query,
    run_raw_query,
    list_available_queries,
)
from sast_agent.tools.python_repl import run_python
from sast_agent.tools.semantic_search import (
    SemanticIndex,
    index_repository,
    format_search_results,
)
from sast_agent.providers import create_embedding_from_config, EmbeddingProvider


# ---------------------------------------------------------------------------
# Bedrock tool definitions (JSON Schema for the Converse API)
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "toolSpec": {
            "name": "shell",
            "description": (
                "Execute a shell command. Allowed: ls, find, cat, head, tail, grep, rg, git, "
                "codeql, python, semgrep, tree, file, stat, diff, sort, uniq, awk, sed, jq. "
                "Blocked: rm -rf, network calls, destructive ops."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "The shell command to run"},
                        "cwd": {"type": "string", "description": "Working directory (default: repo root)"},
                    },
                    "required": ["command"],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "read_file",
            "description": "Read the contents of a file in the repository.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Relative path to the file"},
                    },
                    "required": ["path"],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "list_files",
            "description": "List the repository file structure up to a given depth.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "max_depth": {"type": "integer", "description": "Max directory depth (default 3)"},
                    },
                    "required": [],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "grep_code",
            "description": "Search the codebase for a regex pattern. Returns matching lines with file paths and line numbers.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Regex pattern to search for"},
                        "file_glob": {"type": "string", "description": "Optional file glob filter, e.g. '*.ts'"},
                    },
                    "required": ["pattern"],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "git_diff",
            "description": "Get the unified diff for a specific file or the entire changeset against a base ref.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Optional: specific file to diff"},
                        "base_ref": {"type": "string", "description": "Base ref to diff against (default: main)"},
                    },
                    "required": [],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "git_changed_files",
            "description": "List files changed in the current branch compared to a base ref.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "base_ref": {"type": "string", "description": "Base ref (default: main)"},
                    },
                    "required": [],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "git_log",
            "description": "Get recent git commit log.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "n": {"type": "integer", "description": "Number of commits (default 20)"},
                    },
                    "required": [],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "codeql_run",
            "description": (
                "Run a CodeQL query against the repository database. "
                "Can use built-in queries by risk area (sqli, xss, ssrf, path_traversal, "
                "command_injection, insecure_deserialization, hardcoded_credentials) "
                "or a custom .ql file path."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Risk area key or path to .ql file"},
                        "language": {"type": "string", "description": "Language (auto-detected if omitted)"},
                    },
                    "required": ["query"],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "codeql_list_queries",
            "description": "List available built-in CodeQL queries for a language.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "language": {"type": "string", "description": "Language to list queries for"},
                    },
                    "required": [],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "codeql_query_raw",
            "description": (
                "Write and execute an arbitrary CodeQL query. You provide the full QL source "
                "code and it runs against the repository database. Use this for custom taint-tracking, "
                "data-flow analysis, or any ad-hoc query not covered by the built-in risk areas. "
                "The query must be valid CodeQL QL. A qlpack with the correct language dependencies "
                "is auto-generated. Example:\n"
                "  import javascript\n"
                "  from CallExpr call, StringOps::ConcatenationRoot concat\n"
                "  where call.getCalleeName() = \"query\" and\n"
                "    concat = call.getArgument(0) and\n"
                "    concat.getALeaf() instanceof RemoteFlowSource\n"
                "  select call, \"Potential SQL injection\"\n"
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "ql_source": {
                            "type": "string",
                            "description": (
                                "Full CodeQL QL source code. Must include import statement "
                                "and a select clause."
                            ),
                        },
                    },
                    "required": ["ql_source"],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "python_exec",
            "description": (
                "Execute Python code in a sandboxed REPL. Use for PoC generation, "
                "data parsing, or analysis scripts. No file writes, no dangerous builtins. "
                "When a target URL is configured (DAST mode), 'import requests' is available "
                "but scoped to ONLY reach the target host — use it to send HTTP requests "
                "that verify vulnerabilities against the live application. "
                "Also available in DAST mode: json, re, base64, hashlib, urllib.parse, html."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "Python code to execute"},
                    },
                    "required": ["code"],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "semantic_search",
            "description": (
                "Semantic code search over the repository using vector embeddings. "
                "Finds code chunks (functions, classes, methods, blocks) that are "
                "semantically similar to a natural-language query or code snippet. "
                "Much more powerful than grep for finding related patterns, similar "
                "vulnerabilities, data flows, or code that handles the same concept.\n\n"
                "IMPORTANT: Write queries as natural descriptions of the CODE you want to find, "
                "not as lists of security keywords. The embedding model matches against actual "
                "source code, so describe what the code DOES.\n\n"
                "GOOD queries:\n"
                "  - 'function that builds a MongoDB query from request parameters'\n"
                "  - 'route handler that reads user input and passes it to eval or exec'\n"
                "  - 'middleware that checks authentication tokens or session cookies'\n"
                "  - 'string concatenation used to build an HTML response from user data'\n"
                "BAD queries (keyword soup — won't match well):\n"
                "  - 'SQL injection database query user input'\n"
                "  - 'XSS cross-site scripting template rendering'\n"
                "  - 'authentication bypass session management'"
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "Natural-language description of the code you want to find. "
                                "Describe what the code DOES, not what vulnerability category "
                                "it belongs to. Example: 'function that concatenates user input "
                                "into a database query string'"
                            ),
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results to return (default 10, max 30)",
                        },
                        "file_filter": {
                            "type": "string",
                            "description": "Optional: only return results from files matching this substring",
                        },
                        "chunk_type_filter": {
                            "type": "string",
                            "description": (
                                "Optional: filter by chunk type — function, class, method, "
                                "module_block, import_block, file_block"
                            ),
                        },
                    },
                    "required": ["query"],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "semantic_index_status",
            "description": (
                "Check the status of the semantic search index: how many chunks are indexed, "
                "which files, and whether the index is ready."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "semantic_index_files",
            "description": (
                "Add specific files to the semantic search index. Use when you want to "
                "ensure particular files are indexed before searching. The index is built "
                "automatically on first search, but this lets you add files incrementally."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "files": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of file paths (relative to repo root) to index",
                        },
                    },
                    "required": ["files"],
                }
            },
        }
    },
]


class ToolHandler:
    """Dispatches tool calls from Bedrock to actual implementations."""

    def __init__(self, config: ScanConfig) -> None:
        self.config = config
        self.repo_path = config.repo_path
        self._codeql_db_path: str = config.tools.codeql_db_path
        self._language: str = ""
        self._semantic_index: SemanticIndex | None = None
        self._embedding_client: EmbeddingProvider | None = None

    @property
    def language(self) -> str:
        if not self._language:
            self._language = detect_language(self.repo_path)
        return self._language

    def __call__(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        """Handle a tool call and return the result as a string."""
        handler = getattr(self, f"_handle_{tool_name}", None)
        if handler is None:
            return f"Unknown tool: {tool_name}"
        return handler(tool_input)

    # ---- individual handlers ------------------------------------------------

    def _handle_shell(self, inp: dict) -> str:
        result = run_shell(inp["command"], cwd=inp.get("cwd", self.repo_path))
        if result.returncode != 0:
            return f"STDERR:\n{result.stderr}\n\nSTDOUT:\n{result.stdout}"
        return result.stdout

    def _handle_read_file(self, inp: dict) -> str:
        return get_file_content(self.repo_path, inp["path"])

    def _handle_list_files(self, inp: dict) -> str:
        return get_repo_structure(self.repo_path, max_depth=inp.get("max_depth", 3))

    def _handle_grep_code(self, inp: dict) -> str:
        return grep_codebase(self.repo_path, inp["pattern"], inp.get("file_glob", ""))

    def _handle_git_diff(self, inp: dict) -> str:
        base = inp.get("base_ref", self.config.git_diff_base or "main")
        return get_diff(self.repo_path, base, inp.get("file_path", ""))

    def _handle_git_changed_files(self, inp: dict) -> str:
        base = inp.get("base_ref", self.config.git_diff_base or "main")
        files = get_changed_files(self.repo_path, base)
        return "\n".join(f"{f.status}\t{f.path}" for f in files) or "No changed files found."

    def _handle_git_log(self, inp: dict) -> str:
        return get_git_log(self.repo_path, n=inp.get("n", 20))

    def _handle_codeql_run(self, inp: dict) -> str:
        if not self._codeql_db_path:
            # Auto-create database.
            db = create_database(self.repo_path, self.language)
            if not db.created:
                return "Failed to create CodeQL database. Is codeql CLI installed?"
            self._codeql_db_path = db.path

        query = inp["query"]
        lang = inp.get("language", self.language)

        # Check if it's a built-in risk area key.
        from sast_agent.tools.codeql import BUILTIN_QUERIES
        if query in BUILTIN_QUERIES:
            result = run_builtin_query(self._codeql_db_path, query, lang)
        else:
            result = run_query(self._codeql_db_path, query)

        if not result.success:
            return f"CodeQL error: {result.error}"
        if not result.results:
            return "CodeQL query returned 0 results."
        return json.dumps(result.results, indent=2)

    def _handle_codeql_list_queries(self, inp: dict) -> str:
        lang = inp.get("language", self.language)
        queries = list_available_queries(lang)
        return "\n".join(queries) if queries else f"No built-in queries for {lang}"

    def _handle_codeql_query_raw(self, inp: dict) -> str:
        if not self.config.tools.codeql_enabled:
            return "CodeQL is disabled in this scan configuration."

        if not self._codeql_db_path:
            db = create_database(self.repo_path, self.language)
            if not db.created:
                return "Failed to create CodeQL database. Is codeql CLI installed?"
            self._codeql_db_path = db.path

        ql_source = inp["ql_source"]
        result = run_raw_query(self._codeql_db_path, ql_source)

        if not result.success:
            return f"CodeQL error: {result.error}"
        if not result.results:
            return "CodeQL query returned 0 results."
        return json.dumps(result.results, indent=2)

    def _handle_python_exec(self, inp: dict) -> str:
        if not self.config.tools.poc_generation_enabled:
            return "Python execution is disabled in this scan configuration."
        result = run_python(
            inp["code"],
            target_url=self.config.tools.target_url,
        )
        if result.blocked:
            return f"BLOCKED: {result.block_reason}"
        parts = []
        if result.stdout:
            parts.append(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            parts.append(f"STDERR:\n{result.stderr}")
        if not parts:
            parts.append("(no output)")
        return "\n\n".join(parts)

    # ---- semantic search handlers -------------------------------------------

    def _get_embedding_client(self) -> EmbeddingProvider:
        if self._embedding_client is None:
            self._embedding_client = create_embedding_from_config(self.config.embedding)
        return self._embedding_client

    def _get_semantic_index(self, auto_build: bool = True) -> SemanticIndex:
        """Get or lazily build the semantic index."""
        if self._semantic_index is not None:
            return self._semantic_index

        client = self._get_embedding_client()
        index = SemanticIndex(client)

        # Try loading a cached index.
        cache_path = self.config.tools.semantic_index_path
        if cache_path and index.load(cache_path):
            self._semantic_index = index
            return index

        # Build from scratch if requested.
        if auto_build:
            file_filter = self.config.target_files or None
            index = index_repository(self.repo_path, client, index, file_filter)
            if cache_path:
                try:
                    index.save(cache_path)
                except Exception:
                    pass  # non-fatal
            self._semantic_index = index

        return index

    def _handle_semantic_search(self, inp: dict) -> str:
        if not self.config.tools.semantic_search_enabled:
            return "Semantic search is disabled in this scan configuration."

        index = self._get_semantic_index(auto_build=True)
        if index.size == 0:
            return "Semantic index is empty — no files were indexed."

        top_k = min(inp.get("top_k", 10), 30)
        results = index.search(
            query=inp["query"],
            top_k=top_k,
            file_filter=inp.get("file_filter", ""),
            chunk_type_filter=inp.get("chunk_type_filter", ""),
        )
        return format_search_results(results)

    def _handle_semantic_index_status(self, inp: dict) -> str:
        if self._semantic_index is None:
            return "Semantic index not yet built. It will be created on first semantic_search call."

        index = self._semantic_index
        # Summarize indexed files.
        file_counts: dict[str, int] = {}
        type_counts: dict[str, int] = {}
        for chunk in index.chunks:
            file_counts[chunk.file_path] = file_counts.get(chunk.file_path, 0) + 1
            type_counts[chunk.chunk_type] = type_counts.get(chunk.chunk_type, 0) + 1

        lines = [
            f"Total chunks indexed: {index.size}",
            f"Files indexed: {len(file_counts)}",
            "",
            "Chunk types:",
        ]
        for ct, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            lines.append(f"  {ct}: {count}")

        lines.append("")
        lines.append("Top files by chunk count:")
        for fp, count in sorted(file_counts.items(), key=lambda x: -x[1])[:20]:
            lines.append(f"  {fp}: {count} chunks")

        return "\n".join(lines)

    def _handle_semantic_index_files(self, inp: dict) -> str:
        if not self.config.tools.semantic_search_enabled:
            return "Semantic search is disabled in this scan configuration."

        files = inp.get("files", [])
        if not files:
            return "No files specified."

        client = self._get_embedding_client()
        if self._semantic_index is None:
            self._semantic_index = SemanticIndex(client)

        index = index_repository(
            self.repo_path,
            client,
            self._semantic_index,
            file_filter=files,
        )
        self._semantic_index = index

        return f"Indexed {len(files)} file(s). Total chunks in index: {index.size}"
