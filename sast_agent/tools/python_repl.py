"""Sandboxed Python REPL for PoC generation and analysis scripts.

When a target URL is configured (DAST mode), the sandbox unlocks ``requests``
but wraps it so only the target host can be reached.  All other network
modules remain blocked.
"""

from __future__ import annotations

import ast
import sys
import textwrap
from dataclasses import dataclass, field
from io import StringIO
from typing import Optional
from urllib.parse import urlparse

# Modules that are ALWAYS blocked regardless of DAST mode.
ALWAYS_BLOCKED = frozenset({
    "os",
    "subprocess",
    "shutil",
    "socket",
    "ctypes",
    "multiprocessing",
    "signal",
    "importlib",
    "builtins",
    "code",
    "codeop",
    "compile",
    "compileall",
    "py_compile",
    "webbrowser",
    "ftplib",
    "smtplib",
})

# Modules blocked in pure-SAST mode but unlocked in DAST mode.
NETWORK_MODULES = frozenset({
    "requests",
    "http",
    "urllib",
    "httpx",
    "aiohttp",
})


@dataclass
class ReplResult:
    code: str
    stdout: str
    stderr: str
    success: bool
    blocked: bool = False
    block_reason: str = ""


def _blocked_modules(dast_mode: bool) -> frozenset:
    """Return the set of blocked module roots for the current mode."""
    if dast_mode:
        # In DAST mode only 'requests' is unlocked; the rest stay blocked.
        return ALWAYS_BLOCKED | (NETWORK_MODULES - {"requests"})
    return ALWAYS_BLOCKED | NETWORK_MODULES


def _check_imports(code: str, blocked: frozenset) -> tuple[bool, str]:
    """Static check: reject code that imports blocked modules."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"Syntax error: {exc}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in blocked:
                    return False, f"Import of '{alias.name}' is blocked for safety"
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root = node.module.split(".")[0]
                if root in blocked:
                    return False, f"Import from '{node.module}' is blocked for safety"
    return True, ""


def _check_dangerous_calls(code: str) -> tuple[bool, str]:
    """Static check: reject obviously dangerous function calls."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return True, ""

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in ("eval", "exec", "compile", "__import__"):
                return False, f"Call to '{node.func.id}' is blocked for safety"
            if isinstance(node.func, ast.Name) and node.func.id == "open":
                if len(node.args) >= 2:
                    mode_arg = node.args[1]
                    if isinstance(mode_arg, ast.Constant) and isinstance(mode_arg.value, str):
                        if any(c in mode_arg.value for c in ("w", "a", "x")):
                            return False, "Writing files is blocked in PoC sandbox"
    return True, ""


# ---------------------------------------------------------------------------
# Scoped requests wrapper — only allows requests to the target host
# ---------------------------------------------------------------------------

class _ScopedSession:
    """A requests-like object that only allows calls to allowed hosts."""

    def __init__(self, allowed_origins: list[str]):
        import requests as _real_requests
        self._session = _real_requests.Session()
        self._allowed_origins = allowed_origins  # ["http://localhost:3000"]
        self._real = _real_requests

    def _check_url(self, url: str) -> None:
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        if origin not in self._allowed_origins:
            raise PermissionError(
                f"Request to '{origin}' blocked. "
                f"Only these targets are allowed: {self._allowed_origins}"
            )

    def get(self, url: str, **kwargs):
        self._check_url(url)
        kwargs.setdefault("timeout", 10)
        return self._session.get(url, **kwargs)

    def post(self, url: str, **kwargs):
        self._check_url(url)
        kwargs.setdefault("timeout", 10)
        return self._session.post(url, **kwargs)

    def put(self, url: str, **kwargs):
        self._check_url(url)
        kwargs.setdefault("timeout", 10)
        return self._session.put(url, **kwargs)

    def patch(self, url: str, **kwargs):
        self._check_url(url)
        kwargs.setdefault("timeout", 10)
        return self._session.patch(url, **kwargs)

    def delete(self, url: str, **kwargs):
        self._check_url(url)
        kwargs.setdefault("timeout", 10)
        return self._session.delete(url, **kwargs)

    def head(self, url: str, **kwargs):
        self._check_url(url)
        kwargs.setdefault("timeout", 10)
        return self._session.head(url, **kwargs)

    def options(self, url: str, **kwargs):
        self._check_url(url)
        kwargs.setdefault("timeout", 10)
        return self._session.options(url, **kwargs)

    def request(self, method: str, url: str, **kwargs):
        self._check_url(url)
        kwargs.setdefault("timeout", 10)
        return self._session.request(method, url, **kwargs)


class _ScopedRequestsModule:
    """Drop-in replacement for ``import requests`` that scopes all calls."""

    def __init__(self, allowed_origins: list[str]):
        self._allowed_origins = allowed_origins
        self._session = _ScopedSession(allowed_origins)

    def get(self, url, **kw):
        return self._session.get(url, **kw)

    def post(self, url, **kw):
        return self._session.post(url, **kw)

    def put(self, url, **kw):
        return self._session.put(url, **kw)

    def patch(self, url, **kw):
        return self._session.patch(url, **kw)

    def delete(self, url, **kw):
        return self._session.delete(url, **kw)

    def head(self, url, **kw):
        return self._session.head(url, **kw)

    def options(self, url, **kw):
        return self._session.options(url, **kw)

    def Session(self):
        return self._session


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

_SAFE_BUILTINS = {
    "print": print,
    "len": len,
    "range": range,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "sorted": sorted,
    "reversed": reversed,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "bytes": bytes,
    "bytearray": bytearray,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "hasattr": hasattr,
    "getattr": getattr,
    "type": type,
    "repr": repr,
    "chr": chr,
    "ord": ord,
    "hex": hex,
    "min": min,
    "max": max,
    "sum": sum,
    "abs": abs,
    "any": any,
    "all": all,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "IndexError": IndexError,
    "AttributeError": AttributeError,
    "Exception": Exception,
    "AssertionError": AssertionError,
    "PermissionError": PermissionError,
}


def run_python(
    code: str,
    target_url: str = "",
    timeout_hint: int = 30,
) -> ReplResult:
    """Execute Python code in a restricted sandbox.

    When ``target_url`` is set, ``import requests`` is available but scoped
    so it can ONLY reach the target host.  All other network access is blocked.
    """
    code = textwrap.dedent(code).strip()
    dast_mode = bool(target_url)
    blocked = _blocked_modules(dast_mode)

    # Static safety checks.
    ok, reason = _check_imports(code, blocked)
    if not ok:
        return ReplResult(code=code, stdout="", stderr=reason, success=False, blocked=True, block_reason=reason)

    ok, reason = _check_dangerous_calls(code)
    if not ok:
        return ReplResult(code=code, stdout="", stderr=reason, success=False, blocked=True, block_reason=reason)

    # Build execution globals.
    exec_globals: dict = {"__builtins__": dict(_SAFE_BUILTINS)}

    # In DAST mode, inject the scoped requests module.
    if dast_mode:
        parsed = urlparse(target_url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        try:
            scoped_requests = _ScopedRequestsModule([origin])
            exec_globals["__builtins__"]["__import__"] = _make_scoped_import(scoped_requests)
        except ImportError:
            return ReplResult(
                code=code, stdout="", stderr="'requests' package not installed",
                success=False, blocked=True, block_reason="requests not installed",
            )

    # Execute with captured stdout/stderr.
    old_stdout, old_stderr = sys.stdout, sys.stderr
    captured_out, captured_err = StringIO(), StringIO()
    sys.stdout, sys.stderr = captured_out, captured_err

    success = True
    try:
        exec(compile(code, "<poc>", "exec"), exec_globals)  # noqa: S102
    except PermissionError as exc:
        success = False
        captured_err.write(f"BLOCKED: {exc}\n")
    except Exception as exc:
        success = False
        captured_err.write(f"{type(exc).__name__}: {exc}\n")
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr

    return ReplResult(
        code=code,
        stdout=captured_out.getvalue()[:20_000],
        stderr=captured_err.getvalue()[:5_000],
        success=success,
    )


def _make_scoped_import(scoped_requests: _ScopedRequestsModule):
    """Return a __import__ replacement that intercepts ``import requests``."""
    import json as _json

    # Modules the sandbox is allowed to import (safe, no side effects).
    _ALLOWED_IMPORTS = {
        "json": _json,
        "re": __import__("re"),
        "base64": __import__("base64"),
        "hashlib": __import__("hashlib"),
        "hmac": __import__("hmac"),
        "urllib.parse": __import__("urllib.parse"),
        "html": __import__("html"),
        "string": __import__("string"),
        "textwrap": __import__("textwrap"),
        "math": __import__("math"),
        "collections": __import__("collections"),
        "itertools": __import__("itertools"),
        "functools": __import__("functools"),
        "copy": __import__("copy"),
        "datetime": __import__("datetime"),
        "time": __import__("time"),
    }

    def _scoped_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "requests" or name.startswith("requests."):
            return scoped_requests
        root = name.split(".")[0]
        if root in _ALLOWED_IMPORTS:
            return _ALLOWED_IMPORTS[root]
        if name in _ALLOWED_IMPORTS:
            return _ALLOWED_IMPORTS[name]
        raise ImportError(f"Import of '{name}' is not allowed in the sandbox")

    return _scoped_import
