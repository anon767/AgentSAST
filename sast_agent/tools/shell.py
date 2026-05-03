"""Shell execution tool with safety guardrails."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

# Commands that are never allowed regardless of context.
BLOCKED_COMMANDS = [
    "rm -rf /",
    "mkfs",
    "dd if=",
    ":(){",
    "fork bomb",
    "curl | sh",
    "curl | bash",
    "wget | sh",
    "wget | bash",
    "> /dev/sda",
    "chmod -R 777 /",
    "shutdown",
    "reboot",
    "halt",
    "poweroff",
    "init 0",
    "init 6",
]

# Commands allowed for SAST analysis.
ALLOWED_PREFIXES = [
    "ls",
    "find",
    "cat",
    "head",
    "tail",
    "wc",
    "grep",
    "rg",
    "ag",
    "git",
    "codeql",
    "python",
    "semgrep",
    "tree",
    "file",
    "stat",
    "diff",
    "sort",
    "uniq",
    "awk",
    "sed",
    "jq",
    "echo",
    "printf",
    "basename",
    "dirname",
    "realpath",
    "readlink",
    "xargs",
]


@dataclass
class ShellResult:
    command: str
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool = False


def is_command_safe(command: str) -> tuple[bool, str]:
    """Check whether a shell command is safe to execute.

    Returns (is_safe, reason).
    """
    cmd_lower = command.lower().strip()

    for blocked in BLOCKED_COMMANDS:
        if blocked in cmd_lower:
            return False, f"Blocked dangerous pattern: {blocked}"

    # Extract the base command (first token, ignoring env vars).
    tokens = cmd_lower.split()
    base_cmd = ""
    for token in tokens:
        if "=" in token:
            continue  # skip env var assignments
        base_cmd = token.split("/")[-1]  # handle absolute paths
        break

    if not base_cmd:
        return False, "Could not determine base command"

    if base_cmd not in ALLOWED_PREFIXES:
        return False, f"Command '{base_cmd}' is not in the allow-list"

    return True, "ok"


def run_shell(
    command: str,
    cwd: str = ".",
    timeout: int = 60,
    check_safety: bool = True,
) -> ShellResult:
    """Execute a shell command with safety checks and timeout."""

    if check_safety:
        safe, reason = is_command_safe(command)
        if not safe:
            return ShellResult(
                command=command,
                stdout="",
                stderr=f"BLOCKED: {reason}",
                returncode=-1,
            )

    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return ShellResult(
            command=command,
            stdout=proc.stdout[:50_000],  # cap output size
            stderr=proc.stderr[:10_000],
            returncode=proc.returncode,
        )
    except subprocess.TimeoutExpired:
        return ShellResult(
            command=command,
            stdout="",
            stderr=f"Command timed out after {timeout}s",
            returncode=-1,
            timed_out=True,
        )
    except Exception as exc:
        return ShellResult(
            command=command,
            stdout="",
            stderr=str(exc),
            returncode=-1,
        )
