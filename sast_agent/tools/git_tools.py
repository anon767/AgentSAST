"""Git-related tools for diff context and file discovery."""

from __future__ import annotations

from dataclasses import dataclass

from sast_agent.tools.shell import ShellResult, run_shell


@dataclass
class DiffFile:
    path: str
    status: str  # A, M, D, R
    additions: int = 0
    deletions: int = 0


def get_changed_files(repo_path: str, base_ref: str = "main") -> list[DiffFile]:
    """Get list of files changed compared to a base ref."""
    result = run_shell(
        f"git diff --name-status {base_ref}...HEAD",
        cwd=repo_path,
    )
    if result.returncode != 0:
        # Fallback: try without the triple-dot syntax.
        result = run_shell(
            f"git diff --name-status {base_ref}",
            cwd=repo_path,
        )

    files: list[DiffFile] = []
    for line in result.stdout.strip().splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            status = parts[0][0]  # first char: A, M, D, R
            path = parts[-1]
            files.append(DiffFile(path=path, status=status))
    return files


def get_diff(repo_path: str, base_ref: str = "main", file_path: str = "") -> str:
    """Get the unified diff for a file or the whole PR."""
    file_arg = f"-- {file_path}" if file_path else ""
    result = run_shell(
        f"git diff {base_ref}...HEAD {file_arg}",
        cwd=repo_path,
    )
    if result.returncode != 0:
        result = run_shell(
            f"git diff {base_ref} {file_arg}",
            cwd=repo_path,
        )
    return result.stdout


def get_file_content(repo_path: str, file_path: str) -> str:
    """Read a file from the repo."""
    result = run_shell(f"cat {file_path}", cwd=repo_path)
    return result.stdout if result.returncode == 0 else f"ERROR: {result.stderr}"


def get_repo_structure(repo_path: str, max_depth: int = 3) -> str:
    """Get a tree view of the repository."""
    result = run_shell(
        f"find . -maxdepth {max_depth} -type f "
        f"-not -path './.git/*' "
        f"-not -path './node_modules/*' "
        f"-not -path './__pycache__/*' "
        f"-not -path './venv/*' "
        f"-not -path './.venv/*' "
        f"-not -path './dist/*' "
        f"-not -path './build/*' "
        f"| head -500 | sort",
        cwd=repo_path,
    )
    return result.stdout


def get_git_log(repo_path: str, n: int = 20) -> str:
    """Get recent git log."""
    result = run_shell(
        f"git log --oneline -n {n}",
        cwd=repo_path,
    )
    return result.stdout


def grep_codebase(repo_path: str, pattern: str, file_glob: str = "") -> str:
    """Search the codebase for a pattern."""
    glob_arg = f"--include='{file_glob}'" if file_glob else ""
    result = run_shell(
        f"grep -rn {glob_arg} '{pattern}' . "
        f"--exclude-dir=.git "
        f"--exclude-dir=node_modules "
        f"--exclude-dir=__pycache__ "
        f"--exclude-dir=venv "
        f"| head -100",
        cwd=repo_path,
    )
    return result.stdout
