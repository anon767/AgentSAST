"""Semantic code search using tree-sitter AST chunking, Bedrock Titan embeddings, and FAISS.

Architecture:
  1. Tree-sitter parses source files into real ASTs for all supported languages,
     extracting functions, classes, methods, and top-level blocks with accurate
     line ranges and parent scoping.  Python files use the stdlib ``ast`` module
     for richer signature extraction.
  2. Embeddings are generated via Bedrock Titan Embeddings V2.
  3. FAISS flat-IP index stores the vectors for fast similarity search.
  4. Query interface lets agents search by natural-language description
     or code snippet and get back ranked, contextual results.
  5. tqdm progress bars track file chunking and embedding progress.
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tree-sitter language registry
# ---------------------------------------------------------------------------

# Mapping from tree-sitter language module to the callable that returns the
# language pointer.  We lazy-import so missing grammars don't crash the tool.
_TS_LANGUAGE_MODULES: Dict[str, str] = {
    "javascript":  "tree_sitter_javascript",
    "typescript":  "tree_sitter_typescript",  # has .language_typescript()
    "tsx":         "tree_sitter_typescript",  # has .language_tsx()
    "java":        "tree_sitter_java",
    "go":          "tree_sitter_go",
    "ruby":        "tree_sitter_ruby",
    "rust":        "tree_sitter_rust",
    "c":           "tree_sitter_c",
    "cpp":         "tree_sitter_cpp",
    "c_sharp":     "tree_sitter_c_sharp",
    "python":      "tree_sitter_python",
}

# Node types we consider "definition" nodes per language.
# Each entry is (node_type, chunk_type_label).
_DEFINITION_NODE_TYPES: Dict[str, List[Tuple[str, str]]] = {
    "javascript": [
        ("function_declaration", "function"),
        ("generator_function_declaration", "function"),
        ("class_declaration", "class"),
        ("method_definition", "method"),
        ("export_statement", "export"),       # unwrap later
        ("lexical_declaration", "variable"),   # const/let arrow fns
        ("variable_declaration", "variable"),  # var arrow fns
    ],
    "typescript": [
        ("function_declaration", "function"),
        ("class_declaration", "class"),
        ("method_definition", "method"),
        ("interface_declaration", "interface"),
        ("type_alias_declaration", "type"),
        ("enum_declaration", "enum"),
        ("export_statement", "export"),
        ("lexical_declaration", "variable"),
    ],
    "tsx": [
        ("function_declaration", "function"),
        ("class_declaration", "class"),
        ("method_definition", "method"),
        ("interface_declaration", "interface"),
        ("type_alias_declaration", "type"),
        ("export_statement", "export"),
        ("lexical_declaration", "variable"),
    ],
    "java": [
        ("class_declaration", "class"),
        ("interface_declaration", "interface"),
        ("enum_declaration", "enum"),
        ("method_declaration", "method"),
        ("constructor_declaration", "constructor"),
    ],
    "go": [
        ("function_declaration", "function"),
        ("method_declaration", "method"),
        ("type_declaration", "type"),
    ],
    "ruby": [
        ("method", "function"),
        ("singleton_method", "function"),
        ("class", "class"),
        ("module", "module"),
    ],
    "rust": [
        ("function_item", "function"),
        ("impl_item", "class"),
        ("struct_item", "class"),
        ("enum_item", "enum"),
        ("trait_item", "interface"),
    ],
    "c": [
        ("function_definition", "function"),
        ("struct_specifier", "class"),
        ("enum_specifier", "enum"),
    ],
    "cpp": [
        ("function_definition", "function"),
        ("class_specifier", "class"),
        ("struct_specifier", "class"),
        ("enum_specifier", "enum"),
        ("namespace_definition", "namespace"),
    ],
    "c_sharp": [
        ("class_declaration", "class"),
        ("interface_declaration", "interface"),
        ("method_declaration", "method"),
        ("constructor_declaration", "constructor"),
        ("enum_declaration", "enum"),
        ("namespace_declaration", "namespace"),
    ],
    "python": [
        ("function_definition", "function"),
        ("class_definition", "class"),
    ],
}

# File extensions → tree-sitter language key.
PARSEABLE_EXTENSIONS: Dict[str, str] = {
    ".py":   "python",
    ".js":   "javascript",
    ".mjs":  "javascript",
    ".cjs":  "javascript",
    ".jsx":  "javascript",
    ".ts":   "typescript",
    ".tsx":  "tsx",
    ".java": "java",
    ".go":   "go",
    ".rb":   "ruby",
    ".rs":   "rust",
    ".c":    "c",
    ".h":    "c",
    ".cpp":  "cpp",
    ".cc":   "cpp",
    ".cxx":  "cpp",
    ".hpp":  "cpp",
    ".hxx":  "cpp",
    ".cs":   "c_sharp",
}

MAX_CHUNK_CHARS = 3000
MIN_CHUNK_CHARS = 40


# ---------------------------------------------------------------------------
# CodeChunk dataclass
# ---------------------------------------------------------------------------

@dataclass
class CodeChunk:
    """A semantically meaningful unit of code extracted from a source file."""

    file_path: str
    language: str
    chunk_type: str   # function, class, method, module_block, import_block, etc.
    name: str
    start_line: int
    end_line: int
    source: str
    signature: str = ""
    parent: str = ""
    docstring: str = ""
    hash: str = ""

    def __post_init__(self) -> None:
        if not self.hash:
            self.hash = hashlib.sha256(self.source.encode()).hexdigest()[:16]

    @property
    def embedding_text(self) -> str:
        """Text sent to the embedding model — enriched with metadata."""
        parts = [f"# {self.chunk_type}: {self.qualified_name}"]
        if self.signature:
            parts.append(self.signature)
        if self.docstring:
            parts.append(f'"""{self.docstring}"""')
        parts.append(self.source[:MAX_CHUNK_CHARS])
        parts.append(f"# file: {self.file_path}  lines: {self.start_line}-{self.end_line}")
        return "\n".join(parts)

    @property
    def qualified_name(self) -> str:
        return f"{self.parent}.{self.name}" if self.parent else self.name

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "language": self.language,
            "chunk_type": self.chunk_type,
            "name": self.name,
            "qualified_name": self.qualified_name,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "signature": self.signature,
            "parent": self.parent,
            "docstring": self.docstring[:200],
            "hash": self.hash,
            "source": self.source,
        }


# ---------------------------------------------------------------------------
# Tree-sitter parser cache
# ---------------------------------------------------------------------------

_parser_cache: Dict[str, Any] = {}  # lang_key -> (Parser, Language)


def _get_ts_parser(lang_key: str):
    """Return a (parser, language) tuple for the given language, or None."""
    if lang_key in _parser_cache:
        return _parser_cache[lang_key]

    try:
        import tree_sitter as ts
        mod_name = _TS_LANGUAGE_MODULES.get(lang_key)
        if not mod_name:
            _parser_cache[lang_key] = None
            return None

        import importlib
        mod = importlib.import_module(mod_name)

        # tree_sitter_typescript exposes language_typescript() and language_tsx()
        if lang_key == "tsx" and hasattr(mod, "language_tsx"):
            lang_ptr = mod.language_tsx()
        elif lang_key == "typescript" and hasattr(mod, "language_typescript"):
            lang_ptr = mod.language_typescript()
        else:
            lang_ptr = mod.language()

        language = ts.Language(lang_ptr)
        parser = ts.Parser()
        parser.language = language
        _parser_cache[lang_key] = (parser, language)
        return _parser_cache[lang_key]
    except Exception as exc:
        logger.debug("tree-sitter unavailable for %s: %s", lang_key, exc)
        _parser_cache[lang_key] = None
        return None


# ---------------------------------------------------------------------------
# Tree-sitter chunking
# ---------------------------------------------------------------------------

def _node_text(node, source_bytes: bytes) -> str:
    """Extract the source text for a tree-sitter node."""
    return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _extract_name_from_node(node) -> str:
    """Try to extract a human-readable name from a tree-sitter node."""
    # Most definition nodes have a direct 'name' child.
    for child in node.children:
        if child.type in ("identifier", "property_identifier", "type_identifier",
                          "name", "constant"):
            return child.text.decode("utf-8", errors="replace")
    # For export_statement, dig into the declaration child.
    if node.type == "export_statement":
        for child in node.children:
            if child.type != "export_statement":
                name = _extract_name_from_node(child)
                if name:
                    return name
    # For variable/lexical declarations, look for the variable_declarator.
    if node.type in ("lexical_declaration", "variable_declaration"):
        for child in node.children:
            if child.type == "variable_declarator":
                return _extract_name_from_node(child)
    return ""


def _extract_signature_from_node(node, source_bytes: bytes) -> str:
    """Extract the first line (signature) of a definition node."""
    text = _node_text(node, source_bytes)
    first_line = text.split("\n")[0].strip()
    # Cap at a reasonable length.
    if len(first_line) > 200:
        first_line = first_line[:200] + "..."
    return first_line


def _find_parent_class(node) -> str:
    """Walk up the tree to find an enclosing class/impl/struct name."""
    parent = node.parent
    while parent is not None:
        if parent.type in ("class_declaration", "class_definition", "class_specifier",
                           "impl_item", "class", "interface_declaration",
                           "struct_item", "struct_specifier", "module"):
            return _extract_name_from_node(parent)
        parent = parent.parent
    return ""


def _unwrap_export(node):
    """If node is an export_statement, return the inner declaration."""
    if node.type == "export_statement":
        for child in node.children:
            if child.type not in ("export", "default", "comment", "{", "}", ";"):
                return child
    return node


def chunk_with_treesitter(
    file_path: str,
    source: str,
    lang_key: str,
) -> List[CodeChunk]:
    """Parse a file with tree-sitter and extract definition chunks."""
    ts_result = _get_ts_parser(lang_key)
    if ts_result is None:
        return _chunk_fallback(file_path, source, lang_key)

    parser, language = ts_result
    source_bytes = source.encode("utf-8")
    tree = parser.parse(source_bytes)
    root = tree.root_node

    definition_types = {
        nt for nt, _ in _DEFINITION_NODE_TYPES.get(lang_key, [])
    }
    type_to_label = {
        nt: label for nt, label in _DEFINITION_NODE_TYPES.get(lang_key, [])
    }

    chunks: List[CodeChunk] = []
    visited_ranges: Set[Tuple[int, int]] = set()

    def _walk(node, depth: int = 0):
        """Recursively walk the AST and extract definition nodes."""
        if node.type in definition_types:
            actual_node = _unwrap_export(node) if node.type == "export_statement" else node
            chunk_type = type_to_label.get(actual_node.type, type_to_label.get(node.type, "block"))

            # For variable/lexical declarations, check if they contain arrow functions.
            if chunk_type == "variable" and actual_node.type in ("lexical_declaration", "variable_declaration"):
                text = _node_text(actual_node, source_bytes)
                if "=>" not in text and "function" not in text:
                    # Plain variable, not interesting for chunking — skip.
                    return

            start_line = node.start_point[0] + 1  # tree-sitter is 0-indexed
            end_line = node.end_point[0] + 1
            range_key = (start_line, end_line)

            if range_key in visited_ranges:
                return
            visited_ranges.add(range_key)

            text = _node_text(node, source_bytes)
            if len(text.strip()) < MIN_CHUNK_CHARS:
                return
            if len(text) > MAX_CHUNK_CHARS:
                text = text[:MAX_CHUNK_CHARS] + "\n// ... truncated"

            name = _extract_name_from_node(actual_node) or f"anon_{start_line}"
            signature = _extract_signature_from_node(actual_node, source_bytes)
            parent_name = _find_parent_class(node)

            # Reclassify: if it has a parent class, it's a method.
            if parent_name and chunk_type == "function":
                chunk_type = "method"

            chunks.append(CodeChunk(
                file_path=file_path,
                language=lang_key,
                chunk_type=chunk_type,
                name=name,
                start_line=start_line,
                end_line=end_line,
                source=text,
                signature=signature,
                parent=parent_name,
            ))

        # Recurse into children — but don't recurse into nodes we already
        # extracted (their children are part of the chunk).
        if (node.start_point[0] + 1, node.end_point[0] + 1) not in visited_ranges or node == root:
            for child in node.children:
                _walk(child, depth + 1)

    _walk(root)

    # Catch uncovered top-level code.
    source_lines = source.splitlines()
    covered: Set[int] = set()
    for c in chunks:
        covered.update(range(c.start_line, c.end_line + 1))

    uncovered = _find_contiguous_blocks(source_lines, covered)
    for i, (start, end) in enumerate(uncovered):
        src = "\n".join(source_lines[start - 1:end])
        if len(src.strip()) >= MIN_CHUNK_CHARS:
            chunks.append(CodeChunk(
                file_path=file_path,
                language=lang_key,
                chunk_type="module_block",
                name=f"block_{i}",
                start_line=start,
                end_line=end,
                source=src[:MAX_CHUNK_CHARS],
            ))

    return chunks


# ---------------------------------------------------------------------------
# Python-specific chunking (stdlib ast — richer signatures than tree-sitter)
# ---------------------------------------------------------------------------

def _get_source_segment(source_lines: List[str], start: int, end: int) -> str:
    return "\n".join(source_lines[start - 1:end])


def _extract_signature_py(node: ast.AST) -> str:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        try:
            arg_parts = []
            for arg in node.args.args:
                ann = ""
                if arg.annotation:
                    ann = f": {ast.unparse(arg.annotation)}"
                arg_parts.append(f"{arg.arg}{ann}")
            returns = ""
            if node.returns:
                returns = f" -> {ast.unparse(node.returns)}"
            prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
            return f"{prefix} {node.name}({', '.join(arg_parts)}){returns}"
        except Exception:
            return f"def {node.name}(...)"
    if isinstance(node, ast.ClassDef):
        bases = ", ".join(ast.unparse(b) for b in node.bases) if node.bases else ""
        return f"class {node.name}({bases})" if bases else f"class {node.name}"
    return ""


def _extract_docstring_py(node: ast.AST) -> str:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            return node.body[0].value.value.strip()
    return ""


def chunk_python(file_path: str, source: str) -> List[CodeChunk]:
    """AST-based chunking for Python files using the stdlib ast module."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return _chunk_fallback(file_path, source, "python")

    source_lines = source.splitlines()
    chunks: List[CodeChunk] = []

    # Import block.
    import_lines: List[int] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            import_lines.extend(range(node.lineno, node.end_lineno + 1))
    if import_lines:
        start, end = min(import_lines), max(import_lines)
        chunks.append(CodeChunk(
            file_path=file_path, language="python", chunk_type="import_block",
            name="imports", start_line=start, end_line=end,
            source=_get_source_segment(source_lines, start, end),
        ))

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            src = _get_source_segment(source_lines, node.lineno, node.end_lineno)
            chunks.append(CodeChunk(
                file_path=file_path, language="python", chunk_type="function",
                name=node.name, start_line=node.lineno, end_line=node.end_lineno,
                source=src, signature=_extract_signature_py(node),
                docstring=_extract_docstring_py(node),
            ))
        elif isinstance(node, ast.ClassDef):
            class_src = _get_source_segment(source_lines, node.lineno, node.end_lineno)
            chunks.append(CodeChunk(
                file_path=file_path, language="python", chunk_type="class",
                name=node.name, start_line=node.lineno, end_line=node.end_lineno,
                source=class_src, signature=_extract_signature_py(node),
                docstring=_extract_docstring_py(node),
            ))
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_src = _get_source_segment(source_lines, item.lineno, item.end_lineno)
                    chunks.append(CodeChunk(
                        file_path=file_path, language="python", chunk_type="method",
                        name=item.name, start_line=item.lineno, end_line=item.end_lineno,
                        source=method_src, signature=_extract_signature_py(item),
                        parent=node.name, docstring=_extract_docstring_py(item),
                    ))

    covered: Set[int] = set()
    for c in chunks:
        covered.update(range(c.start_line, c.end_line + 1))
    for i, (start, end) in enumerate(_find_contiguous_blocks(source_lines, covered)):
        src = _get_source_segment(source_lines, start, end)
        if len(src.strip()) >= MIN_CHUNK_CHARS:
            chunks.append(CodeChunk(
                file_path=file_path, language="python", chunk_type="module_block",
                name=f"block_{i}", start_line=start, end_line=end, source=src,
            ))
    return chunks


# ---------------------------------------------------------------------------
# Fallback chunker (fixed-size overlapping windows)
# ---------------------------------------------------------------------------

def _find_contiguous_blocks(
    source_lines: List[str], covered: Set[int],
) -> List[Tuple[int, int]]:
    blocks: List[Tuple[int, int]] = []
    current_start = None
    for i in range(1, len(source_lines) + 1):
        if i not in covered and source_lines[i - 1].strip():
            if current_start is None:
                current_start = i
        else:
            if current_start is not None:
                blocks.append((current_start, i - 1))
                current_start = None
    if current_start is not None:
        blocks.append((current_start, len(source_lines)))
    return blocks


def _chunk_fallback(file_path: str, source: str, language: str) -> List[CodeChunk]:
    source_lines = source.splitlines()
    chunks: List[CodeChunk] = []
    window, stride = 60, 45
    for i in range(0, len(source_lines), stride):
        end = min(i + window, len(source_lines))
        src = "\n".join(source_lines[i:end])
        if len(src.strip()) >= MIN_CHUNK_CHARS:
            chunks.append(CodeChunk(
                file_path=file_path, language=language, chunk_type="file_block",
                name=f"block_{i // stride}", start_line=i + 1, end_line=end,
                source=src,
            ))
        if end >= len(source_lines):
            break
    return chunks


# ---------------------------------------------------------------------------
# Unified chunker entry point
# ---------------------------------------------------------------------------

def chunk_file(file_path: str, source: str) -> List[CodeChunk]:
    """Route to the best available chunker for this file type."""
    ext = Path(file_path).suffix.lower()
    lang_key = PARSEABLE_EXTENSIONS.get(ext, "")

    if not lang_key:
        return _chunk_fallback(file_path, source, "unknown")

    # Python: use stdlib ast for richer signatures.
    if lang_key == "python":
        return chunk_python(file_path, source)

    # Everything else: tree-sitter (falls back to sliding window if grammar missing).
    return chunk_with_treesitter(file_path, source, lang_key)


# ---------------------------------------------------------------------------
# FAISS Index
# ---------------------------------------------------------------------------

class SemanticIndex:
    """FAISS-backed semantic search index over code chunks."""

    def __init__(self, embedding_client) -> None:
        """Accept any object with .dim, .embed_text(), .embed_batch() methods."""
        import faiss
        self.embedding_client = embedding_client
        self.dim = embedding_client.dim
        self.index = faiss.IndexFlatIP(self.dim)
        self.chunks: List[CodeChunk] = []
        self._chunk_hashes: Set[str] = set()

    @property
    def size(self) -> int:
        return self.index.ntotal

    def add_chunks(self, chunks: List[CodeChunk], show_progress: bool = True) -> int:
        """Embed and add chunks to the index."""
        new_chunks = [c for c in chunks if c.hash not in self._chunk_hashes]
        if not new_chunks:
            return 0

        texts = [c.embedding_text for c in new_chunks]
        vectors = self.embedding_client.embed_batch(texts, show_progress=show_progress)
        self.index.add(vectors)

        for c in new_chunks:
            self.chunks.append(c)
            self._chunk_hashes.add(c.hash)

        logger.info("Added %d chunks to index (total: %d)", len(new_chunks), self.size)
        return len(new_chunks)

    def search(
        self,
        query: str,
        top_k: int = 10,
        file_filter: str = "",
        chunk_type_filter: str = "",
    ) -> List[Tuple[CodeChunk, float]]:
        if self.size == 0:
            return []

        query_vec = self.embedding_client.embed_text(query).reshape(1, -1)
        search_k = min(top_k * 3, self.size)
        scores, indices = self.index.search(query_vec, search_k)

        results: List[Tuple[CodeChunk, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self.chunks[idx]
            if file_filter and file_filter not in chunk.file_path:
                continue
            if chunk_type_filter and chunk.chunk_type != chunk_type_filter:
                continue
            results.append((chunk, float(score)))
            if len(results) >= top_k:
                break
        return results

    def save(self, path: str) -> None:
        import faiss
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        faiss.write_index(self.index, f"{path}.faiss")
        with open(f"{path}.meta.json", "w") as f:
            json.dump([c.to_dict() for c in self.chunks], f)
        logger.info("Saved index to %s (%d chunks)", path, self.size)

    def load(self, path: str) -> bool:
        import faiss
        faiss_path = f"{path}.faiss"
        meta_path = f"{path}.meta.json"
        if not os.path.exists(faiss_path) or not os.path.exists(meta_path):
            return False

        self.index = faiss.read_index(faiss_path)
        with open(meta_path) as f:
            metadata = json.load(f)

        self.chunks = []
        self._chunk_hashes = set()
        for m in metadata:
            chunk = CodeChunk(
                file_path=m["file_path"], language=m["language"],
                chunk_type=m["chunk_type"], name=m["name"],
                start_line=m["start_line"], end_line=m["end_line"],
                source=m["source"], signature=m.get("signature", ""),
                parent=m.get("parent", ""), docstring=m.get("docstring", ""),
                hash=m.get("hash", ""),
            )
            self.chunks.append(chunk)
            self._chunk_hashes.add(chunk.hash)

        logger.info("Loaded index from %s (%d chunks)", path, self.size)
        return True


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

INDEXABLE_EXTENSIONS = set(PARSEABLE_EXTENSIONS.keys()) | {
    ".yml", ".yaml", ".json", ".toml", ".xml", ".sql", ".graphql", ".gql",
    ".proto", ".tf", ".hcl", ".sh", ".bash", ".dockerfile", ".md",
}

SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", "venv", ".venv", "dist", "build",
    ".tox", ".mypy_cache", ".pytest_cache", "target", "vendor", ".next",
    "coverage", ".coverage", "htmlcov",
}


def index_repository(
    repo_path: str,
    embedding_client,
    index: Optional[SemanticIndex] = None,
    file_filter: Optional[List[str]] = None,
) -> SemanticIndex:
    """Walk a repository, chunk all source files, and build a FAISS index.

    Shows tqdm progress bars for both file chunking and embedding.
    """
    if index is None:
        index = SemanticIndex(embedding_client)

    repo = Path(repo_path).resolve()
    files_to_index: List[Path] = []

    if file_filter:
        for f in file_filter:
            p = repo / f
            if p.is_file():
                files_to_index.append(p)
    else:
        for root, dirs, files in os.walk(repo):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for fname in files:
                fpath = Path(root) / fname
                if fpath.suffix.lower() in INDEXABLE_EXTENSIONS:
                    files_to_index.append(fpath)

    logger.info("Indexing %d files from %s", len(files_to_index), repo_path)

    # Phase 1: chunk all files (with progress bar).
    all_chunks: List[CodeChunk] = []
    for fpath in tqdm(files_to_index, desc="Chunking files", unit="file"):
        try:
            source = fpath.read_text(errors="replace")
        except Exception as exc:
            logger.warning("Could not read %s: %s", fpath, exc)
            continue
        rel_path = str(fpath.relative_to(repo))
        all_chunks.extend(chunk_file(rel_path, source))

    # Phase 2: embed and index (progress bar inside add_chunks).
    added = index.add_chunks(all_chunks, show_progress=True)
    logger.info("Indexed %d new chunks (total: %d) from %d files",
                added, index.size, len(files_to_index))
    return index


def format_search_results(
    results: List[Tuple[CodeChunk, float]], max_source_lines: int = 30,
) -> str:
    if not results:
        return "No results found."

    parts: List[str] = []
    for i, (chunk, score) in enumerate(results, 1):
        source_preview = chunk.source
        lines = source_preview.splitlines()
        if len(lines) > max_source_lines:
            source_preview = (
                "\n".join(lines[:max_source_lines])
                + f"\n... ({len(lines) - max_source_lines} more lines)"
            )
        parts.append(
            f"--- Result {i} (score: {score:.4f}) ---\n"
            f"File: {chunk.file_path}  Lines: {chunk.start_line}-{chunk.end_line}\n"
            f"Type: {chunk.chunk_type}  Name: {chunk.qualified_name}\n"
            f"Signature: {chunk.signature}\n"
            f"```\n{source_preview}\n```\n"
        )
    return "\n".join(parts)
