#!/usr/bin/env python3
"""Check that the #include lines agree with the unit graph the build declares.

Every unit (cmake/Units.cmake) names the units it may use. This reads those
declarations out of src/CMakeLists.txt and app/CMakeLists.txt, then reads every
file they list, and reports

  * an include of a header owned by a unit this one does not depend on
    -- the rule is "if you include it, depend on it", so the graph in the build
    files is the whole truth and not a lower bound;
  * a header or source that no unit claims, or one claimed twice;
  * a cycle.

    python3 tools/check_units.py [repo root]

Exit status 0 when the graph and the includes agree, 1 otherwise.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

INCLUDE = re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]', re.M)
CALL = re.compile(r"^(sirius_lib_unit|sirius_core_unit|op)\s*\(", re.M)
SOURCE_EXT = (".cpp", ".cu")


def balanced(text: str, start: int) -> tuple[str, int]:
    """The argument text of the call whose '(' follows `start`, and its end."""
    i = text.index("(", start)
    depth, j = 0, i
    while j < len(text):
        if text[j] == "(":
            depth += 1
        elif text[j] == ")":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j], j
        j += 1
    raise SystemExit(f"unbalanced parentheses at offset {start}")


KEYWORDS = (
    "HEADERS",
    "SOURCES",
    "CUDA_SOURCES",
    "NVTIFF_SOURCES",
    "DEPENDS",
    "LINK",
    "CUDA_LINK",
    "NVTIFF_LINK",
    "PUBLIC_LINK",
    "DEFINES",
    "INCLUDE_PUBLIC",
    "INCLUDE_PRIVATE",
)


COMMENT = re.compile(r"#[^\n]*")


def parse_args(body: str) -> tuple[list[str], dict[str, list[str]]]:
    """Positional words before the first keyword, and the keyword lists."""
    words = COMMENT.sub("", body).split()
    positional, kw, current = [], {}, None
    for w in words:
        if w in KEYWORDS:
            current = kw.setdefault(w, [])
        elif current is None:
            positional.append(w)
        else:
            current.append(w)
    return positional, kw


def units_of(cmake: Path, group: str, file_root: Path, header_roots: dict[str, Path]):
    """{unit name: (files, direct dependencies)} from one CMakeLists."""
    text = cmake.read_text()
    out = {}
    for m in CALL.finditer(text):
        kind = m.group(1)
        body, _ = balanced(text, m.start())
        positional, kw = parse_args(body)
        if kind == "op":
            name, files, deps = "ops_" + positional[0], positional[1:], ["ops_factories", "ops_common"]
        else:
            name, files, deps = positional[0], [], []
        files += kw.get("HEADERS", []) + kw.get("SOURCES", []) + kw.get("CUDA_SOURCES", []) + kw.get("NVTIFF_SOURCES", [])
        deps += kw.get("DEPENDS", [])
        paths = []
        for f in files:
            root = file_root
            for prefix, r in header_roots.items():
                if f.startswith(prefix):
                    root, f = r, f[len(prefix) :]
                    break
            paths.append((root / f).resolve())
        out[name] = (paths, [d if "/" in d else f"{group}/{d}" for d in deps], kind == "op")
    return out


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    units: dict[str, tuple[list[Path], list[str]]] = {}
    operations: list[str] = []
    for name, (files, deps, _) in units_of(
        root / "src/CMakeLists.txt", "lib", root / "src", {"sirius/": root / "include/sirius"}
    ).items():
        units[f"lib/{name}"] = (files, deps)
    app = root / "app/CMakeLists.txt"
    if app.is_file():
        for name, (files, deps, is_op) in units_of(app, "core", root / "app/core", {}).items():
            units[f"core/{name}"] = (files, deps)
            if is_op:
                operations.append(f"core/{name}")
    # the registry names every operation through a variable the build fills in
    for unit, (files, deps) in units.items():
        if "core/${SIRIUS_APP_OPS}" in deps:
            units[unit] = (files, [d for d in deps if d != "core/${SIRIUS_APP_OPS}"] + operations)

    owner: dict[Path, str] = {}
    problems: list[str] = []
    for unit, (files, _) in units.items():
        for f in files:
            if f in owner:
                problems.append(f"{f.relative_to(root)}: claimed by both {owner[f]} and {unit}")
            owner[f] = unit

    for d, pattern in ((root / "include/sirius", "*.hpp"), (root / "app/core", "*"), (root / "src", "*")):
        for f in sorted(d.rglob(pattern)):
            if f.is_file() and f.suffix in (".hpp", ".cpp", ".cu", ".cuh") and f not in owner:
                problems.append(f"{f.relative_to(root)}: no unit claims it")

    # every project header a file can include, by the spellings that reach it
    by_spelling: dict[str, str] = {}
    for f, unit in owner.items():
        if f.suffix in SOURCE_EXT:
            continue
        rel = f.relative_to(root).as_posix()
        for spelling in (rel, rel.removeprefix("include/"), rel.removeprefix("src/"), rel.removeprefix("app/")):
            by_spelling.setdefault(spelling, unit)

    for unit, (files, deps) in sorted(units.items()):
        allowed = set(deps) | {unit}
        for f in files:
            if not f.is_file():
                problems.append(f"{unit}: {f.relative_to(root)} does not exist")
                continue
            for inc in INCLUDE.findall(f.read_text(errors="replace")):
                provider = by_spelling.get(inc)
                if provider is None or provider in allowed:
                    continue
                problems.append(f"{f.relative_to(root)}: includes {inc} ({provider}), which {unit} does not depend on")

    # cycles: the declarations are in dependency order, so a back edge is one
    seen: set[str] = set()
    for unit, (_, deps) in units.items():
        for d in deps:
            if d not in seen and d != unit:
                if d not in units:
                    problems.append(f"{unit}: depends on {d}, which is not a unit")
                else:
                    problems.append(f"{unit}: depends on {d}, which is declared after it (a cycle)")
        seen.add(unit)

    for p in problems:
        print(p)
    print(f"{len(units)} units, {len(owner)} files, {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
