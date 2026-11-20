"""Guard against touching held-out data (README "Rule 1"; the list is ../heldout.yaml).

    from curvefaker.heldout import check, check_path
    check("taskset/mlp_family_seed12/adam4p")      # passes
    check("synetune/lcbench/airlines")             # raises HeldOutError
"""
from __future__ import annotations

import fnmatch
from functools import lru_cache
from pathlib import Path

import yaml

LIST = Path(__file__).resolve().parent.parent / "heldout.yaml"


class HeldOutError(RuntimeError):
    pass


@lru_cache(maxsize=None)
def _rules():
    r = yaml.safe_load(LIST.read_text())
    pats = [(e["pattern"].lower(), e["family"]) for e in r["identifiers"]]
    paths = [Path(p).resolve() for p in r.get("paths", [])]
    return pats, paths


def reason(identifier: str) -> str | None:
    """The held-out family an identifier belongs to, or None if it is allowed."""
    ident = identifier.lower()
    for pat, fam in _rules()[0]:
        if fnmatch.fnmatchcase(ident, pat):
            return fam
    return None


def check(identifier: str) -> None:
    fam = reason(identifier)
    if fam is not None:
        raise HeldOutError(f"{identifier!r} is held out ({fam}); see heldout.yaml")


def check_path(path) -> None:
    p = Path(path).resolve()
    for bad in _rules()[1]:
        if p == bad or bad in p.parents:
            raise HeldOutError(f"{str(p)!r} is inside held-out directory {str(bad)!r}; see heldout.yaml")
