"""Shared text utilities + progress bar helper."""

from __future__ import annotations

import re
import sys
import time
from contextlib import contextmanager

_THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL | re.IGNORECASE)


def strip_think(text: str) -> str:
    """Remove `<think>...</think>` blocks from reasoning-model output.

    Also handles the truncated case (`<think>...` with no closing tag) — common
    when `max_tokens` is exhausted during the reasoning phase. Leaves the
    visible answer (or empty string) intact.
    """
    if not text:
        return text
    stripped = _THINK_RE.sub("", text)
    return stripped.strip()


try:
    from alive_progress import alive_bar  # type: ignore
    _HAS_ALIVE = True
except ImportError:
    _HAS_ALIVE = False


def _fmt_secs(s: float) -> str:
    return f"{int(s // 60):02d}:{int(s % 60):02d}"


@contextmanager
def progress(total: int, title: str):
    """Yield a `step()` callable iterated `total` times.

    Uses `alive_progress` with the `waves2` bar style when available; the bar
    natively shows `[elapsed]` on the left and ETA on the right.
    Falls back to a plain stderr ticker (`[elapsed | ETA] title n/N`) otherwise.
    """
    if _HAS_ALIVE:
        # alive-progress: use default smooth bar; valid bars are smooth,
        # classic, classic2, brackets, blocks, bubbles, solid, checks,
        # circles, squares, halloween, filling, notes, ruler, ruler2,
        # fish, scuba.
        with alive_bar(total, title=title, bar="smooth", dual_line=False,
                       elapsed=True, stats=True) as bar:
            yield bar
        return

    start = time.monotonic()
    count = [0]

    def step(n: int = 1):
        count[0] += n
        elapsed = time.monotonic() - start
        eta = (elapsed / count[0]) * (total - count[0]) if count[0] else 0.0
        sys.stderr.write(
            f"\r[{_fmt_secs(elapsed)} eta {_fmt_secs(eta)}] {title} {count[0]}/{total}"
        )
        sys.stderr.flush()

    yield step
    sys.stderr.write("\n")
    sys.stderr.flush()
