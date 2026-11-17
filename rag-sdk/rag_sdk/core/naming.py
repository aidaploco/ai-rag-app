from collections.abc import Callable


def metric_base(fn: Callable) -> str:
    # OTel names are capped at 63 chars; qualnames with <locals> can be arbitrarily long.
    # Keep only the last two meaningful parts (skipping <locals>, <lambda>, etc.).
    parts = [p for p in fn.__qualname__.split(".") if not p.startswith("<")]
    return ".".join(parts[-2:] if len(parts) >= 2 else parts)
