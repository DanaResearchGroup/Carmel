# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0

"""API key discovery for Carmel's agentic layer.

Requiring an operator to hand-export an API key before every run (e.g.
``set -a; . ~/.config/google/env; set +a``) is bad ergonomics when the key already sits
on disk in a well-known place. This module mirrors the credential-discovery idiom used
by the sister project ARC (``arc/settings/settings.py::find_executable``): build an
ORDERED list of candidate locations, try each in turn, first hit wins.

Precedence, first hit wins:

1. ``os.environ[env_var]`` -- an explicitly exported value ALWAYS wins. Nothing found on
   disk may silently override a value the operator deliberately set in their shell.
2. Carmel's own credential files, in order:
   - ``$CARMEL_HOME/credentials.env`` (only when ``CARMEL_HOME`` is set)
   - ``~/.carmel/credentials.env``
   - ``~/.config/carmel/env``
3. Provider-conventional locations (only when ``provider`` is given), e.g.
   ``~/.config/google/env`` for ``google``.

SECURITY: this module handles live API keys. Key VALUES must never be logged, printed,
``repr``'d, or embedded in any exception message -- only variable names, file paths, and
character counts may appear in diagnostics. See :func:`resolve_api_key` and
:func:`_read_env_file` for where this discipline is enforced.
"""

from __future__ import annotations

import os
from pathlib import Path

from carmel.logger import get_logger

logger = get_logger("credentials")

__all__ = ["credential_search_path", "resolve_api_key"]

#: File name Carmel's own credential files use, wherever they live.
_CARMEL_CREDENTIALS_FILENAME = "credentials.env"

#: Provider-conventional credential file locations, keyed by ``AgentProvider.value``.
#: Deliberately small and explicit -- each entry documents the one file that provider's
#: own tooling (or a prior manual setup) is known to write, so this table does not grow
#: into a generic "search everywhere" scan.
_PROVIDER_CREDENTIAL_PATHS: dict[str, tuple[str, ...]] = {
    # The original manual workaround this module replaces: `. ~/.config/google/env`.
    "google": ("~/.config/google/env",),
    "openai": ("~/.config/openai/env",),
    "deepseek": ("~/.config/deepseek/env",),
}


def _carmel_credential_candidates() -> list[Path]:
    """Return Carmel's own credential-file candidates, in precedence order."""
    candidates: list[Path] = []
    carmel_home = os.environ.get("CARMEL_HOME")
    if carmel_home:
        candidates.append(Path(carmel_home).expanduser() / _CARMEL_CREDENTIALS_FILENAME)
    candidates.append(Path("~/.carmel").expanduser() / _CARMEL_CREDENTIALS_FILENAME)
    candidates.append(Path("~/.config/carmel/env").expanduser())
    return candidates


def credential_search_path(env_var: str, *, provider: str | None = None) -> list[Path]:
    """Return the ordered list of files consulted when resolving ``env_var``.

    Does NOT include ``os.environ`` (that is not a file), only the on-disk candidates
    that :func:`resolve_api_key` falls back to when the environment variable is unset.
    Exposed separately so a "key not found" error can name exactly where Carmel looked
    -- a search failure that does not say where it searched is not actionable.

    Args:
        env_var: The environment variable name the key would normally be exported as
            (unused for path construction today, but accepted for symmetry with
            :func:`resolve_api_key` and in case a future provider keys its file name
            off it).
        provider: The ``AgentProvider`` value (e.g. ``"google"``), if known. When given
            and recognized, provider-conventional locations are appended after Carmel's
            own files.
    """
    del env_var  # not currently used for path construction; kept for API symmetry
    candidates = _carmel_credential_candidates()
    if provider is not None:
        for raw_path in _PROVIDER_CREDENTIAL_PATHS.get(provider, ()):
            candidates.append(Path(raw_path).expanduser())
    return candidates


def _warn_if_permissive(path: Path) -> None:
    """Log a WARNING if ``path`` is group- or world-readable, but still use it.

    A permissive credential file is a real risk (other local users/processes can read
    it), but refusing to use it would break a setup that otherwise works -- so we warn
    loudly, by PATH and MODE only, and proceed. Never include the file's contents.
    """
    try:
        mode = path.stat().st_mode
    except OSError:
        return
    if mode & 0o077:
        logger.warning(
            "credential file %s is group/world-readable (mode %s); consider `chmod 600 %s`",
            path,
            oct(mode & 0o777),
            path,
        )


def _parse_env_line(line: str) -> tuple[str, str] | None:
    """Parse one dotenv-style line into a ``(key, value)`` pair, or ``None``.

    Tolerates: blank lines, ``#`` comments, an ``export `` prefix, surrounding single or
    double quotes on the value, and trailing whitespace. Any line that does not fit this
    shape is treated as malformed and skipped -- a stray line in a credentials file must
    never raise and take down a run.
    """
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped[len("export ") :].lstrip()
    if "=" not in stripped:
        return None
    key, _, value = stripped.partition("=")
    key = key.strip()
    value = value.strip()
    if not key:
        return None
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1]
    return key, value


def _read_env_file(path: Path) -> dict[str, str]:
    """Read a dotenv-style file into a dict, defensively.

    A missing file, a directory where a file was expected, a permission error, or
    undecodable bytes are all treated as "no credential here": this returns an empty
    dict rather than raising, so the caller simply moves on to the next candidate.
    """
    try:
        if not path.is_file():
            return {}
        text = path.read_text(encoding="utf-8", errors="strict")
    except OSError:
        return {}
    except UnicodeDecodeError:
        return {}

    result: dict[str, str] = {}
    for line in text.splitlines():
        parsed = _parse_env_line(line)
        if parsed is None:
            continue
        key, value = parsed
        result[key] = value
    return result


def resolve_api_key(env_var: str, *, provider: str | None = None) -> str | None:
    """Resolve an API key for ``env_var``, searching env then well-known files.

    Resolution order, first hit wins (see module docstring for the full rationale):

    1. ``os.environ[env_var]``, if set and non-empty. An explicitly exported value
       always wins -- nothing found on disk may silently override it.
    2. Carmel's own credential files, in order (see :func:`credential_search_path`).
    3. Provider-conventional locations, when ``provider`` is given.

    Args:
        env_var: The environment variable name the key would normally be exported as
            (e.g. ``"GOOGLE_API_KEY"``); also the key looked up inside each candidate
            file.
        provider: The ``AgentProvider`` value (e.g. ``"google"``), used only to extend
            the search to that provider's conventional file location(s).

    Returns:
        The resolved key value, or ``None`` if it was not found anywhere searched.
        Never logs or otherwise exposes the key value itself.
    """
    env_value = os.environ.get(env_var)
    if env_value:
        return env_value

    for path in credential_search_path(env_var, provider=provider):
        values = _read_env_file(path)
        if env_var in values and values[env_var]:
            _warn_if_permissive(path)
            return values[env_var]

    return None
