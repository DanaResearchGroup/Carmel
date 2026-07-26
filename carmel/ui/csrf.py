# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0

"""Session-backed CSRF protection for the Carmel UI.

Hand-rolled on purpose: the project keeps its runtime dependency set small,
and the double-check pattern here (random per-session token, embedded in every
POST form, compared with :func:`hmac.compare_digest`) is standard and small
enough to own directly instead of pulling in ``flask-wtf``.
"""

from __future__ import annotations

import hmac
import secrets

from flask import Flask, abort, request, session

CSRF_SESSION_KEY = "csrf_token"
CSRF_FORM_FIELD = "csrf_token"
UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def generate_csrf_token() -> str:
    """Return the CSRF token for the current session, creating one if absent.

    The token is generated with :func:`secrets.token_urlsafe` and stored in
    the Flask session so that repeated form renders within one session reuse
    the same token.

    Returns:
        The session's CSRF token.
    """
    token = session.get(CSRF_SESSION_KEY)
    if not isinstance(token, str) or not token:
        token = secrets.token_urlsafe(32)
        session[CSRF_SESSION_KEY] = token
    return token


def validate_csrf() -> None:
    """Reject unsafe requests that lack a valid CSRF token.

    Safe (read-only) methods pass through untouched. For unsafe methods the
    submitted ``csrf_token`` form field must match the session token under
    :func:`hmac.compare_digest`.

    Raises:
        werkzeug.exceptions.BadRequest: If the token is missing or invalid.
    """
    if request.method not in UNSAFE_METHODS:
        return
    expected = session.get(CSRF_SESSION_KEY)
    submitted = request.form.get(CSRF_FORM_FIELD)
    if not isinstance(expected, str) or not expected or not submitted:
        abort(400, description="Missing CSRF token.")
    if not hmac.compare_digest(expected, submitted):
        abort(400, description="Invalid CSRF token.")


def init_csrf(app: Flask) -> None:
    """Wire CSRF protection into a Flask app.

    Exposes ``csrf_token()`` as a Jinja global for embedding the token in
    POST forms, and enforces token validation on every unsafe request.

    Args:
        app: The Flask application to protect.
    """
    app.jinja_env.globals["csrf_token"] = generate_csrf_token
    app.before_request(validate_csrf)
