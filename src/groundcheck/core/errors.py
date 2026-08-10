"""GroundCheck exception hierarchy.

Every raised exception in this package is a subclass of ``GroundCheckError``.
Nothing in the codebase may raise a bare ``Exception``.
"""

from __future__ import annotations


class GroundCheckError(Exception):
    """Base class for all GroundCheck exceptions."""


class ConfigError(GroundCheckError):
    """Configuration is invalid, or a required credential/dependency is missing."""


class ExtractionError(GroundCheckError):
    """Claim extraction failed irrecoverably (all fallbacks exhausted)."""


class VerifierError(GroundCheckError):
    """A verifier backend failed irrecoverably (e.g. judge output unparseable after repair)."""
