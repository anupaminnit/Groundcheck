"""GroundCheck: verify RAG answers against retrieved evidence and apply a policy to
unsupported claims.
"""

from __future__ import annotations

from groundcheck.config import GuardConfig
from groundcheck.core.guard import Guard
from groundcheck.core.schemas import Evidence, GuardReport, Verdict

__all__ = ["Guard", "GuardConfig", "GuardReport", "Verdict", "Evidence"]
