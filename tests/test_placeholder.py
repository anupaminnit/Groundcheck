"""Placeholder test to keep CI green until Phase 1 adds real coverage."""

import groundcheck


def test_package_imports() -> None:
    assert groundcheck is not None
