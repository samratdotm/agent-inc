"""Test fixtures for Agent Inc."""

import sys
from pathlib import Path

import pytest

# Make `env` and `tasks` importable.
ROOT = str(Path(__file__).parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture(autouse=True)
def reset_state():
    """Clear per-episode state so it doesn't bleed across tests."""
    import env

    env._SCENARIO = env._OFFER = env._DELIVERABLE = None
    env._TOOL_CALLS = 0
    yield
    env._SCENARIO = env._OFFER = env._DELIVERABLE = None
    env._TOOL_CALLS = 0
