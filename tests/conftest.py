"""Test fixtures for the autobiz kernel."""

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

    env._OFFER = env._DELIVERABLE = env._FIX = env._REPLY = None
    yield
    env._OFFER = env._DELIVERABLE = env._FIX = env._REPLY = None
