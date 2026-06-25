"""
conftest.py — Shared pytest fixtures and configuration.
"""

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: marks tests that call external APIs (skip with -m 'not integration')"
    )
