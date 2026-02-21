"""Tests targeting uncovered code paths in server.py to push coverage to 95%+."""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import patch, MagicMock, PropertyMock


class TestMonitoringDeprecationFix:
    """Verify datetime.utcnow() deprecation warnings are eliminated."""

    def test_no_utcnow_in_monitoring(self):
        """Ensure monitoring.py does not use deprecated datetime.utcnow()."""
        import inspect
        import monitoring
        source = inspect.getsource(monitoring)
        assert "utcnow()" not in source, (
            "monitoring.py still uses deprecated datetime.utcnow()"
        )
