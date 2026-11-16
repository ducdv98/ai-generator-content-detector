"""Switch the public-deployment limiter off for the functional tests.

The limiter itself is tested in test_limits.py, which builds its own instances.
"""
import os

os.environ.setdefault("AIGC_LIMITS", "off")
