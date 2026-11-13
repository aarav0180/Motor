"""
server/routes package — REST routers for Tier 2 API.
"""

from . import health, telemetry, alerts, feedback, config, push

__all__ = ["health", "telemetry", "alerts", "feedback", "config", "push"]
