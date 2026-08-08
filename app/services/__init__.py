"""Domain service layer.

Each class owns one area of responsibility and takes its dependencies through its
constructor, so the API layer stays thin and the services stay testable in isolation.
"""

from app.services.complaint_manager import ComplaintManager
from app.services.notifications import NotificationManager
from app.services.statistics import StatisticsService

__all__ = ["ComplaintManager", "NotificationManager", "StatisticsService"]
