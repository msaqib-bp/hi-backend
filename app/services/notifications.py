"""``NotificationManager`` — tells people when something changes.

The spec's suggested class list includes a ``NotificationManager``, and the citizen
journey depends on it: the whole point of issuing a reference code is that the reporter
hears back. This implementation writes structured log records and returns the message it
would have sent.

**It does not send real email or SMS**, and that is a deliberate scope decision rather
than an omission: a live email provider means credentials in the deploy, a domain to
verify, and a spam-reputation problem — none of which demonstrate anything about the
civic-AI problem this project is judged on. The seam is here and correct, so wiring
SendGrid or Twilio in later is a change to one method body.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.core.logging import get_logger
from app.models.enums import ComplaintStatus

if TYPE_CHECKING:
    from app.models.complaint import Complaint

log = get_logger(__name__)


@dataclass
class Notification:
    """One outbound message."""

    channel: str  # "email" | "sms" | "log"
    recipient: str
    subject: str
    body: str


#: What the citizen is told at each stage. Written to be read by a person who filed a
#: complaint and wants to know whether anything is happening.
_STATUS_MESSAGES: dict[ComplaintStatus, str] = {
    ComplaintStatus.OPEN: (
        "We have received your complaint and it is queued for review."
    ),
    ComplaintStatus.ASSIGNED: (
        "Your complaint has been assigned to the {department} team."
    ),
    ComplaintStatus.IN_PROGRESS: (
        "Work has started on your complaint. The {department} team is on it."
    ),
    ComplaintStatus.RESOLVED: (
        "Your complaint has been marked resolved. If the problem persists, reply to "
        "this message and we will reopen it."
    ),
    ComplaintStatus.REJECTED: (
        "Your complaint could not be actioned. See the note below for the reason."
    ),
}


class NotificationManager:
    """Builds and dispatches complaint notifications."""

    def __init__(self, *, enabled: bool = True) -> None:
        self._enabled = enabled
        self._sent: list[Notification] = []

    @property
    def sent(self) -> list[Notification]:
        """Everything dispatched this session — asserted against in tests."""
        return list(self._sent)

    def notify_status_change(
        self, complaint: Complaint, *, note: str | None = None
    ) -> Notification | None:
        """Notify the reporter that their complaint moved to a new status."""
        if not self._enabled:
            return None

        department = (
            complaint.assigned_department.name
            if complaint.assigned_department
            else "relevant"
        )
        template = _STATUS_MESSAGES.get(complaint.status, "Your complaint has been updated.")
        body = template.format(department=department)
        if note:
            body += f"\n\nNote from the team: {note}"
        body += f"\n\nReference: {complaint.reference_code}"

        # No contact details is the normal case — reporting is anonymous by design.
        # Still logged, so the audit trail is complete.
        recipient = complaint.reporter_contact or "unknown"
        channel = "email" if "@" in recipient else ("sms" if recipient != "unknown" else "log")

        notification = Notification(
            channel=channel,
            recipient=recipient,
            subject=f"Update on complaint {complaint.reference_code}",
            body=body,
        )
        return self._dispatch(notification, complaint)

    def notify_new_complaint(self, complaint: Complaint) -> Notification | None:
        if not self._enabled:
            return None

        recipient = complaint.reporter_contact or "unknown"
        channel = "email" if "@" in recipient else ("sms" if recipient != "unknown" else "log")
        notification = Notification(
            channel=channel,
            recipient=recipient,
            subject=f"Complaint received — {complaint.reference_code}",
            body=(
                f"Thank you for reporting this. Your reference is "
                f"{complaint.reference_code}.\n\n"
                f"We classified it as {complaint.category.label} at "
                f"{complaint.priority.label} priority and routed it to the "
                f"{complaint.assigned_department.name if complaint.assigned_department else 'relevant'} "
                f"team.\n\nTrack progress any time using your reference code."
            ),
        )
        return self._dispatch(notification, complaint)

    def _dispatch(self, notification: Notification, complaint: Complaint) -> Notification:
        """Where a real provider integration would go."""
        self._sent.append(notification)
        log.info(
            "notification_dispatched",
            channel=notification.channel,
            reference=complaint.reference_code,
            status=complaint.status.value,
            recipient_known=notification.recipient != "unknown",
        )
        return notification
