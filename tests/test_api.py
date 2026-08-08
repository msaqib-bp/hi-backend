"""API tests: the citizen journey, the admin journey, and the boundaries between them."""

from __future__ import annotations

import pytest
from httpx import AsyncClient

API = "/api/v1"


class TestHealth:
    async def test_health_reports_database_and_ai(self, client: AsyncClient) -> None:
        response = await client.get("/health")
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "healthy"
        assert "ai" in payload

    async def test_root_advertises_docs(self, client: AsyncClient) -> None:
        response = await client.get("/")
        assert response.status_code == 200
        assert response.json()["docs"] == "/docs"


class TestComplaintSubmission:
    """The citizen path — public, no account required."""

    async def test_submit_returns_ai_triage_immediately(self, client: AsyncClient) -> None:
        response = await client.post(
            f"{API}/complaints",
            json={
                "description": "There is a large water leak near the main road and "
                "traffic is becoming difficult.",
                "location": "MG Road, Ward 12",
            },
        )
        assert response.status_code == 201, response.text
        body = response.json()
        complaint = body["complaint"]

        assert complaint["reference_code"].startswith("CIV-")
        assert complaint["category"]
        assert complaint["priority"]
        assert complaint["ai_summary"]
        assert complaint["ai_output"]["engine"]
        # Routed to a department on intake, so the queue is never unassigned.
        assert complaint["assigned_department"] is not None
        assert complaint["status"] == "assigned"

    async def test_submission_records_an_audit_event(self, client: AsyncClient) -> None:
        response = await client.post(
            f"{API}/complaints",
            json={
                "description": "The drain outside our gate is blocked and overflowing.",
                "location": "Gandhi Nagar",
            },
        )
        events = response.json()["complaint"]["events"]
        assert len(events) == 1
        assert events[0]["from_status"] is None
        assert events[0]["actor"] == "system"

    @pytest.mark.parametrize(
        "payload",
        [
            {"description": "short", "location": "Somewhere"},
            {"description": "a" * 30, "location": "X"},
            {"location": "Missing description"},
            {"description": "The drain is blocked badly here today."},
        ],
    )
    async def test_invalid_submissions_are_rejected(
        self, client: AsyncClient, payload: dict
    ) -> None:
        response = await client.post(f"{API}/complaints", json=payload)
        assert response.status_code == 422

    async def test_gibberish_is_rejected(self, client: AsyncClient) -> None:
        """Placeholder text would pollute the statistics, so it is refused at the door."""
        response = await client.post(
            f"{API}/complaints",
            json={"description": "aaaaaaaaaaaaaaaaaaaaaaaa", "location": "Main Road"},
        )
        assert response.status_code == 422
        assert "description" in response.json()["error"]["detail"]


class TestTracking:
    async def test_track_by_reference_needs_no_auth(self, client: AsyncClient) -> None:
        created = await client.post(
            f"{API}/complaints",
            json={
                "description": "Streetlight has been off for two weeks on our lane.",
                "location": "Sector 7",
            },
        )
        reference = created.json()["complaint"]["reference_code"]

        response = await client.get(f"{API}/complaints/track/{reference}")
        assert response.status_code == 200
        assert response.json()["reference_code"] == reference

    async def test_reference_lookup_is_case_insensitive(self, client: AsyncClient) -> None:
        """People retype the code from a note; casing should not matter."""
        created = await client.post(
            f"{API}/complaints",
            json={
                "description": "Garbage bin near the market is overflowing badly.",
                "location": "Market Street",
            },
        )
        reference = created.json()["complaint"]["reference_code"]

        response = await client.get(f"{API}/complaints/track/{reference.lower()}")
        assert response.status_code == 200

    async def test_unknown_reference_returns_404(self, client: AsyncClient) -> None:
        response = await client.get(f"{API}/complaints/track/CIV-NOPE99")
        assert response.status_code == 404
        assert response.json()["error"]["type"] == "not_found"


class TestAuthentication:
    async def test_login_returns_token_and_user(self, client: AsyncClient) -> None:
        response = await client.post(
            f"{API}/auth/login",
            json={"email": "admin@civic.gov", "password": "admin123"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["access_token"]
        assert body["user"]["role"] == "admin"

    async def test_wrong_password_rejected(self, client: AsyncClient) -> None:
        response = await client.post(
            f"{API}/auth/login",
            json={"email": "admin@civic.gov", "password": "wrongpassword"},
        )
        assert response.status_code == 401

    async def test_unknown_email_gives_the_same_error(self, client: AsyncClient) -> None:
        """Identical message for both cases, so accounts cannot be enumerated."""
        unknown = await client.post(
            f"{API}/auth/login",
            json={"email": "nobody@civic.gov", "password": "admin123"},
        )
        wrong_password = await client.post(
            f"{API}/auth/login",
            json={"email": "admin@civic.gov", "password": "wrongpassword"},
        )
        assert unknown.status_code == wrong_password.status_code == 401
        assert unknown.json()["error"]["message"] == wrong_password.json()["error"]["message"]

    async def test_me_requires_a_token(self, client: AsyncClient) -> None:
        assert (await client.get(f"{API}/auth/me")).status_code == 401

    async def test_garbage_token_rejected(self, client: AsyncClient) -> None:
        response = await client.get(
            f"{API}/auth/me", headers={"Authorization": "Bearer not-a-real-token"}
        )
        assert response.status_code == 401


class TestAdminAccessControl:
    @pytest.mark.parametrize(
        "path", ["/complaints", "/complaints?status=open"]
    )
    async def test_listing_requires_auth(self, client: AsyncClient, path: str) -> None:
        assert (await client.get(f"{API}{path}")).status_code == 401

    async def test_listing_works_with_auth(
        self, client: AsyncClient, auth_headers: dict
    ) -> None:
        response = await client.get(f"{API}/complaints", headers=auth_headers)
        assert response.status_code == 200
        assert "items" in response.json()


class TestComplaintManagement:
    @pytest.fixture
    async def complaint_id(self, client: AsyncClient) -> str:
        response = await client.post(
            f"{API}/complaints",
            json={
                "description": "Sewage is overflowing from the manhole near the temple.",
                "location": "Temple Street, Ward 3",
            },
        )
        return response.json()["complaint"]["id"]

    async def test_status_transition_records_actor_and_note(
        self, client: AsyncClient, auth_headers: dict, complaint_id: str
    ) -> None:
        response = await client.patch(
            f"{API}/complaints/{complaint_id}",
            headers=auth_headers,
            json={"status": "in_progress", "note": "Crew dispatched"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "in_progress"

        latest = body["events"][-1]
        assert latest["to_status"] == "in_progress"
        assert latest["note"] == "Crew dispatched"
        assert latest["actor"] == "Municipal Administrator"

    async def test_resolving_sets_timestamp_and_duration(
        self, client: AsyncClient, auth_headers: dict, complaint_id: str
    ) -> None:
        await client.patch(
            f"{API}/complaints/{complaint_id}",
            headers=auth_headers,
            json={"status": "in_progress"},
        )
        response = await client.patch(
            f"{API}/complaints/{complaint_id}",
            headers=auth_headers,
            json={"status": "resolved", "resolution_note": "Repaired on site"},
        )
        body = response.json()
        assert body["status"] == "resolved"
        assert body["resolved_at"] is not None
        assert body["resolution_hours"] is not None
        assert body["is_overdue"] is False  # terminal states are never overdue

    async def test_illegal_transition_is_blocked(
        self, client: AsyncClient, auth_headers: dict, complaint_id: str
    ) -> None:
        """assigned -> resolved is legal; resolved -> open is not."""
        await client.patch(
            f"{API}/complaints/{complaint_id}",
            headers=auth_headers,
            json={"status": "resolved"},
        )
        response = await client.patch(
            f"{API}/complaints/{complaint_id}",
            headers=auth_headers,
            json={"status": "open"},
        )
        assert response.status_code == 409
        assert response.json()["error"]["type"] == "invalid_transition"
        assert "allowed" in response.json()["error"]["detail"]

    async def test_overriding_the_ai_is_flagged(
        self, client: AsyncClient, auth_headers: dict, complaint_id: str
    ) -> None:
        """The override flag feeds the dashboard's real-world accuracy metric."""
        response = await client.patch(
            f"{API}/complaints/{complaint_id}",
            headers=auth_headers,
            json={"category": "safety"},
        )
        body = response.json()
        assert body["category"] == "safety"
        assert body["ai_overridden"] is True

    async def test_reanalysis_preserves_human_override(
        self, client: AsyncClient, auth_headers: dict, complaint_id: str
    ) -> None:
        """Re-running the AI must not silently undo a deliberate correction."""
        await client.patch(
            f"{API}/complaints/{complaint_id}",
            headers=auth_headers,
            json={"category": "safety"},
        )
        await client.post(
            f"{API}/ai/complaints/{complaint_id}/reanalyze", headers=auth_headers
        )

        response = await client.get(f"{API}/complaints/{complaint_id}", headers=auth_headers)
        assert response.json()["category"] == "safety"


class TestFiltering:
    @pytest.fixture
    async def populated(self, client: AsyncClient) -> None:
        for index in range(5):
            await client.post(
                f"{API}/complaints",
                json={
                    "description": f"Water pipeline is leaking badly at location {index} here.",
                    "location": "Ward 5" if index % 2 else "Ward 9",
                },
            )

    async def test_pagination_metadata(
        self, client: AsyncClient, auth_headers: dict, populated
    ) -> None:
        response = await client.get(
            f"{API}/complaints?page=1&page_size=2", headers=auth_headers
        )
        body = response.json()
        assert len(body["items"]) == 2
        assert body["total"] == 5
        assert body["pages"] == 3

    async def test_filter_by_location(
        self, client: AsyncClient, auth_headers: dict, populated
    ) -> None:
        response = await client.get(
            f"{API}/complaints?location=Ward 9", headers=auth_headers
        )
        items = response.json()["items"]
        assert items
        assert all("Ward 9" in item["location"] for item in items)

    async def test_free_text_search(
        self, client: AsyncClient, auth_headers: dict, populated
    ) -> None:
        response = await client.get(f"{API}/complaints?q=pipeline", headers=auth_headers)
        assert response.json()["total"] == 5

    async def test_filter_by_status(
        self, client: AsyncClient, auth_headers: dict, populated
    ) -> None:
        response = await client.get(
            f"{API}/complaints?status=resolved", headers=auth_headers
        )
        assert response.json()["total"] == 0


class TestAnalyticsEndpoints:
    async def test_overview_is_public(self, client: AsyncClient) -> None:
        response = await client.get(f"{API}/analytics/overview")
        assert response.status_code == 200
        assert "interpretation" in response.json()

    @pytest.mark.parametrize(
        "dimension", ["category", "priority", "status", "location", "department"]
    )
    async def test_every_distribution_dimension(
        self, client: AsyncClient, dimension: str
    ) -> None:
        response = await client.get(f"{API}/analytics/distribution/{dimension}")
        assert response.status_code == 200
        assert response.json()["dimension"] == dimension

    async def test_unknown_dimension_is_rejected(self, client: AsyncClient) -> None:
        response = await client.get(f"{API}/analytics/distribution/nonsense")
        assert response.status_code == 422

    async def test_resolution_time_report_shape(self, client: AsyncClient) -> None:
        response = await client.get(f"{API}/analytics/resolution-time")
        body = response.json()
        assert "descriptive" in body
        assert "quartiles" in body
        assert "sla_breach_rate" in body

    async def test_trends_respects_day_bounds(self, client: AsyncClient) -> None:
        assert (await client.get(f"{API}/analytics/trends?days=7")).status_code == 200
        assert (await client.get(f"{API}/analytics/trends?days=3")).status_code == 422

    async def test_every_analytics_response_carries_an_interpretation(
        self, client: AsyncClient
    ) -> None:
        """The Batch 4 requirement, enforced as a test."""
        for path in [
            "/analytics/overview",
            "/analytics/distribution/category",
            "/analytics/trends",
            "/analytics/departments",
            "/analytics/resolution-time",
        ]:
            body = (await client.get(f"{API}{path}")).json()
            assert body.get("interpretation"), f"{path} has no interpretation"


class TestAIEndpoints:
    async def test_status_reports_engines(self, client: AsyncClient) -> None:
        response = await client.get(f"{API}/ai/status")
        body = response.json()
        assert "active_engine" in body
        assert len(body["categories"]) == 7
        assert len(body["priorities"]) == 4

    async def test_assistant_answers_and_returns_its_grounding(
        self, client: AsyncClient
    ) -> None:
        response = await client.post(
            f"{API}/ai/assistant", json={"question": "How many complaints are open?"}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["answer"]
        # The context is echoed so any claim in the answer can be checked.
        assert "kpis" in body["grounded_on"]

    async def test_departments_are_listed(self, client: AsyncClient) -> None:
        response = await client.get(f"{API}/departments")
        assert response.status_code == 200
        assert len(response.json()) == 7
