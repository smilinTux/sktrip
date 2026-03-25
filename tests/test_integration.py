"""Tests for the integration engine."""

from sktrip.integration import Insight, IntegrationReport


class TestInsight:
    def test_to_dict(self):
        insight = Insight(
            title="Mycelium-Sovereignty Connection",
            description="Underground networks mirror decentralized governance",
            domains_bridged=["mycology", "sovereignty"],
            novelty_score=8.5,
            category="connection",
        )
        d = insight.to_dict()
        assert d["title"] == "Mycelium-Sovereignty Connection"
        assert d["novelty_score"] == 8.5
        assert "mycology" in d["domains_bridged"]


class TestIntegrationReport:
    def test_to_markdown(self):
        report = IntegrationReport(
            session_id="abc123",
            substance="psilocybin",
            intention="explore connections",
            duration_s=1800.0,
            total_turns=8,
            peak_intensity=7,
            insights=[
                Insight(
                    title="Test Insight",
                    description="A novel connection was found",
                    domains_bridged=["tech", "nature"],
                    novelty_score=7.5,
                    category="connection",
                ),
            ],
            recurring_themes=["interconnection", "dissolution"],
            entity_descriptions=["A geometric being made of light"],
            overall_summary="The session revealed hidden patterns.",
            novelty_average=7.5,
            timestamp="2026-03-25T03:00:00Z",
        )
        md = report.to_markdown()
        assert "SKTrip Integration Report" in md
        assert "psilocybin" in md
        assert "Test Insight" in md
        assert "interconnection" in md
        assert "geometric being" in md
        assert "7.5/10" in md
