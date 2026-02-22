"""Tests for verify_transaction_mandate and detect_agent_collusion MCP tools."""

from datetime import datetime


class TestVerifyTransactionMandateImpl:
    """Test verify_transaction_mandate_impl function."""

    def test_returns_dict(self):
        """verify_transaction_mandate_impl returns a dict."""
        from server import verify_transaction_mandate_impl

        result = verify_transaction_mandate_impl(
            transaction_data={
                "amount": 50.0,
                "timestamp": datetime.now().isoformat(),
            },
            mandate={},
        )
        assert isinstance(result, dict)

    def test_compliant_transaction(self):
        """Compliant transaction returns compliant=True."""
        from server import verify_transaction_mandate_impl

        result = verify_transaction_mandate_impl(
            transaction_data={
                "amount": 50.0,
                "merchant": "Amazon",
                "timestamp": datetime.now().isoformat(),
            },
            mandate={"max_amount": 100.0, "allowed_merchants": ["Amazon"]},
        )
        assert result["compliant"] is True
        assert result["status"] == "verified"

    def test_non_compliant_transaction(self):
        """Non-compliant transaction returns violations."""
        from server import verify_transaction_mandate_impl

        result = verify_transaction_mandate_impl(
            transaction_data={
                "amount": 500.0,
                "merchant": "Casino",
                "timestamp": datetime.now().isoformat(),
            },
            mandate={"max_amount": 100.0, "blocked_merchants": ["Casino"]},
        )
        assert result["compliant"] is False
        assert len(result["violations"]) >= 1

    def test_invalid_transaction_data(self):
        """Invalid input returns error."""
        from server import verify_transaction_mandate_impl

        result = verify_transaction_mandate_impl(
            transaction_data="not a dict",
            mandate={},
        )
        assert "error" in result

    def test_has_timestamp(self):
        """Result includes analysis_timestamp."""
        from server import verify_transaction_mandate_impl

        result = verify_transaction_mandate_impl(
            transaction_data={
                "amount": 50.0,
                "timestamp": datetime.now().isoformat(),
            },
            mandate={},
        )
        assert "analysis_timestamp" in result


class TestDetectAgentCollusionImpl:
    """Test detect_agent_collusion_impl function."""

    def test_returns_dict(self):
        """detect_agent_collusion_impl returns a dict."""
        from server import detect_agent_collusion_impl

        result = detect_agent_collusion_impl(
            agent_ids=[],
            window_seconds=3600,
        )
        assert isinstance(result, dict)

    def test_empty_agents_safe(self):
        """No agents returns low collusion score."""
        from server import detect_agent_collusion_impl

        result = detect_agent_collusion_impl(agent_ids=[], window_seconds=3600)
        assert result["collusion_score"] == 0.0
        assert result["status"] == "analyzed"

    def test_invalid_agent_ids(self):
        """Non-list agent_ids returns error."""
        from server import detect_agent_collusion_impl

        result = detect_agent_collusion_impl(
            agent_ids="not_a_list",
            window_seconds=3600,
        )
        assert "error" in result

    def test_has_graph_metrics(self):
        """Result includes graph_metrics."""
        from server import detect_agent_collusion_impl

        result = detect_agent_collusion_impl(agent_ids=[], window_seconds=3600)
        assert "graph_metrics" in result

    def test_has_timestamp(self):
        """Result includes analysis_timestamp."""
        from server import detect_agent_collusion_impl

        result = detect_agent_collusion_impl(agent_ids=[], window_seconds=3600)
        assert "analysis_timestamp" in result


class TestMCPToolRegistration:
    """Verify MCP tool registration for Phase D tools."""

    def test_mcp_has_verify_transaction_mandate(self):
        """MCP server has verify_transaction_mandate tool."""
        from server import mcp

        tool_names = list(mcp._tool_manager._tools.keys())
        assert "verify_transaction_mandate" in tool_names

    def test_mcp_has_detect_agent_collusion(self):
        """MCP server has detect_agent_collusion tool."""
        from server import mcp

        tool_names = list(mcp._tool_manager._tools.keys())
        assert "detect_agent_collusion" in tool_names

    def test_total_mcp_tools_count_is_18(self):
        """Server should now have 19 MCP tools registered."""
        from server import mcp

        tool_count = len(mcp._tool_manager._tools)
        assert tool_count == 19
