"""Tests for score_agent_reputation MCP tool."""


class TestScoreAgentReputationImpl:
    """Test score_agent_reputation_impl function."""

    def test_returns_dict(self):
        """score_agent_reputation_impl returns a dict."""
        from server import score_agent_reputation_impl

        result = score_agent_reputation_impl("test-agent")
        assert isinstance(result, dict)

    def test_has_reputation_score(self):
        """Result includes reputation_score."""
        from server import score_agent_reputation_impl

        result = score_agent_reputation_impl("test-agent")
        assert "reputation_score" in result
        assert 0.0 <= result["reputation_score"] <= 1.0

    def test_has_status(self):
        """Result includes status field."""
        from server import score_agent_reputation_impl

        result = score_agent_reputation_impl("test-agent")
        assert result["status"] == "scored"

    def test_has_timestamp(self):
        """Result includes analysis_timestamp."""
        from server import score_agent_reputation_impl

        result = score_agent_reputation_impl("test-agent")
        assert "analysis_timestamp" in result

    def test_invalid_agent_id(self):
        """Empty agent_id returns error."""
        from server import score_agent_reputation_impl

        result = score_agent_reputation_impl("")
        assert "error" in result

    def test_none_agent_id(self):
        """None agent_id returns error."""
        from server import score_agent_reputation_impl

        result = score_agent_reputation_impl(None)
        assert "error" in result

    def test_has_components(self):
        """Result includes components breakdown."""
        from server import score_agent_reputation_impl

        result = score_agent_reputation_impl("test-agent")
        assert "components" in result


class TestMCPToolRegistration:
    """Verify MCP tool registration for score_agent_reputation."""

    def test_mcp_has_score_agent_reputation(self):
        """MCP server has score_agent_reputation tool."""
        from server import mcp

        tool_names = list(mcp._tool_manager._tools.keys())
        assert "score_agent_reputation" in tool_names

    def test_total_mcp_tools_count_is_19(self):
        """Server should now have 24 MCP tools registered (19 core + 5 compliance)."""
        from server import mcp

        tool_count = len(mcp._tool_manager._tools)
        assert tool_count == 24
