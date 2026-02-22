"""
Tests for Defense Compliance Modules.

Covers:
- Insider threat detection (insider_threat.py)
- SIEM integration (siem_integration.py)
- Cleared personnel analytics (cleared_personnel.py)
- Compliance dashboard metrics (dashboard_metrics.py)
- NTAS alignment verification across all modules
"""

import json
import re
import threading
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from compliance import (
    ADJUDICATIVE_GUIDELINES,
    BEHAVIORAL_INDICATORS,
    COMPLIANCE_CONTROLS,
    CORRELATION_RULES,
    DOD_INCIDENT_CATEGORIES,
    MATURITY_CRITERIA,
    MITRE_ATTACK_MAP,
    NIST_CONTROLS,
    BehavioralIndicator,
    ClearanceLevel,
    ClearanceStatus,
    ClearedPersonnelAnalyzer,
    ClearedPersonnelRecord,
    ComplianceDashboard,
    CorrelationRule,
    EventFormatter,
    EventSeverity,
    InsiderThreatAssessor,
    MaturityLevel,
    PolygraphType,
    SIEMIntegration,
    ThreatLevel,
    score_to_threat_level,
)


# =============================================================================
# NTAS Alignment Verification
# =============================================================================


class TestNTASAlignment:
    """Verify NTAS alignment across ALL modules and references."""

    def test_threat_level_enum_values(self):
        """ThreatLevel enum uses NTAS values, not HSAS."""
        assert ThreatLevel.BASELINE.value == "BASELINE"
        assert ThreatLevel.ADVISORY.value == "ADVISORY"
        assert ThreatLevel.ELEVATED.value == "ELEVATED"
        assert ThreatLevel.IMMINENT.value == "IMMINENT"

    def test_threat_level_enum_count(self):
        """Exactly 4 NTAS-aligned threat levels."""
        assert len(ThreatLevel) == 4

    def test_no_hsas_values_in_threat_level(self):
        """No deprecated HSAS values (LOW, GUARDED, HIGH, SEVERE, GREEN, BLUE, YELLOW, ORANGE, RED)."""
        hsas_values = {"LOW", "GUARDED", "HIGH", "SEVERE", "GREEN", "BLUE", "YELLOW", "ORANGE", "RED"}
        actual_values = {level.value for level in ThreatLevel}
        assert actual_values & hsas_values == set(), (
            f"HSAS values found in ThreatLevel enum: {actual_values & hsas_values}"
        )

    def test_score_to_threat_level_boundaries(self):
        """Score mapping uses correct NTAS boundaries."""
        assert score_to_threat_level(0) == ThreatLevel.BASELINE
        assert score_to_threat_level(15) == ThreatLevel.BASELINE
        assert score_to_threat_level(30) == ThreatLevel.BASELINE
        assert score_to_threat_level(31) == ThreatLevel.ADVISORY
        assert score_to_threat_level(45) == ThreatLevel.ADVISORY
        assert score_to_threat_level(60) == ThreatLevel.ADVISORY
        assert score_to_threat_level(61) == ThreatLevel.ELEVATED
        assert score_to_threat_level(70) == ThreatLevel.ELEVATED
        assert score_to_threat_level(80) == ThreatLevel.ELEVATED
        assert score_to_threat_level(81) == ThreatLevel.IMMINENT
        assert score_to_threat_level(90) == ThreatLevel.IMMINENT
        assert score_to_threat_level(100) == ThreatLevel.IMMINENT

    def test_cleared_personnel_risk_levels_use_ntas(self):
        """ClearedPersonnelAnalyzer._calculate_overall_risk uses NTAS values."""
        analyzer = ClearedPersonnelAnalyzer()
        # Check all risk levels produced by _calculate_overall_risk
        for score, expected in [(0, "BASELINE"), (29, "BASELINE"),
                                (30, "ADVISORY"), (59, "ADVISORY"),
                                (60, "ELEVATED"), (79, "ELEVATED"),
                                (80, "IMMINENT"), (100, "IMMINENT")]:
            result = analyzer._calculate_overall_risk(
                [{"severity": "CRITICAL"}] * (score // 25),
                {"aggregate_risk_score": 0},
            )
            assert result["level"] in {"BASELINE", "ADVISORY", "ELEVATED", "IMMINENT"}, (
                f"Non-NTAS risk level '{result['level']}' produced for score ~{score}"
            )

    def test_assessor_threat_descriptions_use_ntas(self):
        """InsiderThreatAssessor descriptions reference NTAS, not HSAS."""
        assessor = InsiderThreatAssessor()
        for level in ThreatLevel:
            desc = assessor._threat_level_description(level)
            assert "HSAS" not in desc, f"HSAS reference in {level.value} description"
            # ELEVATED and IMMINENT should reference NTAS
            if level in (ThreatLevel.ELEVATED, ThreatLevel.IMMINENT):
                assert "NTAS" in desc, f"Missing NTAS reference in {level.value} description"


# =============================================================================
# Behavioral Indicators Tests
# =============================================================================


class TestBehavioralIndicators:
    """Verify 28 behavioral indicators are defined and functional."""

    def test_exactly_28_indicators(self):
        """Registry contains exactly 28 indicators."""
        assert len(BEHAVIORAL_INDICATORS) == 28

    def test_indicator_ids_sequential(self):
        """All IND-001 through IND-028 are present."""
        for i in range(1, 29):
            ind_id = f"IND-{i:03d}"
            assert ind_id in BEHAVIORAL_INDICATORS, f"Missing indicator {ind_id}"

    def test_indicators_have_required_fields(self):
        """Each indicator has all required attributes."""
        for ind_id, indicator in BEHAVIORAL_INDICATORS.items():
            assert isinstance(indicator, BehavioralIndicator)
            assert indicator.indicator_id == ind_id
            assert indicator.name, f"{ind_id} missing name"
            assert indicator.category, f"{ind_id} missing category"
            assert isinstance(indicator.weight, (int, float)), f"{ind_id} weight not numeric"
            assert 0 < indicator.weight <= 10, f"{ind_id} weight {indicator.weight} out of range"
            assert isinstance(indicator.nist_controls, list), f"{ind_id} nist_controls not a list"
            assert len(indicator.nist_controls) > 0, f"{ind_id} has no NIST control mappings"
            assert isinstance(indicator.mitre_techniques, list), f"{ind_id} mitre_techniques not a list"
            assert indicator.description, f"{ind_id} missing description"
            assert indicator.detection_logic, f"{ind_id} missing detection_logic"

    def test_indicator_categories_coverage(self):
        """Indicators cover expected threat categories."""
        categories = {ind.category for ind in BEHAVIORAL_INDICATORS.values()}
        expected = {"access", "data_movement", "evasion", "foreign_nexus",
                    "personal", "ci", "physical", "reconnaissance"}
        assert expected.issubset(categories), (
            f"Missing categories: {expected - categories}"
        )

    def test_nist_control_references_valid(self):
        """All NIST control references in indicators are defined."""
        for ind_id, indicator in BEHAVIORAL_INDICATORS.items():
            for ctrl in indicator.nist_controls:
                assert ctrl in NIST_CONTROLS, (
                    f"{ind_id} references undefined NIST control {ctrl}"
                )

    def test_mitre_technique_references_valid(self):
        """Most MITRE technique references in indicators are in the MITRE_ATTACK_MAP."""
        for ind_id, indicator in BEHAVIORAL_INDICATORS.items():
            for tech in indicator.mitre_techniques:
                # Some indicators may have no techniques (e.g., personal indicators)
                if tech:
                    assert tech in MITRE_ATTACK_MAP, (
                        f"{ind_id} references unmapped MITRE technique {tech}"
                    )


# =============================================================================
# Insider Threat Assessor Tests
# =============================================================================


class TestInsiderThreatAssessor:
    """Test the core insider threat assessment engine."""

    def setup_method(self):
        self.assessor = InsiderThreatAssessor()

    def test_basic_assessment_returns_structure(self):
        """Assessment returns all required fields."""
        result = self.assessor.assess_user("user-001", {})
        assert "assessment_id" in result
        assert "user_id" in result
        assert "risk_score" in result
        assert "threat_level" in result
        assert "indicators_evaluated" in result
        assert "indicators_triggered" in result
        assert "triggered_indicators" in result
        assert "nist_control_violations" in result
        assert "recommended_actions" in result
        assert "assessment_metadata" in result

    def test_clean_user_baseline_score(self):
        """User with no suspicious activity gets BASELINE."""
        result = self.assessor.assess_user("clean-user", {})
        assert result["risk_score"] == 0.0
        assert result["threat_level"] == "BASELINE"
        assert result["indicators_triggered"] == 0

    def test_unauthorized_classified_access_triggers(self):
        """IND-001 fires for classification breach."""
        self.assessor.update_profile("user-002", clearance_level="SECRET")
        result = self.assessor.assess_user("user-002", {
            "accessed_classification": "SCI",
        })
        triggered_ids = [t["indicator_id"] for t in result["triggered_indicators"]]
        assert "IND-001" in triggered_ids

    def test_after_hours_access_triggers(self):
        """IND-002 fires for off-hours login with sufficient baseline."""
        profile = self.assessor.get_or_create_profile("user-003")
        # Build a baseline of normal-hours logins
        for _ in range(20):
            profile.record_login(10, "10.0.0.1", True)
        result = self.assessor.assess_user("user-003", {"login_hour": 3})
        triggered_ids = [t["indicator_id"] for t in result["triggered_indicators"]]
        assert "IND-002" in triggered_ids

    def test_after_hours_skipped_without_baseline(self):
        """IND-002 does not fire without sufficient baseline observations."""
        result = self.assessor.assess_user("new-user", {"login_hour": 3})
        triggered_ids = [t["indicator_id"] for t in result["triggered_indicators"]]
        assert "IND-002" not in triggered_ids

    def test_mass_data_download_triggers(self):
        """IND-003 fires for data volume >3x average."""
        profile = self.assessor.get_or_create_profile("user-004")
        for _ in range(10):
            profile.record_data_transfer(1000, "internal")
        result = self.assessor.assess_user("user-004", {
            "data_volume_bytes": 50000,
        })
        triggered_ids = [t["indicator_id"] for t in result["triggered_indicators"]]
        assert "IND-003" in triggered_ids

    def test_access_outside_scope_triggers(self):
        """IND-004 fires for resource outside authorized list."""
        self.assessor.update_profile(
            "user-005", authorized_resources=["proj-alpha", "proj-beta"]
        )
        result = self.assessor.assess_user("user-005", {
            "resource_id": "proj-classified-gamma",
        })
        triggered_ids = [t["indicator_id"] for t in result["triggered_indicators"]]
        assert "IND-004" in triggered_ids

    def test_removable_media_triggers(self):
        """IND-005 fires for unauthorized removable media."""
        result = self.assessor.assess_user("user-006", {
            "removable_media_type": "USB",
            "device_id": "DEV-UNKNOWN",
            "approved_devices": ["DEV-001", "DEV-002"],
        })
        triggered_ids = [t["indicator_id"] for t in result["triggered_indicators"]]
        assert "IND-005" in triggered_ids

    def test_security_bypass_triggers(self):
        """IND-006 fires for security control bypass."""
        result = self.assessor.assess_user("user-007", {
            "security_bypass_actions": ["proxy_avoidance", "av_disabled"],
        })
        triggered_ids = [t["indicator_id"] for t in result["triggered_indicators"]]
        assert "IND-006" in triggered_ids

    def test_foreign_travel_to_adversary_triggers(self):
        """IND-007 fires for travel to adversary nation."""
        result = self.assessor.assess_user("user-008", {
            "travel_destination": "CN-Beijing",
        })
        triggered_ids = [t["indicator_id"] for t in result["triggered_indicators"]]
        assert "IND-007" in triggered_ids

    def test_post_termination_access_triggers(self):
        """IND-025 fires for terminated employee access."""
        result = self.assessor.assess_user("user-025", {
            "employment_status": "terminated",
        })
        triggered_ids = [t["indicator_id"] for t in result["triggered_indicators"]]
        assert "IND-025" in triggered_ids
        # Should be IMMINENT-level due to high weight
        assert result["risk_score"] > 0

    def test_multiple_indicators_increase_score(self):
        """Multiple triggered indicators produce higher risk score."""
        self.assessor.update_profile(
            "user-multi", clearance_level="SECRET",
            authorized_resources=["proj-alpha"],
        )
        result = self.assessor.assess_user("user-multi", {
            "accessed_classification": "SCI",
            "resource_id": "classified-system",
            "security_bypass_actions": ["proxy_avoidance"],
            "employment_status": "terminated",
        })
        assert result["indicators_triggered"] >= 3
        assert result["risk_score"] > 30  # Multiple indicators push well above BASELINE

    def test_assessment_records_history(self):
        """Assessments are recorded in profile history."""
        self.assessor.assess_user("user-history", {})
        self.assessor.assess_user("user-history", {})
        stats = self.assessor.get_assessment_stats()
        assert stats["total_assessments"] >= 2

    def test_nist_violations_populated(self):
        """NIST control violations are populated when indicators fire."""
        self.assessor.update_profile("user-nist", clearance_level="SECRET")
        result = self.assessor.assess_user("user-nist", {
            "accessed_classification": "SCI",
        })
        if result["indicators_triggered"] > 0:
            assert len(result["nist_control_violations"]) > 0
            violation = result["nist_control_violations"][0]
            assert "control_id" in violation
            assert "control_name" in violation
            assert "triggered_by" in violation

    def test_recommended_actions_match_threat_level(self):
        """Recommended actions are appropriate for the threat level."""
        # BASELINE
        result = self.assessor.assess_user("user-actions-clean", {})
        actions = [a["action"] for a in result["recommended_actions"]]
        assert "continue_monitoring" in actions

    def test_case_referral_generation(self):
        """Case referral produces a complete report."""
        self.assessor.assess_user("user-referral", {
            "employment_status": "terminated",
        })
        referral = self.assessor.generate_case_referral("user-referral")
        assert referral["referral_id"].startswith("ITCR-")
        assert "executive_summary" in referral
        assert "risk_summary" in referral
        assert "indicator_summary" in referral
        assert "legal_notice" in referral
        assert "FOR OFFICIAL USE ONLY" in referral["classification"]

    def test_list_indicators_returns_all(self):
        """list_indicators returns all 28 indicators."""
        indicators = self.assessor.list_indicators()
        assert len(indicators) == 28

    def test_thread_safety(self):
        """Concurrent assessments do not corrupt state."""
        errors = []

        def assess_user(user_id):
            try:
                for _ in range(10):
                    self.assessor.assess_user(user_id, {
                        "login_hour": 14,
                        "data_volume_bytes": 500,
                    })
            except Exception as e:
                errors.append(str(e))

        threads = [
            threading.Thread(target=assess_user, args=(f"thread-user-{i}",))
            for i in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"

    def test_assessment_metadata_references(self):
        """Assessment metadata contains correct framework references."""
        result = self.assessor.assess_user("user-meta", {})
        meta = result["assessment_metadata"]
        assert meta["framework"] == "NITTF Insider Threat Guide"
        assert meta["executive_order"] == "EO 13587"
        assert meta["uam_standard"] == "CNSSD 504"
        assert meta["indicator_count"] == 28

    def test_indicator_evaluation_returns_evidence(self):
        """Triggered indicators include evidence details."""
        result = self.assessor.assess_user("user-evidence", {
            "employment_status": "terminated",
        })
        for trigger in result["triggered_indicators"]:
            assert "evidence" in trigger
            assert "details" in trigger
            assert "confidence" in trigger
            assert 0 <= trigger["confidence"] <= 1.0
            assert "timestamp" in trigger

    def test_risk_score_bounded_0_100(self):
        """Risk score is always in [0, 100]."""
        # Clean
        r1 = self.assessor.assess_user("user-bound-clean", {})
        assert 0 <= r1["risk_score"] <= 100

        # Maximum indicators
        r2 = self.assessor.assess_user("user-bound-max", {
            "employment_status": "terminated",
            "security_bypass_actions": ["proxy_avoidance"],
            "network_scanning_detected": True,
            "scan_targets": 100,
            "covert_channel_indicators": ["dns_tunnel"],
            "accessed_classification": "SCI",
        })
        assert 0 <= r2["risk_score"] <= 100


# =============================================================================
# SIEM Integration Tests
# =============================================================================


class TestSIEMIntegration:
    """Test SIEM event generation, formatting, and correlation."""

    def setup_method(self):
        self.siem = SIEMIntegration()

    def test_cef_format_header(self):
        """CEF output follows ArcSight CEF standard format."""
        cef = EventFormatter.to_cef(
            "EVT-001", "Test Event", EventSeverity.HIGH,
            {"src": "user1", "msg": "test"},
        )
        assert cef.startswith("CEF:0|")
        parts = cef.split("|")
        assert len(parts) >= 7
        assert parts[1] == "2AcreStudios"
        assert parts[2] == "FraudDetectionMCP"
        # Severity 8 for HIGH
        assert parts[6] == "8"

    def test_cef_severity_mapping(self):
        """CEF severity values map correctly."""
        for sev, expected in [
            (EventSeverity.INFORMATIONAL, "0"),
            (EventSeverity.LOW, "2"),
            (EventSeverity.MEDIUM, "5"),
            (EventSeverity.HIGH, "8"),
            (EventSeverity.CRITICAL, "10"),
        ]:
            cef = EventFormatter.to_cef("EVT", "Test", sev, {})
            parts = cef.split("|")
            assert parts[6] == expected, f"CEF severity for {sev.name}"

    def test_cef_escaping(self):
        """CEF values are properly escaped."""
        cef = EventFormatter.to_cef(
            "EVT|002", "Name|With|Pipes", EventSeverity.LOW,
            {"key": "val=ue", "msg": "line1\nline2"},
        )
        # Pipes in event name should be escaped
        assert "\\|" in cef
        # Equals in extension value should be escaped
        assert "val\\=ue" in cef

    def test_leef_format_header(self):
        """LEEF output follows IBM QRadar format."""
        leef = EventFormatter.to_leef(
            "EVT-001", "Test Event", EventSeverity.MEDIUM,
            {"usrName": "user1"},
        )
        assert leef.startswith("LEEF:2.0|")
        parts = leef.split("|")
        assert parts[1] == "2AcreStudios"
        assert parts[2] == "FraudDetectionMCP"

    def test_syslog_rfc5424_format(self):
        """Syslog output follows RFC 5424 format."""
        syslog = EventFormatter.to_syslog_rfc5424(
            "EVT-001", "Test Event", EventSeverity.HIGH,
            {"userId": "user1", "riskScore": "75"},
        )
        # RFC 5424: <PRI>VERSION TIMESTAMP HOSTNAME APP-NAME PROCID MSGID [SD] MSG
        # PRI for facility 4 (security), severity 3 (error for HIGH) = 35
        assert syslog.startswith("<35>1 ")
        assert "insider-threat@2acrestudios" in syslog
        assert "fraud-detection-mcp" in syslog

    def test_syslog_severity_mapping(self):
        """Syslog priorities map to RFC 5424 correctly."""
        for sev, expected_sev_val in [
            (EventSeverity.CRITICAL, 2),
            (EventSeverity.HIGH, 3),
            (EventSeverity.MEDIUM, 4),
            (EventSeverity.LOW, 5),
            (EventSeverity.INFORMATIONAL, 6),
        ]:
            syslog = EventFormatter.to_syslog_rfc5424("E", "T", sev, {})
            # PRI = facility(4) * 8 + severity
            expected_pri = 4 * 8 + expected_sev_val
            assert syslog.startswith(f"<{expected_pri}>"), (
                f"Syslog PRI for {sev.name}: expected <{expected_pri}>"
            )

    def test_generate_events_all_formats(self):
        """generate_events produces CEF, LEEF, and Syslog output."""
        assessment = {
            "user_id": "user-siem",
            "risk_score": 75,
            "threat_level": "ELEVATED",
            "threat_level_description": "Elevated threat",
            "triggered_indicators": [
                {
                    "indicator_id": "IND-001",
                    "indicator_name": "Unauthorized Classified Access",
                    "category": "access",
                    "weight": 9.0,
                    "confidence": 0.95,
                    "details": "Classification breach",
                    "evidence": ["evidence1"],
                    "nist_controls": ["AC-2"],
                    "mitre_techniques": ["T1078"],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            ],
        }
        result = self.siem.generate_events(assessment)
        assert "events" in result
        assert "cef" in result["events"]
        assert "leef" in result["events"]
        assert "syslog" in result["events"]
        assert len(result["events"]["cef"]) > 0
        assert len(result["events"]["leef"]) > 0
        assert len(result["events"]["syslog"]) > 0

    def test_mitre_enrichment(self):
        """Events are enriched with MITRE ATT&CK data."""
        assessment = {
            "user_id": "user-mitre",
            "risk_score": 50,
            "threat_level": "ADVISORY",
            "triggered_indicators": [
                {
                    "indicator_id": "IND-016",
                    "indicator_name": "Privilege Escalation",
                    "category": "access",
                    "weight": 8.5,
                    "confidence": 0.9,
                    "details": "Escalation detected",
                    "evidence": [],
                    "nist_controls": ["AC-2"],
                    "mitre_techniques": ["T1548", "T1068"],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            ],
        }
        result = self.siem.generate_events(assessment)
        enrichment = result["mitre_enrichment"]
        technique_ids = [e["technique_id"] for e in enrichment]
        assert "T1548" in technique_ids
        assert "T1068" in technique_ids
        # Check enrichment has details
        for entry in enrichment:
            assert "technique_name" in entry
            assert "tactic" in entry
            assert "reference_url" in entry

    def test_dod_category_classification(self):
        """Events are classified into DoD incident categories."""
        # CAT-1: Root level intrusion (privilege escalation)
        result = self.siem.generate_events({
            "user_id": "u", "risk_score": 80, "threat_level": "IMMINENT",
            "triggered_indicators": [
                {"indicator_id": "IND-016", "indicator_name": "Priv Esc",
                 "category": "access", "weight": 8.5, "confidence": 0.9,
                 "details": "", "evidence": [], "nist_controls": ["AC-2"],
                 "mitre_techniques": ["T1548"],
                 "timestamp": datetime.now(timezone.utc).isoformat()},
            ],
        })
        assert result["dod_incident_category"]["category_id"] == "CAT-1"

    def test_8_correlation_rules_defined(self):
        """Exactly 8 correlation rules are defined."""
        assert len(CORRELATION_RULES) == 8

    def test_correlation_rules_have_required_fields(self):
        """Each correlation rule has all required attributes."""
        for rule in CORRELATION_RULES:
            assert isinstance(rule, CorrelationRule)
            assert rule.rule_id.startswith("CR-")
            assert rule.name
            assert rule.description
            assert len(rule.required_indicators) > 0
            assert rule.time_window_minutes > 0
            assert rule.min_match_count > 0
            assert isinstance(rule.severity, EventSeverity)
            assert rule.dod_category in DOD_INCIDENT_CATEGORIES
            assert len(rule.mitre_techniques) > 0

    def test_correlation_rule_fires_on_pattern(self):
        """Correlation rules fire when enough indicators match in window."""
        # CR-002: Credential Compromise Chain needs IND-017 + IND-016 in 60 min
        ts = datetime.now(timezone.utc).isoformat()
        indicators = [
            {"indicator_id": "IND-017", "indicator_name": "Failed Logins",
             "category": "access", "weight": 5.5, "confidence": 0.8,
             "details": "", "evidence": [], "nist_controls": ["AC-2"],
             "mitre_techniques": ["T1110"], "timestamp": ts},
            {"indicator_id": "IND-016", "indicator_name": "Priv Esc",
             "category": "access", "weight": 8.5, "confidence": 0.9,
             "details": "", "evidence": [], "nist_controls": ["AC-2"],
             "mitre_techniques": ["T1548"], "timestamp": ts},
        ]
        result = self.siem.generate_events({
            "user_id": "user-corr",
            "risk_score": 70,
            "threat_level": "ELEVATED",
            "triggered_indicators": indicators,
        })
        # Should fire CR-002
        alert_rule_ids = [a["rule_id"] for a in result.get("correlation_alerts", [])]
        assert "CR-002" in alert_rule_ids

    def test_7_dod_incident_categories(self):
        """All 7 DoD incident categories are defined."""
        assert len(DOD_INCIDENT_CATEGORIES) == 7
        for cat_id in ["CAT-1", "CAT-2", "CAT-3", "CAT-4", "CAT-5", "CAT-6", "CAT-7"]:
            assert cat_id in DOD_INCIDENT_CATEGORIES

    def test_mitre_attack_map_coverage(self):
        """MITRE ATT&CK map covers key techniques."""
        assert len(MITRE_ATTACK_MAP) >= 25
        # Verify key techniques
        for tech_id in ["T1078", "T1548", "T1005", "T1048", "T1110", "T1046"]:
            assert tech_id in MITRE_ATTACK_MAP
            info = MITRE_ATTACK_MAP[tech_id]
            assert "tactic" in info
            assert "name" in info
            assert "url" in info

    def test_event_severity_from_risk_score(self):
        """EventSeverity.from_risk_score maps correctly."""
        assert EventSeverity.from_risk_score(0) == EventSeverity.INFORMATIONAL
        assert EventSeverity.from_risk_score(10) == EventSeverity.LOW
        assert EventSeverity.from_risk_score(30) == EventSeverity.MEDIUM
        assert EventSeverity.from_risk_score(60) == EventSeverity.HIGH
        assert EventSeverity.from_risk_score(85) == EventSeverity.CRITICAL

    def test_batch_export_json(self):
        """Batch export produces valid JSON output."""
        # Generate some events first
        self.siem.generate_events({
            "user_id": "u1", "risk_score": 50, "threat_level": "ADVISORY",
            "triggered_indicators": [],
        })
        result = self.siem.batch_export(output_format="json")
        assert result["format"] == "json"
        assert "event_count" in result

    def test_batch_export_csv(self):
        """Batch export produces valid CSV output."""
        self.siem.generate_events({
            "user_id": "u1", "risk_score": 50, "threat_level": "ADVISORY",
            "triggered_indicators": [],
        })
        result = self.siem.batch_export(output_format="csv")
        assert result["format"] == "csv"
        assert "content" in result
        assert "event_id" in result["content"]

    def test_forwarding_destination_registration(self):
        """Forwarding destinations can be registered."""
        dest = self.siem.add_forwarding_destination(
            "test-siem", "syslog", "10.0.0.1", 514,
            protocol="tcp", format_type="cef",
        )
        assert dest["name"] == "test-siem"
        assert dest["enabled"] is True

    def test_stats_tracking(self):
        """SIEM stats are tracked correctly."""
        self.siem.generate_events({
            "user_id": "u1", "risk_score": 40, "threat_level": "ADVISORY",
            "triggered_indicators": [],
        })
        stats = self.siem.get_stats()
        assert stats["events_generated"] >= 1
        assert "correlation_rules_count" in stats
        assert stats["correlation_rules_count"] == 8

    def test_list_correlation_rules(self):
        """list_correlation_rules returns all 8 rules."""
        rules = self.siem.list_correlation_rules()
        assert len(rules) == 8
        for rule in rules:
            assert "rule_id" in rule
            assert "name" in rule
            assert "description" in rule


# =============================================================================
# Cleared Personnel Analytics Tests
# =============================================================================


class TestClearedPersonnelAnalyzer:
    """Test SEAD 4/6 cleared personnel analytics."""

    def setup_method(self):
        self.analyzer = ClearedPersonnelAnalyzer()

    def test_13_adjudicative_guidelines_defined(self):
        """All 13 SEAD 4 adjudicative guidelines are present."""
        assert len(ADJUDICATIVE_GUIDELINES) == 13
        expected_letters = set("ABCDEFGHIJKLM")
        actual_letters = {g.letter for g in ADJUDICATIVE_GUIDELINES.values()}
        assert actual_letters == expected_letters

    def test_guideline_fields_complete(self):
        """Each guideline has required attributes."""
        for gid, g in ADJUDICATIVE_GUIDELINES.items():
            assert g.guideline_id == gid
            assert g.letter in "ABCDEFGHIJKLM"
            assert g.name
            assert g.description
            assert len(g.risk_indicators) > 0
            assert len(g.mitigating_factors) > 0

    def test_clearance_level_ordering(self):
        """Clearance levels are in ascending order."""
        assert ClearanceLevel.UNCLASSIFIED.value < ClearanceLevel.CUI.value
        assert ClearanceLevel.CUI.value < ClearanceLevel.CONFIDENTIAL.value
        assert ClearanceLevel.CONFIDENTIAL.value < ClearanceLevel.SECRET.value
        assert ClearanceLevel.SECRET.value < ClearanceLevel.TOP_SECRET.value
        assert ClearanceLevel.TOP_SECRET.value < ClearanceLevel.SCI.value
        assert ClearanceLevel.SCI.value < ClearanceLevel.SAP.value

    def test_clearance_level_from_string(self):
        """ClearanceLevel.from_string parses correctly."""
        assert ClearanceLevel.from_string("SECRET") == ClearanceLevel.SECRET
        assert ClearanceLevel.from_string("TS/SCI") == ClearanceLevel.SCI
        assert ClearanceLevel.from_string("TS") == ClearanceLevel.TOP_SECRET
        assert ClearanceLevel.from_string("garbage") == ClearanceLevel.UNCLASSIFIED

    def test_clearance_status_values(self):
        """All expected clearance statuses exist."""
        expected = {"PENDING", "INTERIM", "FINAL", "SUSPENDED",
                    "REVOKED", "EXPIRED", "DENIED"}
        actual = {s.value for s in ClearanceStatus}
        assert actual == expected

    def test_polygraph_types(self):
        """All polygraph types are defined."""
        assert PolygraphType.CI.value == "CI"
        assert PolygraphType.FS.value == "FS"
        assert PolygraphType.LIFESTYLE.value == "LIFESTYLE"

    def test_set_clearance(self):
        """Setting clearance data is reflected in the record."""
        record = self.analyzer.set_clearance(
            "person-001", "TOP SECRET", "FINAL",
            compartments=["HCS", "SI"],
            sap_accesses=["SAP-1"],
            date_granted="2024-01-15",
            sponsoring_agency="DoD",
            investigation_type="T5",
        )
        assert record.clearance_level == ClearanceLevel.TOP_SECRET
        assert record.clearance_status == ClearanceStatus.FINAL
        assert "HCS" in record.compartments
        assert record.sponsoring_agency == "DoD"

    def test_record_polygraph(self):
        """Polygraph data is recorded."""
        self.analyzer.record_polygraph(
            "person-poly", "CI", "2024-06-01", "PASS",
            next_due="2029-06-01",
        )
        record = self.analyzer.get_or_create_record("person-poly")
        assert record.polygraph_type == PolygraphType.CI
        assert record.polygraph_result == "PASS"

    def test_ntk_verification_approved(self):
        """NTK passes for user with correct clearance and compartments."""
        self.analyzer.set_clearance(
            "person-ntk", "TOP SECRET", "FINAL", compartments=["HCS"],
        )
        result = self.analyzer.evaluate_cleared_personnel("person-ntk", {
            "accessed_classification": "SECRET",
            "accessed_compartments": ["HCS"],
            "justification": "Mission requirement",
        })
        ntk = result["need_to_know_verification"]
        assert ntk["verification_status"] == "APPROVED"

    def test_ntk_denied_clearance_insufficient(self):
        """NTK denied when clearance level is insufficient."""
        self.analyzer.set_clearance("person-ntk-low", "CONFIDENTIAL", "FINAL")
        result = self.analyzer.evaluate_cleared_personnel("person-ntk-low", {
            "accessed_classification": "TOP SECRET",
        })
        ntk = result["need_to_know_verification"]
        assert ntk["verification_status"] == "DENIED"

    def test_ntk_denied_unauthorized_compartment(self):
        """NTK denied for unauthorized compartment access."""
        self.analyzer.set_clearance(
            "person-ntk-comp", "TOP SECRET", "FINAL", compartments=["HCS"],
        )
        result = self.analyzer.evaluate_cleared_personnel("person-ntk-comp", {
            "accessed_compartments": ["SI", "TK"],
        })
        ntk = result["need_to_know_verification"]
        assert ntk["verification_status"] == "DENIED"

    def test_continuous_evaluation_financial(self):
        """CE detects financial stress events."""
        self.analyzer.set_clearance("person-ce-fin", "SECRET", "FINAL")
        result = self.analyzer.evaluate_cleared_personnel("person-ce-fin", {
            "financial_changes": [
                {"type": "bankruptcy", "detail": "Chapter 7 filing"},
            ],
        })
        ce = result["continuous_evaluation"]
        assert ce["total_findings"] > 0
        financial_findings = [
            f for f in ce["findings"] if f["type"] == "ce_financial"
        ]
        assert len(financial_findings) > 0

    def test_continuous_evaluation_unreported_travel(self):
        """CE detects unreported foreign travel."""
        self.analyzer.set_clearance("person-ce-travel", "SECRET", "FINAL")
        result = self.analyzer.evaluate_cleared_personnel("person-ce-travel", {
            "foreign_travel": [
                {"destination": "Iran", "travel_date": "2024-06-01"},
            ],
        })
        findings = result["findings"]
        travel_findings = [f for f in findings if f["type"] == "ce_unreported_travel"]
        assert len(travel_findings) > 0

    def test_whole_person_assessment(self):
        """WPA evaluates all 13 adjudicative guidelines."""
        self.analyzer.set_clearance("person-wpa", "SECRET", "FINAL")
        result = self.analyzer.evaluate_cleared_personnel("person-wpa", {
            "guideline_data": {
                "F": {
                    "delinquent_debts": True,
                    "unexplained_affluence": True,
                    "good_faith_debt_resolution": True,
                },
            },
        })
        wpa = result["whole_person_assessment"]
        assert wpa["guidelines_evaluated"] == 13
        assert wpa["framework"] == "SEAD 4 Whole Person Assessment"
        # Guideline F should have triggers
        f_assessment = wpa["guideline_assessments"]["SEAD4-F"]
        assert len(f_assessment["risk_indicators_triggered"]) > 0
        assert len(f_assessment["mitigating_factors_active"]) > 0

    def test_sf86_consistency_check(self):
        """SF-86 consistency check detects missing fields."""
        self.analyzer.set_clearance("person-sf86", "SECRET", "FINAL")
        result = self.analyzer.evaluate_cleared_personnel("person-sf86", {
            "sf86_current": {
                "full_name": "John Doe",
                # Missing many required fields
            },
        })
        sf86 = result["sf86_consistency"]
        assert sf86["status"] == "REVIEWED"
        assert sf86["discrepancies_found"] > 0

    def test_polygraph_compliance_missing(self):
        """Polygraph compliance flags missing polygraph for SCI."""
        self.analyzer.set_clearance(
            "person-poly-miss", "SCI", "FINAL",
        )
        result = self.analyzer.evaluate_cleared_personnel(
            "person-poly-miss", {},
        )
        poly = result["polygraph_compliance"]
        poly_findings = [
            f for f in poly["findings"] if f["type"] == "polygraph_missing"
        ]
        assert len(poly_findings) > 0

    def test_clearance_revoked_finding(self):
        """Revoked clearance generates CRITICAL finding."""
        self.analyzer.set_clearance("person-revoked", "SECRET", "REVOKED")
        result = self.analyzer.evaluate_cleared_personnel("person-revoked", {})
        revoked_findings = [
            f for f in result["findings"]
            if f.get("finding", "").startswith("Clearance has been REVOKED")
        ]
        assert len(revoked_findings) > 0
        assert revoked_findings[0]["severity"] == "CRITICAL"

    def test_reporting_compliance_unreported_contact(self):
        """Unreported foreign contact generates finding."""
        self.analyzer.set_clearance("person-contact", "SECRET", "FINAL")
        result = self.analyzer.evaluate_cleared_personnel("person-contact", {
            "foreign_contacts": [
                {"name": "Li Wei", "country": "China", "reported": False},
            ],
        })
        rc = result["reporting_compliance"]
        assert rc["violations_found"] > 0

    def test_personnel_security_action_report(self):
        """PSA report is generated correctly."""
        self.analyzer.set_clearance("person-psa", "SECRET", "FINAL")
        self.analyzer.evaluate_cleared_personnel("person-psa", {})
        report = self.analyzer.generate_personnel_security_action_report(
            "person-psa", "REVIEW", "Routine review"
        )
        assert report["report_id"].startswith("PSA-")
        assert report["report_type"] == "PERSONNEL_SECURITY_ACTION"
        assert "legal_basis" in report
        assert "appeal_rights" in report

    def test_overall_risk_uses_ntas_levels(self):
        """Overall risk levels are NTAS-aligned."""
        self.analyzer.set_clearance("person-ntas", "SECRET", "FINAL")
        result = self.analyzer.evaluate_cleared_personnel("person-ntas", {})
        assert result["overall_risk_level"] in {
            "BASELINE", "ADVISORY", "ELEVATED", "IMMINENT"
        }

    def test_stats(self):
        """Stats are tracked correctly."""
        self.analyzer.set_clearance("person-stats", "SECRET", "FINAL")
        self.analyzer.evaluate_cleared_personnel("person-stats", {})
        stats = self.analyzer.get_stats()
        assert stats["total_evaluations"] >= 1
        assert stats["adjudicative_guidelines"] == 13


# =============================================================================
# Compliance Dashboard Tests
# =============================================================================


class TestComplianceDashboard:
    """Test NITTF maturity model, KRIs, compliance posture, and reporting."""

    def setup_method(self):
        self.dashboard = ComplianceDashboard()

    def test_maturity_levels_defined(self):
        """5 maturity levels from INITIAL to OPTIMIZING."""
        assert len(MaturityLevel) == 5
        assert MaturityLevel.INITIAL.value == 1
        assert MaturityLevel.MANAGED.value == 2
        assert MaturityLevel.DEFINED.value == 3
        assert MaturityLevel.QUANTITATIVELY_MANAGED.value == 4
        assert MaturityLevel.OPTIMIZING.value == 5

    def test_maturity_criteria_per_level(self):
        """Each maturity level has defined criteria."""
        for level in MaturityLevel:
            assert level in MATURITY_CRITERIA
            info = MATURITY_CRITERIA[level]
            assert "name" in info
            assert "description" in info
            assert "criteria" in info
            assert len(info["criteria"]) > 0

    def test_maturity_not_assessed_when_empty(self):
        """Maturity returns NOT_ASSESSED when no criteria are seeded."""
        result = self.dashboard.calculate_maturity_score()
        assert result["assessment_status"] == "NOT_ASSESSED"
        assert result["data_seeded"] is False

    def test_maturity_score_with_criteria(self):
        """Maturity score calculated correctly with seeded criteria."""
        # Seed all INITIAL criteria
        for criterion in MATURITY_CRITERIA[MaturityLevel.INITIAL]["criteria"]:
            self.dashboard.set_maturity_criterion(criterion, True)
        result = self.dashboard.calculate_maturity_score()
        assert result["assessment_status"] == "ASSESSED"
        assert result["data_seeded"] is True
        assert result["maturity_level"] == "INITIAL"
        assert result["overall_score"] > 0

    def test_maturity_advancement(self):
        """Maturity advances when enough criteria are met."""
        # Meet all criteria for INITIAL and MANAGED
        for level in [MaturityLevel.INITIAL, MaturityLevel.MANAGED]:
            for criterion in MATURITY_CRITERIA[level]["criteria"]:
                self.dashboard.set_maturity_criterion(criterion, True)
        result = self.dashboard.calculate_maturity_score()
        assert result["maturity_level_number"] >= 2

    def test_compliance_controls_defined(self):
        """NIST 800-53 PS/PE/AC control families are defined."""
        assert "PS" in COMPLIANCE_CONTROLS
        assert "PE" in COMPLIANCE_CONTROLS
        assert "AC" in COMPLIANCE_CONTROLS
        assert COMPLIANCE_CONTROLS["PS"]["family_name"] == "Personnel Security"

    def test_compliance_not_assessed_when_empty(self):
        """Compliance posture returns NOT_ASSESSED without data."""
        result = self.dashboard.calculate_compliance_posture()
        assert result["assessment_status"] == "NOT_ASSESSED"

    def test_compliance_posture_scoring(self):
        """Compliance posture scores controls correctly."""
        self.dashboard.set_control_assessment(
            "PS-3", "implemented", "Background checks active"
        )
        self.dashboard.set_control_assessment(
            "AC-2", "partially_implemented", "Automated provisioning partial"
        )
        result = self.dashboard.calculate_compliance_posture()
        assert result["assessment_status"] == "ASSESSED"
        assert result["overall_compliance_score"] > 0

    def test_kri_calculation(self):
        """KRIs are calculated with proper structure."""
        # Record some data
        now = datetime.now(timezone.utc)
        self.dashboard.record_detection_time("evt-1", 1800)
        self.dashboard.record_response_time("evt-1", 7200)
        self.dashboard.record_alert_outcome("alert-1", "true_positive")
        self.dashboard.record_alert_outcome("alert-2", "false_positive")

        result = self.dashboard.calculate_kris()
        assert "kris" in result
        assert "mttd" in result["kris"]
        assert "mttr" in result["kris"]
        assert "false_positive_rate" in result["kris"]
        assert "true_positive_rate" in result["kris"]
        assert "alert_volume" in result["kris"]
        assert "overall_health" in result

    def test_kri_has_targets_and_status(self):
        """Each KRI has target values and status indicators."""
        self.dashboard.record_detection_time("evt-1", 100)
        self.dashboard.record_response_time("evt-1", 200)
        self.dashboard.record_alert_outcome("alert-1", "true_positive")
        result = self.dashboard.calculate_kris()
        for kri_name in ["mttd", "mttr", "false_positive_rate", "true_positive_rate"]:
            kri = result["kris"][kri_name]
            assert "target" in kri
            assert "status" in kri
            assert "trend" in kri
            assert "current_7d" in kri
            assert "current_30d" in kri

    def test_model_drift_detection(self):
        """Model drift detection works with sufficient data."""
        # Seed baseline data
        for i in range(20):
            self.dashboard.record_model_metric("isolation_forest", "accuracy", 0.95)
        # Add recent drift
        for i in range(10):
            self.dashboard.record_model_metric("isolation_forest", "accuracy", 0.75)

        result = self.dashboard.detect_model_drift()
        key = "isolation_forest.accuracy"
        assert key in result["metrics"]
        metric_result = result["metrics"][key]
        assert metric_result["status"] == "DRIFT_DETECTED"
        assert abs(metric_result["z_score"]) > 2.0

    def test_model_drift_insufficient_data(self):
        """Model drift returns INSUFFICIENT_DATA with few points."""
        self.dashboard.record_model_metric("new_model", "f1", 0.9)
        result = self.dashboard.detect_model_drift()
        assert result["metrics"]["new_model.f1"]["status"] == "INSUFFICIENT_DATA"

    def test_executive_summary(self):
        """Executive summary contains all sections."""
        report = self.dashboard.generate_executive_summary()
        assert "report_id" in report
        assert report["report_type"] == "EXECUTIVE_SUMMARY"
        assert "program_maturity" in report
        assert "key_risk_indicators" in report
        assert "compliance_posture" in report
        assert "model_health" in report
        assert "recommendations" in report

    def test_executive_summary_recommendations(self):
        """Executive summary includes recommendations."""
        report = self.dashboard.generate_executive_summary()
        assert isinstance(report["recommendations"], list)
        assert len(report["recommendations"]) > 0
        for rec in report["recommendations"]:
            assert "area" in rec
            assert "priority" in rec
            assert "recommendation" in rec

    def test_export_json(self):
        """JSON export contains all metric sections."""
        result = self.dashboard.export_metrics("json")
        assert result["format"] == "json"
        data = result["data"]
        assert "maturity" in data
        assert "kris" in data
        assert "compliance" in data
        assert "model_drift" in data

    def test_export_csv(self):
        """CSV export contains rows."""
        result = self.dashboard.export_metrics("csv")
        assert result["format"] == "csv"
        assert "content" in result
        assert "metric_category" in result["content"]

    def test_export_with_history(self):
        """Export with history includes KRI history."""
        self.dashboard.record_detection_time("e1", 100)
        self.dashboard.calculate_kris()  # Generates a snapshot
        result = self.dashboard.export_metrics("json", include_history=True)
        assert "kri_history" in result["data"]

    def test_stats(self):
        """Dashboard stats reflect recorded data."""
        self.dashboard.record_detection_time("e1", 100)
        self.dashboard.record_response_time("e1", 200)
        self.dashboard.record_alert_outcome("a1", "true_positive")
        self.dashboard.set_maturity_criterion("basic_access_controls", True)
        self.dashboard.set_control_assessment("PS-3", "implemented")
        stats = self.dashboard.get_stats()
        assert stats["detection_times_recorded"] >= 1
        assert stats["response_times_recorded"] >= 1
        assert stats["alert_outcomes_recorded"] >= 1
        assert stats["maturity_criteria_set"] >= 1
        assert stats["control_assessments"] >= 1


# =============================================================================
# NIST Controls & Cross-References
# =============================================================================


class TestNISTControls:
    """Test NIST 800-53 control definitions and cross-references."""

    def test_6_nist_controls_defined(self):
        """6 NIST controls are defined in insider_threat module."""
        assert len(NIST_CONTROLS) == 6
        expected = {"PS-3", "PS-4", "PS-5", "PS-6", "PE-2", "AC-2"}
        assert set(NIST_CONTROLS.keys()) == expected

    def test_nist_controls_have_structure(self):
        """Each control has name, family, description, and indicators."""
        for ctrl_id, ctrl in NIST_CONTROLS.items():
            assert "name" in ctrl
            assert "family" in ctrl
            assert "description" in ctrl
            assert "indicators" in ctrl
            assert len(ctrl["indicators"]) > 0

    def test_compliance_controls_3_families(self):
        """Dashboard covers PS, PE, AC families."""
        assert len(COMPLIANCE_CONTROLS) == 3
        families = set(COMPLIANCE_CONTROLS.keys())
        assert families == {"PS", "PE", "AC"}

    def test_compliance_controls_weighted(self):
        """All dashboard controls have positive weights."""
        for family_id, family in COMPLIANCE_CONTROLS.items():
            for ctrl_id, ctrl in family["controls"].items():
                assert "weight" in ctrl
                assert ctrl["weight"] > 0, f"{ctrl_id} has non-positive weight"


# =============================================================================
# Integration: Server _impl Functions
# =============================================================================


class TestComplianceServerIntegration:
    """Test the _impl functions that bridge compliance modules to MCP tools."""

    def test_assess_insider_threat_impl(self):
        """assess_insider_threat_impl returns valid results."""
        # Import from server (may fail if server.py deps missing, skip gracefully)
        try:
            from server import assess_insider_threat_impl
        except ImportError:
            pytest.skip("server.py dependencies not available")

        result = assess_insider_threat_impl("test-user", {
            "employment_status": "terminated",
        })
        if "error" in result and "not available" in result.get("error", ""):
            pytest.skip("Compliance modules not available in server context")
        assert result.get("available") is True
        assert "risk_score" in result
        assert "threat_level" in result

    def test_generate_siem_events_impl(self):
        """generate_siem_events_impl returns valid results."""
        try:
            from server import generate_siem_events_impl
        except ImportError:
            pytest.skip("server.py dependencies not available")

        result = generate_siem_events_impl({
            "user_id": "test-siem",
            "risk_score": 50,
            "threat_level": "ADVISORY",
            "triggered_indicators": [],
        })
        if "error" in result and "not available" in result.get("error", ""):
            pytest.skip("Compliance modules not available in server context")
        assert result.get("available") is True

    def test_evaluate_cleared_personnel_impl(self):
        """evaluate_cleared_personnel_impl returns valid results."""
        try:
            from server import evaluate_cleared_personnel_impl
        except ImportError:
            pytest.skip("server.py dependencies not available")

        result = evaluate_cleared_personnel_impl("test-person", {})
        if "error" in result and "not available" in result.get("error", ""):
            pytest.skip("Compliance modules not available in server context")
        assert result.get("available") is True

    def test_get_compliance_dashboard_impl(self):
        """get_compliance_dashboard_impl returns valid results."""
        try:
            from server import get_compliance_dashboard_impl
        except ImportError:
            pytest.skip("server.py dependencies not available")

        result = get_compliance_dashboard_impl()
        if "error" in result and "not available" in result.get("error", ""):
            pytest.skip("Compliance modules not available in server context")
        assert result.get("available") is True

    def test_generate_threat_referral_impl(self):
        """generate_threat_referral_impl returns valid results."""
        try:
            from server import generate_threat_referral_impl
        except ImportError:
            pytest.skip("server.py dependencies not available")

        result = generate_threat_referral_impl("test-user", "insider_threat")
        if "error" in result and "not available" in result.get("error", ""):
            pytest.skip("Compliance modules not available in server context")
        assert result.get("available") is True
