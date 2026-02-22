"""
Defense Compliance Modules for Fraud Detection MCP

Provides insider threat detection, SIEM integration, cleared personnel
analytics, and compliance dashboard metrics per Executive Order 13587,
NITTF guidance, SEAD 4/6, and NIST SP 800-53 Rev 5.
"""

from compliance.insider_threat import (
    InsiderThreatAssessor,
    BehavioralIndicator,
    BEHAVIORAL_INDICATORS,
    NIST_CONTROLS,
    ThreatLevel,
    score_to_threat_level,
)

from compliance.siem_integration import (
    SIEMIntegration,
    EventFormatter,
    EventSeverity,
    CorrelationRule,
    CORRELATION_RULES,
    DOD_INCIDENT_CATEGORIES,
    MITRE_ATTACK_MAP,
)

from compliance.cleared_personnel import (
    ClearedPersonnelAnalyzer,
    ClearedPersonnelRecord,
    ClearanceLevel,
    ClearanceStatus,
    PolygraphType,
    ADJUDICATIVE_GUIDELINES,
)

from compliance.dashboard_metrics import (
    ComplianceDashboard,
    MaturityLevel,
    MATURITY_CRITERIA,
    COMPLIANCE_CONTROLS,
)

__all__ = [
    # Insider Threat
    "InsiderThreatAssessor",
    "BehavioralIndicator",
    "BEHAVIORAL_INDICATORS",
    "NIST_CONTROLS",
    "ThreatLevel",
    "score_to_threat_level",
    # SIEM Integration
    "SIEMIntegration",
    "EventFormatter",
    "EventSeverity",
    "CorrelationRule",
    "CORRELATION_RULES",
    "DOD_INCIDENT_CATEGORIES",
    "MITRE_ATTACK_MAP",
    # Cleared Personnel
    "ClearedPersonnelAnalyzer",
    "ClearedPersonnelRecord",
    "ClearanceLevel",
    "ClearanceStatus",
    "PolygraphType",
    "ADJUDICATIVE_GUIDELINES",
    # Dashboard Metrics
    "ComplianceDashboard",
    "MaturityLevel",
    "MATURITY_CRITERIA",
    "COMPLIANCE_CONTROLS",
]
