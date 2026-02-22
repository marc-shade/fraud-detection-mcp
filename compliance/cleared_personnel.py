"""
Cleared Personnel Analytics Module

Monitoring and analytics for users with security clearances.
Implements Continuous Evaluation (CE) hooks per SEAD 6,
Whole Person Assessment per SEAD 4 (13 adjudicative guidelines),
and clearance lifecycle management.

References:
- SEAD 4: National Security Adjudicative Guidelines
- SEAD 6: Continuous Evaluation
- SF-86: Questionnaire for National Security Positions
- NIST SP 800-53 Rev 5: AC-3 (Access Enforcement), AC-25 (Reference Monitor)
- ICD 704: Personnel Security Adjudicative Guidelines
"""

import json
import logging
import threading
import uuid
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Clearance Levels and States
# =============================================================================

class ClearanceLevel(Enum):
    """Security clearance levels in ascending order."""
    UNCLASSIFIED = 0
    CUI = 1             # Controlled Unclassified Information
    CONFIDENTIAL = 2
    SECRET = 3
    TOP_SECRET = 4
    SCI = 5              # Sensitive Compartmented Information
    SAP = 6              # Special Access Program

    @classmethod
    def from_string(cls, level: str) -> "ClearanceLevel":
        """Parse clearance level from string."""
        mapping = {
            "UNCLASSIFIED": cls.UNCLASSIFIED,
            "CUI": cls.CUI,
            "CONFIDENTIAL": cls.CONFIDENTIAL,
            "SECRET": cls.SECRET,
            "TOP SECRET": cls.TOP_SECRET,
            "TS": cls.TOP_SECRET,
            "SCI": cls.SCI,
            "TS/SCI": cls.SCI,
            "SAP": cls.SAP,
        }
        return mapping.get(level.upper().strip(), cls.UNCLASSIFIED)


class ClearanceStatus(Enum):
    """Clearance lifecycle status."""
    PENDING = "PENDING"
    INTERIM = "INTERIM"
    FINAL = "FINAL"
    SUSPENDED = "SUSPENDED"
    REVOKED = "REVOKED"
    EXPIRED = "EXPIRED"
    DENIED = "DENIED"


class PolygraphType(Enum):
    """Types of polygraph examinations."""
    CI = "CI"               # Counter-Intelligence
    FS = "FS"               # Full Scope
    LIFESTYLE = "LIFESTYLE"  # Lifestyle polygraph


# =============================================================================
# SEAD 4: 13 Adjudicative Guidelines (Whole Person Assessment)
# =============================================================================

class AdjudicativeGuideline:
    """Single SEAD 4 adjudicative guideline for the Whole Person Assessment."""

    def __init__(
        self,
        guideline_id: str,
        letter: str,
        name: str,
        description: str,
        risk_indicators: List[str],
        mitigating_factors: List[str],
    ):
        self.guideline_id = guideline_id
        self.letter = letter
        self.name = name
        self.description = description
        self.risk_indicators = risk_indicators
        self.mitigating_factors = mitigating_factors


ADJUDICATIVE_GUIDELINES: Dict[str, AdjudicativeGuideline] = {}


def _register_guideline(
    guideline_id: str,
    letter: str,
    name: str,
    description: str,
    risk_indicators: List[str],
    mitigating_factors: List[str],
) -> AdjudicativeGuideline:
    g = AdjudicativeGuideline(
        guideline_id, letter, name, description,
        risk_indicators, mitigating_factors,
    )
    ADJUDICATIVE_GUIDELINES[guideline_id] = g
    return g


# Guideline A: Allegiance to the United States
_register_guideline(
    "SEAD4-A", "A", "Allegiance to the United States",
    "Conditions that could raise a security concern about an individual's allegiance",
    [
        "involvement_in_acts_against_us",
        "association_with_subversive_groups",
        "advocacy_of_force_against_government",
        "support_for_foreign_preference_over_us",
    ],
    [
        "allegiance_demonstrated_by_service",
        "renunciation_of_prior_associations",
        "lack_of_intent_to_undermine",
    ],
)

# Guideline B: Foreign Influence
_register_guideline(
    "SEAD4-B", "B", "Foreign Influence",
    "Foreign contacts and interests that may present a security concern",
    [
        "close_foreign_family_members",
        "foreign_business_interests",
        "foreign_government_contact",
        "foreign_property_ownership",
        "obligation_to_foreign_entity",
    ],
    [
        "contacts_are_casual_and_infrequent",
        "no_conflict_of_interest",
        "us_ties_outweigh_foreign",
        "foreign_property_minimal_value",
    ],
)

# Guideline C: Foreign Preference
_register_guideline(
    "SEAD4-C", "C", "Foreign Preference",
    "Actions indicating a preference for a foreign country over the United States",
    [
        "dual_citizenship_exercise",
        "foreign_passport_use",
        "foreign_military_service",
        "foreign_voting_participation",
        "foreign_benefits_acceptance",
    ],
    [
        "dual_citizenship_based_on_birth",
        "passport_destroyed_or_surrendered",
        "willingness_to_renounce",
    ],
)

# Guideline D: Sexual Behavior
_register_guideline(
    "SEAD4-D", "D", "Sexual Behavior",
    "Sexual behavior that involves a criminal offense or creates vulnerability",
    [
        "criminal_sexual_conduct",
        "behavior_creating_blackmail_vulnerability",
        "compulsive_or_addictive_behavior",
        "public_sexual_behavior",
    ],
    [
        "behavior_occurred_long_ago",
        "no_evidence_of_vulnerability",
        "successful_treatment",
    ],
)

# Guideline E: Personal Conduct
_register_guideline(
    "SEAD4-E", "E", "Personal Conduct",
    "Conduct involving questionable judgment, lack of candor, or unwillingness to comply",
    [
        "deliberate_omission_or_falsification",
        "pattern_of_dishonesty",
        "violation_of_rules_or_commitments",
        "association_with_persons_engaged_in_criminal_activity",
        "sf86_inconsistencies",
    ],
    [
        "prompt_good_faith_correction",
        "offense_minor_and_isolated",
        "counseling_received_and_positive_changes",
    ],
)

# Guideline F: Financial Considerations
_register_guideline(
    "SEAD4-F", "F", "Financial Considerations",
    "Inability or unwillingness to satisfy debts and meet financial obligations",
    [
        "delinquent_debts",
        "bankruptcy_filing",
        "tax_evasion_or_fraud",
        "unexplained_affluence",
        "financial_irresponsibility",
        "gambling_problems",
    ],
    [
        "conditions_beyond_control",
        "financial_counseling_received",
        "good_faith_debt_resolution",
        "affluence_from_legal_source",
    ],
)

# Guideline G: Alcohol Consumption
_register_guideline(
    "SEAD4-G", "G", "Alcohol Consumption",
    "Excessive alcohol consumption leading to questionable judgment",
    [
        "alcohol_related_incidents",
        "habitual_or_binge_consumption",
        "diagnosis_of_alcohol_use_disorder",
        "relapse_after_treatment",
    ],
    [
        "no_current_alcohol_problem",
        "completion_of_treatment_program",
        "established_pattern_of_responsible_use",
    ],
)

# Guideline H: Drug Involvement and Substance Misuse
_register_guideline(
    "SEAD4-H", "H", "Drug Involvement and Substance Misuse",
    "Use of controlled substances or misuse of prescription drugs",
    [
        "illegal_drug_use",
        "prescription_drug_misuse",
        "drug_use_while_holding_clearance",
        "drug_purchase_or_distribution",
        "positive_drug_test",
    ],
    [
        "experimental_use_long_ago",
        "completion_of_treatment",
        "disassociation_from_drug_users",
        "signed_statement_of_intent",
    ],
)

# Guideline I: Psychological Conditions
_register_guideline(
    "SEAD4-I", "I", "Psychological Conditions",
    "Conditions that may impair judgment, reliability, or trustworthiness",
    [
        "condition_impairing_judgment",
        "failure_to_follow_treatment",
        "opinion_from_qualified_professional",
    ],
    [
        "condition_is_under_treatment",
        "favorable_prognosis",
        "no_indication_of_current_problem",
    ],
)

# Guideline J: Criminal Conduct
_register_guideline(
    "SEAD4-J", "J", "Criminal Conduct",
    "Criminal activity that creates doubt about judgment and trustworthiness",
    [
        "single_serious_crime",
        "multiple_lesser_offenses",
        "allegation_of_criminal_conduct",
        "violation_of_probation",
    ],
    [
        "acquittal_or_charges_dropped",
        "passage_of_time",
        "evidence_of_rehabilitation",
        "pressured_into_criminal_act",
    ],
)

# Guideline K: Handling Protected Information
_register_guideline(
    "SEAD4-K", "K", "Handling Protected Information",
    "Deliberate or negligent mishandling of protected information",
    [
        "unauthorized_disclosure",
        "failure_to_protect_classified",
        "violation_of_markings_or_handling",
        "unauthorized_removal_of_materials",
    ],
    [
        "isolated_and_infrequent",
        "responded_favorably_to_counseling",
        "inadvertent_violation",
    ],
)

# Guideline L: Outside Activities
_register_guideline(
    "SEAD4-L", "L", "Outside Activities",
    "Involvement with foreign governments or entities that may create a conflict",
    [
        "service_for_foreign_government",
        "employment_by_foreign_company",
        "contact_with_foreign_intelligence",
        "unauthorized_technology_transfer",
    ],
    [
        "activity_sanctioned_by_agency",
        "contact_approved_and_reported",
        "no_conflict_of_interest",
    ],
)

# Guideline M: Use of Information Technology
_register_guideline(
    "SEAD4-M", "M", "Use of Information Technology",
    "Failure to comply with rules for information technology systems",
    [
        "unauthorized_system_modification",
        "introduction_of_malicious_code",
        "unauthorized_access_or_use",
        "negligent_security_practices",
    ],
    [
        "minor_and_infrequent",
        "prompt_reporting_of_incident",
        "compliance_with_remedial_actions",
    ],
)


# =============================================================================
# Cleared Personnel Record
# =============================================================================

class ClearedPersonnelRecord:
    """Maintains the security clearance record and CE data for a single individual."""

    def __init__(self, person_id: str):
        self.person_id = person_id
        self._lock = threading.Lock()

        # Clearance data
        self.clearance_level: ClearanceLevel = ClearanceLevel.UNCLASSIFIED
        self.clearance_status: ClearanceStatus = ClearanceStatus.PENDING
        self.compartments: List[str] = []
        self.sap_accesses: List[str] = []
        self.date_granted: Optional[str] = None
        self.date_expires: Optional[str] = None
        self.sponsoring_agency: str = ""
        self.investigation_type: str = ""  # e.g., SSBI, T5, T3

        # Polygraph
        self.polygraph_type: Optional[PolygraphType] = None
        self.polygraph_date: Optional[str] = None
        self.polygraph_result: Optional[str] = None
        self.polygraph_next_due: Optional[str] = None

        # SF-86 tracking
        self.sf86_last_submitted: Optional[str] = None
        self.sf86_next_due: Optional[str] = None
        self.sf86_discrepancies: List[Dict[str, Any]] = []

        # Reporting requirements
        self.reported_foreign_travel: List[Dict[str, Any]] = []
        self.reported_foreign_contacts: List[Dict[str, Any]] = []
        self.reported_financial_changes: List[Dict[str, Any]] = []
        self.reporting_violations: List[Dict[str, Any]] = []

        # Whole Person Assessment scores
        self.guideline_assessments: Dict[str, Dict[str, Any]] = {}

        # CE events
        self.ce_events: deque = deque(maxlen=500)
        self.ce_alerts: List[Dict[str, Any]] = []

        # Need-to-know verification log
        self.ntk_verifications: deque = deque(maxlen=1000)

        # Personnel security actions
        self.security_actions: List[Dict[str, Any]] = []

        self.created_at = datetime.now(timezone.utc).isoformat()
        self.last_evaluated = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the record for reporting."""
        with self._lock:
            return {
                "person_id": self.person_id,
                "clearance_level": self.clearance_level.name,
                "clearance_status": self.clearance_status.value,
                "compartments": self.compartments,
                "sap_accesses": self.sap_accesses,
                "date_granted": self.date_granted,
                "date_expires": self.date_expires,
                "sponsoring_agency": self.sponsoring_agency,
                "investigation_type": self.investigation_type,
                "polygraph_type": self.polygraph_type.value if self.polygraph_type else None,
                "polygraph_date": self.polygraph_date,
                "polygraph_next_due": self.polygraph_next_due,
                "sf86_last_submitted": self.sf86_last_submitted,
                "sf86_next_due": self.sf86_next_due,
                "sf86_discrepancy_count": len(self.sf86_discrepancies),
                "reporting_violation_count": len(self.reporting_violations),
                "ce_event_count": len(self.ce_events),
                "ce_alert_count": len(self.ce_alerts),
                "guideline_assessments_completed": len(self.guideline_assessments),
                "security_action_count": len(self.security_actions),
                "created_at": self.created_at,
                "last_evaluated": self.last_evaluated,
            }


# =============================================================================
# Cleared Personnel Analyzer (Core Engine)
# =============================================================================

class ClearedPersonnelAnalyzer:
    """
    Analytics engine for monitoring cleared personnel.

    Implements Continuous Evaluation per SEAD 6, Whole Person Assessment
    per SEAD 4, need-to-know verification, and clearance lifecycle management.
    Thread-safe for concurrent evaluation.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._records: Dict[str, ClearedPersonnelRecord] = {}
        self._evaluation_history: deque = deque(maxlen=10000)
        logger.info(
            "ClearedPersonnelAnalyzer initialized with %d adjudicative guidelines",
            len(ADJUDICATIVE_GUIDELINES)
        )

    def get_or_create_record(self, person_id: str) -> ClearedPersonnelRecord:
        """Get existing record or create a new one."""
        with self._lock:
            if person_id not in self._records:
                self._records[person_id] = ClearedPersonnelRecord(person_id)
            return self._records[person_id]

    def set_clearance(
        self,
        person_id: str,
        level: str,
        status: str = "FINAL",
        compartments: Optional[List[str]] = None,
        sap_accesses: Optional[List[str]] = None,
        date_granted: Optional[str] = None,
        date_expires: Optional[str] = None,
        sponsoring_agency: str = "",
        investigation_type: str = "",
    ) -> ClearedPersonnelRecord:
        """Set or update a person's clearance data."""
        record = self.get_or_create_record(person_id)
        with record._lock:
            record.clearance_level = ClearanceLevel.from_string(level)
            record.clearance_status = ClearanceStatus(status.upper())
            if compartments is not None:
                record.compartments = compartments
            if sap_accesses is not None:
                record.sap_accesses = sap_accesses
            if date_granted:
                record.date_granted = date_granted
            if date_expires:
                record.date_expires = date_expires
            record.sponsoring_agency = sponsoring_agency
            record.investigation_type = investigation_type
        return record

    def record_polygraph(
        self,
        person_id: str,
        polygraph_type: str,
        date: str,
        result: str,
        next_due: Optional[str] = None,
    ) -> None:
        """Record a polygraph examination."""
        record = self.get_or_create_record(person_id)
        with record._lock:
            record.polygraph_type = PolygraphType(polygraph_type.upper())
            record.polygraph_date = date
            record.polygraph_result = result
            record.polygraph_next_due = next_due

    def evaluate_cleared_personnel(
        self,
        person_id: str,
        activity_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Run a comprehensive cleared personnel evaluation.

        Combines Continuous Evaluation checks, Whole Person Assessment,
        need-to-know verification, clearance status validation,
        SF-86 consistency checks, and reporting compliance.

        Args:
            person_id: Personnel identifier
            activity_data: Dictionary with evaluation data fields:
                - accessed_classification: str - classification of accessed data
                - accessed_compartments: list - compartments accessed
                - accessed_resource: str - resource identifier
                - justification: str - stated reason for access
                - foreign_travel: list - recent travel entries
                - foreign_contacts: list - foreign contact entries
                - financial_changes: list - financial change entries
                - guideline_data: dict - data for adjudicative guideline evaluation
                - sf86_current: dict - current SF-86 data for consistency check

        Returns:
            Comprehensive evaluation result with clearance status,
            CE findings, WPA scores, NTK verification, and actions
        """
        evaluation_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        record = self.get_or_create_record(person_id)

        findings: List[Dict[str, Any]] = []

        # --- 1. Clearance Status Validation ---
        clearance_findings = self._validate_clearance_status(record, timestamp)
        findings.extend(clearance_findings)

        # --- 2. Need-to-Know Verification (AC-3, AC-25) ---
        ntk_result = self._verify_need_to_know(record, activity_data)
        findings.extend(ntk_result.get("findings", []))

        # --- 3. Continuous Evaluation Checks (SEAD 6) ---
        ce_result = self._run_continuous_evaluation(record, activity_data, timestamp)
        findings.extend(ce_result.get("findings", []))

        # --- 4. Whole Person Assessment (SEAD 4) ---
        wpa_result = self._run_whole_person_assessment(
            record, activity_data.get("guideline_data", {}), timestamp
        )

        # --- 5. SF-86 Consistency Check ---
        sf86_result = self._check_sf86_consistency(
            record, activity_data.get("sf86_current", {}), timestamp
        )
        findings.extend(sf86_result.get("findings", []))

        # --- 6. Reporting Compliance ---
        reporting_result = self._check_reporting_compliance(record, activity_data, timestamp)
        findings.extend(reporting_result.get("findings", []))

        # --- 7. Polygraph Compliance ---
        poly_result = self._check_polygraph_compliance(record, timestamp)
        findings.extend(poly_result.get("findings", []))

        # Calculate overall risk
        overall_risk = self._calculate_overall_risk(findings, wpa_result)

        # Determine recommended actions
        actions = self._determine_personnel_actions(
            overall_risk, findings, record
        )

        result = {
            "evaluation_id": evaluation_id,
            "person_id": person_id,
            "timestamp": timestamp,
            "clearance_summary": record.to_dict(),
            "overall_risk_score": overall_risk["score"],
            "overall_risk_level": overall_risk["level"],
            "findings": findings,
            "finding_count": len(findings),
            "need_to_know_verification": ntk_result,
            "continuous_evaluation": ce_result,
            "whole_person_assessment": wpa_result,
            "sf86_consistency": sf86_result,
            "reporting_compliance": reporting_result,
            "polygraph_compliance": poly_result,
            "recommended_actions": actions,
            "nist_controls_evaluated": ["AC-3", "AC-25", "PS-3", "PS-5", "PS-6"],
            "evaluation_metadata": {
                "framework": "SEAD 4/6",
                "guidelines_evaluated": len(ADJUDICATIVE_GUIDELINES),
                "nist_reference": "SP 800-53 Rev 5",
            },
        }

        # Update record
        with record._lock:
            record.last_evaluated = timestamp
            record.ce_events.append({
                "evaluation_id": evaluation_id,
                "risk_score": overall_risk["score"],
                "finding_count": len(findings),
                "timestamp": timestamp,
            })

        with self._lock:
            self._evaluation_history.append({
                "evaluation_id": evaluation_id,
                "person_id": person_id,
                "risk_score": overall_risk["score"],
                "risk_level": overall_risk["level"],
                "finding_count": len(findings),
                "timestamp": timestamp,
            })

        logger.info(
            "Cleared personnel evaluation %s for %s: risk=%.1f level=%s findings=%d",
            evaluation_id, person_id, overall_risk["score"],
            overall_risk["level"], len(findings)
        )

        return result

    def _validate_clearance_status(
        self,
        record: ClearedPersonnelRecord,
        timestamp: str,
    ) -> List[Dict[str, Any]]:
        """Validate current clearance status and expiration."""
        findings: List[Dict[str, Any]] = []

        with record._lock:
            status = record.clearance_status

        if status == ClearanceStatus.SUSPENDED:
            findings.append({
                "type": "clearance_status",
                "severity": "HIGH",
                "finding": "Clearance is currently SUSPENDED",
                "detail": "All access should be restricted pending resolution",
                "nist_control": "PS-3",
            })
        elif status == ClearanceStatus.REVOKED:
            findings.append({
                "type": "clearance_status",
                "severity": "CRITICAL",
                "finding": "Clearance has been REVOKED",
                "detail": "All access must be immediately terminated",
                "nist_control": "PS-4",
            })
        elif status == ClearanceStatus.EXPIRED:
            findings.append({
                "type": "clearance_status",
                "severity": "HIGH",
                "finding": "Clearance has EXPIRED",
                "detail": "Reinvestigation required before continued access",
                "nist_control": "PS-3",
            })
        elif status == ClearanceStatus.INTERIM:
            findings.append({
                "type": "clearance_status",
                "severity": "LOW",
                "finding": "Clearance is INTERIM",
                "detail": "Access may be limited pending final adjudication",
                "nist_control": "PS-3",
            })

        # Check expiration date
        with record._lock:
            expires = record.date_expires

        if expires:
            try:
                exp_date = datetime.fromisoformat(expires.replace("Z", "+00:00"))
                now = datetime.now(timezone.utc)
                days_until = (exp_date - now).days
                if days_until < 0:
                    findings.append({
                        "type": "clearance_expiration",
                        "severity": "HIGH",
                        "finding": f"Clearance expired {abs(days_until)} days ago",
                        "detail": "Immediate reinvestigation required",
                        "nist_control": "PS-3",
                    })
                elif days_until < 90:
                    findings.append({
                        "type": "clearance_expiration",
                        "severity": "MEDIUM",
                        "finding": f"Clearance expires in {days_until} days",
                        "detail": "Initiate periodic reinvestigation process",
                        "nist_control": "PS-3",
                    })
            except ValueError:
                pass

        return findings

    def _verify_need_to_know(
        self,
        record: ClearedPersonnelRecord,
        activity_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Verify need-to-know per NIST 800-53 AC-3 and AC-25.

        Checks that the user has both the clearance level AND
        the specific need-to-know for the data being accessed.
        """
        findings: List[Dict[str, Any]] = []
        accessed_classification = activity_data.get("accessed_classification", "")
        accessed_compartments = activity_data.get("accessed_compartments", [])
        accessed_resource = activity_data.get("accessed_resource", "")
        justification = activity_data.get("justification", "")

        verification_status = "APPROVED"

        with record._lock:
            user_level = record.clearance_level
            user_compartments = record.compartments
            user_status = record.clearance_status

        # Check clearance level
        if accessed_classification:
            required_level = ClearanceLevel.from_string(accessed_classification)
            if required_level.value > user_level.value:
                verification_status = "DENIED"
                findings.append({
                    "type": "ntk_clearance_insufficient",
                    "severity": "CRITICAL",
                    "finding": (
                        f"User clearance {user_level.name} insufficient for "
                        f"{accessed_classification} data"
                    ),
                    "detail": "Access denied: clearance level mismatch",
                    "nist_control": "AC-3",
                })

        # Check clearance status
        if user_status not in (ClearanceStatus.FINAL, ClearanceStatus.INTERIM):
            verification_status = "DENIED"
            findings.append({
                "type": "ntk_status_invalid",
                "severity": "HIGH",
                "finding": f"Clearance status is {user_status.value}, access not permitted",
                "detail": "Active clearance required for classified access",
                "nist_control": "AC-3",
            })

        # Check compartment access
        if accessed_compartments:
            unauthorized_compartments = [
                c for c in accessed_compartments if c not in user_compartments
            ]
            if unauthorized_compartments:
                verification_status = "DENIED"
                findings.append({
                    "type": "ntk_compartment_violation",
                    "severity": "CRITICAL",
                    "finding": (
                        f"Unauthorized compartment access: "
                        f"{', '.join(unauthorized_compartments)}"
                    ),
                    "detail": f"User authorized for: {', '.join(user_compartments) or 'none'}",
                    "nist_control": "AC-25",
                })

        # Check justification provided
        if accessed_classification and not justification:
            findings.append({
                "type": "ntk_no_justification",
                "severity": "MEDIUM",
                "finding": "No need-to-know justification provided for classified access",
                "detail": "Policy requires documented justification",
                "nist_control": "AC-3",
            })

        # Log the verification
        with record._lock:
            record.ntk_verifications.append({
                "resource": accessed_resource,
                "classification": accessed_classification,
                "compartments": accessed_compartments,
                "justification": justification,
                "status": verification_status,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        return {
            "verification_status": verification_status,
            "user_clearance": user_level.name,
            "user_status": user_status.value,
            "user_compartments": user_compartments,
            "requested_classification": accessed_classification,
            "requested_compartments": accessed_compartments,
            "justification_provided": bool(justification),
            "findings": findings,
        }

    def _run_continuous_evaluation(
        self,
        record: ClearedPersonnelRecord,
        activity_data: Dict[str, Any],
        timestamp: str,
    ) -> Dict[str, Any]:
        """
        Run Continuous Evaluation checks per SEAD 6.

        CE continuously monitors cleared personnel for security-relevant
        information rather than relying solely on periodic reinvestigation.
        """
        findings: List[Dict[str, Any]] = []
        ce_checks: Dict[str, Any] = {}

        # Check 1: Financial monitoring
        financial_changes = activity_data.get("financial_changes", [])
        if financial_changes:
            for change in financial_changes:
                change_type = change.get("type", "unknown")
                if change_type in ("bankruptcy", "foreclosure", "tax_lien", "garnishment"):
                    findings.append({
                        "type": "ce_financial",
                        "severity": "HIGH",
                        "finding": f"Financial event detected: {change_type}",
                        "detail": change.get("detail", ""),
                        "nist_control": "PS-3",
                        "sead_reference": "SEAD 6 CE",
                    })
                    with record._lock:
                        record.reported_financial_changes.append({
                            **change,
                            "timestamp": timestamp,
                        })
            ce_checks["financial"] = {
                "events_reviewed": len(financial_changes),
                "concerns_found": len([
                    f for f in findings if f["type"] == "ce_financial"
                ]),
            }

        # Check 2: Criminal records
        criminal_events = activity_data.get("criminal_events", [])
        for event in criminal_events:
            findings.append({
                "type": "ce_criminal",
                "severity": "HIGH",
                "finding": f"Criminal event: {event.get('type', 'unknown')}",
                "detail": event.get("detail", ""),
                "nist_control": "PS-3",
                "sead_reference": "SEAD 6 CE",
            })
        ce_checks["criminal"] = {
            "events_reviewed": len(criminal_events),
            "concerns_found": len(criminal_events),
        }

        # Check 3: Foreign travel monitoring
        foreign_travel = activity_data.get("foreign_travel", [])
        if foreign_travel:
            with record._lock:
                reported = {
                    t.get("destination") for t in record.reported_foreign_travel
                }

            for trip in foreign_travel:
                dest = trip.get("destination", "")
                if dest and dest not in reported:
                    findings.append({
                        "type": "ce_unreported_travel",
                        "severity": "MEDIUM",
                        "finding": f"Unreported foreign travel to {dest}",
                        "detail": "Travel not found in reporting database",
                        "nist_control": "PS-3",
                        "sead_reference": "SEAD 6 CE",
                    })
                with record._lock:
                    record.reported_foreign_travel.append({
                        **trip,
                        "timestamp": timestamp,
                    })
            ce_checks["foreign_travel"] = {
                "trips_reviewed": len(foreign_travel),
                "unreported": len([
                    f for f in findings if f["type"] == "ce_unreported_travel"
                ]),
            }

        # Check 4: Foreign contacts
        foreign_contacts = activity_data.get("foreign_contacts", [])
        if foreign_contacts:
            with record._lock:
                reported_contacts = {
                    c.get("name") for c in record.reported_foreign_contacts
                }

            for contact in foreign_contacts:
                name = contact.get("name", "")
                if name and name not in reported_contacts:
                    findings.append({
                        "type": "ce_unreported_contact",
                        "severity": "MEDIUM",
                        "finding": f"Unreported foreign contact: {name}",
                        "detail": f"Country: {contact.get('country', 'unknown')}",
                        "nist_control": "PS-3",
                        "sead_reference": "SEAD 6 CE",
                    })
                with record._lock:
                    record.reported_foreign_contacts.append({
                        **contact,
                        "timestamp": timestamp,
                    })
            ce_checks["foreign_contacts"] = {
                "contacts_reviewed": len(foreign_contacts),
                "unreported": len([
                    f for f in findings if f["type"] == "ce_unreported_contact"
                ]),
            }

        # Check 5: Social media / public records
        public_records_flags = activity_data.get("public_records_flags", [])
        for flag in public_records_flags:
            findings.append({
                "type": "ce_public_record",
                "severity": flag.get("severity", "MEDIUM"),
                "finding": flag.get("finding", "Public record flag"),
                "detail": flag.get("detail", ""),
                "nist_control": "PS-3",
                "sead_reference": "SEAD 6 CE",
            })
        ce_checks["public_records"] = {
            "flags_reviewed": len(public_records_flags),
        }

        return {
            "checks_performed": ce_checks,
            "total_findings": len(findings),
            "findings": findings,
            "ce_standard": "SEAD 6",
        }

    def _run_whole_person_assessment(
        self,
        record: ClearedPersonnelRecord,
        guideline_data: Dict[str, Any],
        timestamp: str,
    ) -> Dict[str, Any]:
        """
        Run Whole Person Assessment per SEAD 4 (13 adjudicative guidelines).

        Evaluates each guideline and produces risk/concern scores.
        """
        assessments: Dict[str, Dict[str, Any]] = {}

        for gid, guideline in ADJUDICATIVE_GUIDELINES.items():
            gl_data = guideline_data.get(guideline.letter, {})
            if not gl_data:
                gl_data = guideline_data.get(gid, {})

            # Evaluate risk indicators
            triggered_risks: List[str] = []
            for indicator in guideline.risk_indicators:
                if gl_data.get(indicator, False):
                    triggered_risks.append(indicator)

            # Evaluate mitigating factors
            active_mitigations: List[str] = []
            for factor in guideline.mitigating_factors:
                if gl_data.get(factor, False):
                    active_mitigations.append(factor)

            # Calculate concern level
            risk_count = len(triggered_risks)
            mitigation_count = len(active_mitigations)
            total_possible = len(guideline.risk_indicators)

            if total_possible > 0:
                raw_risk = risk_count / total_possible
            else:
                raw_risk = 0

            # Apply mitigation reduction
            if mitigation_count > 0 and risk_count > 0:
                mitigation_factor = min(0.6, mitigation_count * 0.15)
                adjusted_risk = raw_risk * (1 - mitigation_factor)
            else:
                adjusted_risk = raw_risk

            concern_level = "NONE"
            if adjusted_risk > 0.6:
                concern_level = "SIGNIFICANT"
            elif adjusted_risk > 0.3:
                concern_level = "MODERATE"
            elif adjusted_risk > 0:
                concern_level = "MINOR"

            assessment = {
                "guideline_id": gid,
                "guideline_letter": guideline.letter,
                "guideline_name": guideline.name,
                "risk_indicators_triggered": triggered_risks,
                "mitigating_factors_active": active_mitigations,
                "raw_risk_score": round(raw_risk, 3),
                "adjusted_risk_score": round(adjusted_risk, 3),
                "concern_level": concern_level,
            }
            assessments[gid] = assessment

        # Update record
        with record._lock:
            record.guideline_assessments = assessments

        # Calculate aggregate WPA score
        scores = [a["adjusted_risk_score"] for a in assessments.values()]
        avg_score = sum(scores) / len(scores) if scores else 0
        max_score = max(scores) if scores else 0
        significant_count = sum(
            1 for a in assessments.values() if a["concern_level"] == "SIGNIFICANT"
        )

        return {
            "framework": "SEAD 4 Whole Person Assessment",
            "guidelines_evaluated": len(assessments),
            "aggregate_risk_score": round(avg_score * 100, 2),
            "peak_guideline_risk": round(max_score * 100, 2),
            "significant_concerns": significant_count,
            "moderate_concerns": sum(
                1 for a in assessments.values() if a["concern_level"] == "MODERATE"
            ),
            "guideline_assessments": assessments,
            "overall_adjudicative_concern": (
                "FAVORABLE" if significant_count == 0 and max_score < 0.3
                else "UNFAVORABLE" if significant_count >= 2 or max_score > 0.6
                else "CONDITIONAL"
            ),
        }

    def _check_sf86_consistency(
        self,
        record: ClearedPersonnelRecord,
        sf86_data: Dict[str, Any],
        timestamp: str,
    ) -> Dict[str, Any]:
        """Check SF-86 data for inconsistencies and completeness."""
        findings: List[Dict[str, Any]] = []

        if not sf86_data:
            return {
                "status": "NO_DATA",
                "findings": [],
                "detail": "No SF-86 data provided for consistency check",
            }

        # Check for missing required fields
        required_fields = [
            "full_name", "date_of_birth", "citizenship", "residence_history",
            "employment_history", "education", "references",
        ]
        missing = [f for f in required_fields if not sf86_data.get(f)]
        if missing:
            findings.append({
                "type": "sf86_incomplete",
                "severity": "MEDIUM",
                "finding": f"SF-86 missing required fields: {', '.join(missing)}",
                "detail": f"{len(missing)} of {len(required_fields)} required fields missing",
                "nist_control": "PS-3",
            })

        # Check for discrepancies against known data
        reported_travel = sf86_data.get("foreign_travel", [])
        with record._lock:
            known_travel = [t.get("destination") for t in record.reported_foreign_travel]

        unreported_in_sf86 = [t for t in known_travel if t and t not in str(reported_travel)]
        if unreported_in_sf86:
            findings.append({
                "type": "sf86_travel_discrepancy",
                "severity": "HIGH",
                "finding": (
                    f"SF-86 does not list {len(unreported_in_sf86)} "
                    f"known foreign travel destinations"
                ),
                "detail": f"Discrepant destinations: {unreported_in_sf86[:5]}",
                "nist_control": "PS-3",
            })

        # Check submission date
        with record._lock:
            last_submitted = record.sf86_last_submitted

        if sf86_data.get("submission_date"):
            with record._lock:
                record.sf86_last_submitted = sf86_data["submission_date"]

        # Track discrepancies
        if findings:
            with record._lock:
                for f in findings:
                    if f["type"].startswith("sf86_"):
                        record.sf86_discrepancies.append({
                            **f,
                            "timestamp": timestamp,
                        })

        return {
            "status": "REVIEWED",
            "fields_checked": len(required_fields),
            "fields_complete": len(required_fields) - len(missing),
            "discrepancies_found": len(findings),
            "findings": findings,
        }

    def _check_reporting_compliance(
        self,
        record: ClearedPersonnelRecord,
        activity_data: Dict[str, Any],
        timestamp: str,
    ) -> Dict[str, Any]:
        """Check compliance with reporting requirements for cleared personnel."""
        findings: List[Dict[str, Any]] = []

        # Check foreign travel reporting timeliness
        recent_travel = activity_data.get("foreign_travel", [])
        for trip in recent_travel:
            reported_date = trip.get("reported_date")
            travel_date = trip.get("travel_date")
            if travel_date and not reported_date:
                findings.append({
                    "type": "reporting_travel_delinquent",
                    "severity": "MEDIUM",
                    "finding": (
                        f"Foreign travel to {trip.get('destination', 'unknown')} "
                        f"not reported"
                    ),
                    "detail": "Cleared personnel must report foreign travel per policy",
                    "nist_control": "PS-6",
                })

        # Check foreign contact reporting
        contacts = activity_data.get("foreign_contacts", [])
        for contact in contacts:
            if not contact.get("reported", False):
                findings.append({
                    "type": "reporting_contact_delinquent",
                    "severity": "MEDIUM",
                    "finding": (
                        f"Foreign contact with {contact.get('name', 'unknown')} "
                        f"({contact.get('country', 'unknown')}) not reported"
                    ),
                    "detail": "Cleared personnel must report close and continuing foreign contacts",
                    "nist_control": "PS-6",
                })

        # Check financial change reporting
        financial = activity_data.get("financial_changes", [])
        for change in financial:
            if change.get("type") in (
                "bankruptcy", "foreclosure", "garnishment"
            ) and not change.get("reported", False):
                findings.append({
                    "type": "reporting_financial_delinquent",
                    "severity": "HIGH",
                    "finding": f"Significant financial event ({change['type']}) not reported",
                    "detail": "Reportable financial events must be disclosed within required timeframe",
                    "nist_control": "PS-3",
                })

        if findings:
            with record._lock:
                for f in findings:
                    record.reporting_violations.append({
                        **f,
                        "timestamp": timestamp,
                    })

        return {
            "travel_items_checked": len(recent_travel),
            "contact_items_checked": len(contacts),
            "financial_items_checked": len(financial),
            "violations_found": len(findings),
            "findings": findings,
        }

    def _check_polygraph_compliance(
        self,
        record: ClearedPersonnelRecord,
        timestamp: str,
    ) -> Dict[str, Any]:
        """Check polygraph examination compliance."""
        findings: List[Dict[str, Any]] = []

        with record._lock:
            poly_type = record.polygraph_type
            poly_date = record.polygraph_date
            poly_next_due = record.polygraph_next_due
            poly_result = record.polygraph_result
            clearance_level = record.clearance_level

        # Check if polygraph is required but not on file
        if clearance_level.value >= ClearanceLevel.SCI.value and poly_type is None:
            findings.append({
                "type": "polygraph_missing",
                "severity": "HIGH",
                "finding": "No polygraph examination on record for SCI-cleared personnel",
                "detail": "SCI access typically requires at minimum a CI polygraph",
                "nist_control": "PS-3",
            })

        # Check if polygraph is overdue
        if poly_next_due:
            try:
                due_date = datetime.fromisoformat(poly_next_due.replace("Z", "+00:00"))
                now = datetime.now(timezone.utc)
                if due_date < now:
                    days_overdue = (now - due_date).days
                    findings.append({
                        "type": "polygraph_overdue",
                        "severity": "MEDIUM",
                        "finding": f"Polygraph examination overdue by {days_overdue} days",
                        "detail": f"Type: {poly_type.value if poly_type else 'Unknown'}, Due: {poly_next_due}",
                        "nist_control": "PS-3",
                    })
            except ValueError:
                pass

        return {
            "polygraph_on_file": poly_type is not None,
            "polygraph_type": poly_type.value if poly_type else None,
            "last_examination": poly_date,
            "next_due": poly_next_due,
            "last_result": poly_result,
            "findings": findings,
        }

    def _calculate_overall_risk(
        self,
        findings: List[Dict[str, Any]],
        wpa_result: Dict[str, Any],
    ) -> Dict[str, str]:
        """Calculate overall risk from all evaluation components."""
        # Weight findings by severity
        severity_weights = {
            "CRITICAL": 25,
            "HIGH": 15,
            "MEDIUM": 8,
            "LOW": 3,
        }

        finding_score = sum(
            severity_weights.get(f.get("severity", "LOW"), 3) for f in findings
        )

        # Include WPA score
        wpa_score = wpa_result.get("aggregate_risk_score", 0)

        # Combine (finding_score has diminishing returns)
        combined = min(100, finding_score + wpa_score * 0.5)

        if combined >= 80:
            level = "SEVERE"
        elif combined >= 60:
            level = "HIGH"
        elif combined >= 40:
            level = "ELEVATED"
        elif combined >= 20:
            level = "GUARDED"
        else:
            level = "LOW"

        return {
            "score": round(combined, 2),
            "level": level,
            "finding_contribution": round(finding_score, 2),
            "wpa_contribution": round(wpa_score * 0.5, 2),
        }

    def _determine_personnel_actions(
        self,
        overall_risk: Dict[str, Any],
        findings: List[Dict[str, Any]],
        record: ClearedPersonnelRecord,
    ) -> List[Dict[str, Any]]:
        """Determine recommended personnel security actions."""
        actions: List[Dict[str, Any]] = []
        level = overall_risk.get("level", "LOW")

        if level == "SEVERE":
            actions.extend([
                {
                    "action": "suspend_clearance",
                    "priority": "IMMEDIATE",
                    "description": "Suspend security clearance pending formal review",
                },
                {
                    "action": "access_revocation",
                    "priority": "IMMEDIATE",
                    "description": "Revoke all classified system access",
                },
                {
                    "action": "formal_adjudication",
                    "priority": "URGENT",
                    "description": "Initiate formal adjudicative review per SEAD 4",
                },
            ])
        elif level == "HIGH":
            actions.extend([
                {
                    "action": "enhanced_monitoring",
                    "priority": "HIGH",
                    "description": "Place under enhanced continuous evaluation monitoring",
                },
                {
                    "action": "access_review",
                    "priority": "HIGH",
                    "description": "Conduct comprehensive access review",
                },
                {
                    "action": "supervisor_interview",
                    "priority": "HIGH",
                    "description": "Conduct supervisor interview regarding concerns",
                },
            ])
        elif level == "ELEVATED":
            actions.extend([
                {
                    "action": "additional_monitoring",
                    "priority": "MEDIUM",
                    "description": "Increase CE monitoring frequency",
                },
                {
                    "action": "counseling",
                    "priority": "MEDIUM",
                    "description": "Provide security awareness counseling",
                },
            ])
        elif level == "GUARDED":
            actions.append({
                "action": "noted_for_review",
                "priority": "LOW",
                "description": "Document findings for next periodic review",
            })
        else:
            actions.append({
                "action": "routine_monitoring",
                "priority": "ROUTINE",
                "description": "Continue routine CE monitoring per SEAD 6",
            })

        # Check for specific reporting violations
        reporting_findings = [f for f in findings if f["type"].startswith("reporting_")]
        if reporting_findings:
            actions.append({
                "action": "reporting_remediation",
                "priority": "MEDIUM",
                "description": (
                    f"Address {len(reporting_findings)} reporting compliance "
                    f"violations with subject"
                ),
            })

        return actions

    def generate_personnel_security_action_report(
        self,
        person_id: str,
        action_type: str = "REVIEW",
        narrative: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a Personnel Security Action report.

        Args:
            person_id: Personnel identifier
            action_type: Type of action (REVIEW, SUSPENSION, REVOCATION, REINSTATEMENT)
            narrative: Free-text narrative from the security officer

        Returns:
            Formatted personnel security action report
        """
        report_id = (
            f"PSA-{datetime.now(timezone.utc).strftime('%Y%m%d')}-"
            f"{uuid.uuid4().hex[:8].upper()}"
        )
        timestamp = datetime.now(timezone.utc).isoformat()

        record = self.get_or_create_record(person_id)

        with record._lock:
            recent_evals = list(record.ce_events)[-5:]
            wpa = dict(record.guideline_assessments)
            discrepancies = list(record.sf86_discrepancies)
            violations = list(record.reporting_violations)

        report = {
            "report_id": report_id,
            "classification": "FOR OFFICIAL USE ONLY",
            "report_type": "PERSONNEL_SECURITY_ACTION",
            "action_type": action_type,
            "generated_at": timestamp,
            "subject": record.to_dict(),
            "recent_evaluations": recent_evals,
            "whole_person_assessment_summary": {
                gid: {
                    "guideline": a.get("guideline_name"),
                    "concern_level": a.get("concern_level"),
                    "adjusted_risk": a.get("adjusted_risk_score"),
                }
                for gid, a in wpa.items()
            },
            "sf86_discrepancies": discrepancies,
            "reporting_violations": violations,
            "narrative": narrative or "No narrative provided.",
            "legal_basis": (
                "This action is taken pursuant to Executive Order 12968, "
                "SEAD 4 (National Security Adjudicative Guidelines), and "
                "agency-specific personnel security regulations."
            ),
            "appeal_rights": (
                "The subject has the right to appeal this action in accordance "
                "with applicable regulations and Executive Order 12968, Section 5.2."
            ),
        }

        with record._lock:
            record.security_actions.append({
                "report_id": report_id,
                "action_type": action_type,
                "timestamp": timestamp,
            })

        logger.info(
            "Generated personnel security action report %s for %s (type: %s)",
            report_id, person_id, action_type
        )

        return report

    def get_stats(self) -> Dict[str, Any]:
        """Get aggregate statistics."""
        with self._lock:
            history = list(self._evaluation_history)

        return {
            "total_evaluations": len(history),
            "personnel_tracked": len(self._records),
            "adjudicative_guidelines": len(ADJUDICATIVE_GUIDELINES),
            "clearance_distribution": self._get_clearance_distribution(),
            "status_distribution": self._get_status_distribution(),
        }

    def _get_clearance_distribution(self) -> Dict[str, int]:
        """Get distribution of clearance levels."""
        dist: Dict[str, int] = defaultdict(int)
        with self._lock:
            for record in self._records.values():
                dist[record.clearance_level.name] += 1
        return dict(dist)

    def _get_status_distribution(self) -> Dict[str, int]:
        """Get distribution of clearance statuses."""
        dist: Dict[str, int] = defaultdict(int)
        with self._lock:
            for record in self._records.values():
                dist[record.clearance_status.value] += 1
        return dict(dist)
