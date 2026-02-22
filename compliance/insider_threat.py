"""
Insider Threat Detection Module

Per Executive Order 13587 and NITTF (National Insider Threat Task Force) guidance.
Implements User Activity Monitoring (UAM) aligned with CNSSD 504 requirements.
Maps behavioral indicators to NIST 800-53 controls: PS-3, PS-4, PS-5, PS-6, PE-2, AC-2.

References:
- Executive Order 13587: Structural Reforms to Improve Security of Classified Networks
- CNSSD 504: Directive on Protecting National Security Systems
- NITTF Insider Threat Guide
- NIST SP 800-53 Rev 5: Security and Privacy Controls
- DHS NTAS Threat Level System
"""

import logging
import math
import threading
import uuid
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Threat Level Definitions (mapped to DHS NTAS levels)
# =============================================================================

class ThreatLevel(Enum):
    """Threat levels mapped to DHS National Terrorism Advisory System."""
    LOW = "LOW"                 # Score 0-20: Normal activity
    GUARDED = "GUARDED"         # Score 21-40: Minor anomalies
    ELEVATED = "ELEVATED"       # Score 41-60: Notable concerns
    HIGH = "HIGH"               # Score 61-80: Serious threat indicators
    SEVERE = "SEVERE"           # Score 81-100: Critical, immediate action required


def score_to_threat_level(score: float) -> ThreatLevel:
    """Map a 0-100 risk score to a DHS NTAS-aligned threat level."""
    if score <= 20:
        return ThreatLevel.LOW
    elif score <= 40:
        return ThreatLevel.GUARDED
    elif score <= 60:
        return ThreatLevel.ELEVATED
    elif score <= 80:
        return ThreatLevel.HIGH
    else:
        return ThreatLevel.SEVERE


# =============================================================================
# NIST 800-53 Control Mappings
# =============================================================================

NIST_CONTROLS = {
    "PS-3": {
        "name": "Personnel Screening",
        "family": "Personnel Security",
        "description": "Screen individuals prior to authorizing access to the system",
        "indicators": [
            "background_check_anomaly", "foreign_contact_violation",
            "financial_stress", "criminal_conduct"
        ]
    },
    "PS-4": {
        "name": "Personnel Termination",
        "family": "Personnel Security",
        "description": "Upon termination, disable system access and retrieve credentials",
        "indicators": [
            "post_termination_access", "credential_retention",
            "data_exfiltration_pre_departure"
        ]
    },
    "PS-5": {
        "name": "Personnel Transfer",
        "family": "Personnel Security",
        "description": "Review and confirm ongoing need for access upon transfer",
        "indicators": [
            "access_outside_new_scope", "retained_old_permissions",
            "cross_department_data_access"
        ]
    },
    "PS-6": {
        "name": "Access Agreements",
        "family": "Personnel Security",
        "description": "Ensure individuals sign appropriate access agreements",
        "indicators": [
            "unsigned_nda", "expired_access_agreement",
            "agreement_violation"
        ]
    },
    "PE-2": {
        "name": "Physical Access Authorizations",
        "family": "Physical and Environmental Protection",
        "description": "Authorize physical access and maintain access lists",
        "indicators": [
            "badge_tailgating", "unauthorized_area_access",
            "after_hours_physical_access", "lost_badge_not_reported"
        ]
    },
    "AC-2": {
        "name": "Account Management",
        "family": "Access Control",
        "description": "Manage information system accounts",
        "indicators": [
            "privilege_escalation", "dormant_account_reactivation",
            "shared_account_usage", "unauthorized_account_creation"
        ]
    },
}


# =============================================================================
# Behavioral Indicators (25+ from NITTF Insider Threat Guide)
# =============================================================================

class BehavioralIndicator:
    """Single behavioral indicator with weight and NIST control mapping."""

    def __init__(
        self,
        indicator_id: str,
        name: str,
        category: str,
        weight: float,
        nist_controls: List[str],
        mitre_techniques: List[str],
        description: str,
        detection_logic: str,
    ):
        self.indicator_id = indicator_id
        self.name = name
        self.category = category
        self.weight = weight
        self.nist_controls = nist_controls
        self.mitre_techniques = mitre_techniques
        self.description = description
        self.detection_logic = detection_logic


# Registry of all behavioral indicators
BEHAVIORAL_INDICATORS: Dict[str, BehavioralIndicator] = {}


def _register_indicator(
    indicator_id: str,
    name: str,
    category: str,
    weight: float,
    nist_controls: List[str],
    mitre_techniques: List[str],
    description: str,
    detection_logic: str,
) -> BehavioralIndicator:
    """Register a behavioral indicator in the global registry."""
    indicator = BehavioralIndicator(
        indicator_id=indicator_id,
        name=name,
        category=category,
        weight=weight,
        nist_controls=nist_controls,
        mitre_techniques=mitre_techniques,
        description=description,
        detection_logic=detection_logic,
    )
    BEHAVIORAL_INDICATORS[indicator_id] = indicator
    return indicator


# --- Access-Based Indicators ---
_register_indicator(
    "IND-001", "Unauthorized Classified Access",
    "access", 9.0, ["AC-2", "PS-3"],
    ["T1078", "T1548"],
    "Unauthorized access attempts to classified systems",
    "Detect access attempts to systems/resources beyond user's clearance level or need-to-know"
)

_register_indicator(
    "IND-002", "After-Hours Access Anomaly",
    "access", 5.0, ["AC-2", "PE-2"],
    ["T1078.004"],
    "Unusual after-hours access patterns deviating from established baseline",
    "Compare login timestamps against user's established working hours profile (>2 std dev)"
)

_register_indicator(
    "IND-003", "Mass Data Download",
    "data_movement", 8.5, ["AC-2"],
    ["T1005", "T1039", "T1119"],
    "Bulk file transfers or mass data downloads exceeding baseline",
    "Monitor data volume per session; alert when >3x user's 30-day moving average"
)

_register_indicator(
    "IND-004", "Access Outside Job Scope",
    "access", 7.0, ["PS-5", "AC-2"],
    ["T1083", "T1135"],
    "Accessing data, systems, or repositories outside assigned job responsibilities",
    "Cross-reference accessed resources against user's role-based access profile"
)

_register_indicator(
    "IND-005", "Unauthorized Removable Media",
    "data_movement", 8.0, ["AC-2", "PE-2"],
    ["T1052", "T1091"],
    "Use of unauthorized removable media (USB, external drives)",
    "Detect USB device connections not on approved device whitelist"
)

_register_indicator(
    "IND-006", "Security Control Bypass",
    "evasion", 9.5, ["AC-2"],
    ["T1562", "T1548", "T1036"],
    "Attempts to bypass, disable, or circumvent security controls",
    "Monitor for DLP bypass, proxy avoidance, AV disablement, firewall rule changes"
)

# --- Travel and Foreign Contact Indicators ---
_register_indicator(
    "IND-007", "Unusual Foreign Travel",
    "foreign_nexus", 6.5, ["PS-3"],
    ["T1583"],
    "Travel patterns to foreign adversary nations or unreported foreign travel",
    "Flag travel to ODNI-designated countries of concern; compare against reported travel"
)

_register_indicator(
    "IND-008", "Foreign Contact Violation",
    "foreign_nexus", 7.5, ["PS-3", "PS-6"],
    ["T1583"],
    "Failure to report foreign contacts as required by security protocols",
    "Cross-reference communication patterns with foreign contact reporting database"
)

# --- Financial and Personal Indicators ---
_register_indicator(
    "IND-009", "Financial Stress Indicators",
    "personal", 4.5, ["PS-3"],
    [],
    "Indicators of financial distress for cleared personnel",
    "Aggregate signals: credit alerts, garnishment filings, bankruptcy indicators"
)

_register_indicator(
    "IND-010", "Disgruntlement Indicators",
    "personal", 5.0, ["PS-3"],
    [],
    "Behavioral patterns indicating workplace disgruntlement or dissatisfaction",
    "NLP analysis of communications, HR incident reports, performance review flags"
)

# --- Counter-Intelligence Indicators ---
_register_indicator(
    "IND-011", "Counter-Intelligence Indicators",
    "ci", 9.0, ["PS-3"],
    ["T1583", "T1589"],
    "Patterns suggesting intelligence gathering for a foreign entity",
    "Detect systematic collection of sensitive data across compartments"
)

_register_indicator(
    "IND-012", "Badge Tailgating",
    "physical", 5.5, ["PE-2"],
    ["T1200"],
    "Physical access anomalies including badge tailgating or piggybacking",
    "Correlate badge-in events with camera analytics; detect double-entries on single badge"
)

# --- Digital Exfiltration Indicators ---
_register_indicator(
    "IND-013", "Email to Personal Account",
    "data_movement", 7.5, ["AC-2"],
    ["T1048", "T1567"],
    "Forwarding sensitive documents or data to personal email accounts",
    "Monitor outbound email for personal domain destinations with attachment analysis"
)

_register_indicator(
    "IND-014", "After-Hours Printing",
    "data_movement", 6.0, ["PE-2", "AC-2"],
    ["T1005"],
    "Printing sensitive documents outside normal working hours",
    "Correlate print job timestamps and classification levels with work schedule"
)

_register_indicator(
    "IND-015", "Unusual VPN Location",
    "access", 6.5, ["AC-2"],
    ["T1133"],
    "VPN connections from unusual or suspicious geographic locations",
    "Compare VPN source IPs against user's established location profile"
)

_register_indicator(
    "IND-016", "Privilege Escalation Attempt",
    "access", 8.5, ["AC-2"],
    ["T1548", "T1068"],
    "Attempts to escalate privileges beyond authorized level",
    "Detect sudo abuse, role change requests, admin tool usage without authorization"
)

# --- Additional NITTF Indicators ---
_register_indicator(
    "IND-017", "Excessive Failed Logins",
    "access", 5.5, ["AC-2"],
    ["T1110"],
    "Repeated failed login attempts suggesting credential compromise or brute force",
    "Alert when failed attempts exceed 5 within 15 minutes for any single account"
)

_register_indicator(
    "IND-018", "Dormant Account Reactivation",
    "access", 7.0, ["AC-2"],
    ["T1078"],
    "Reactivation or use of dormant accounts (>90 days inactive)",
    "Flag accounts dormant >90 days that resume activity; verify authorization"
)

_register_indicator(
    "IND-019", "Cloud Storage Upload",
    "data_movement", 7.0, ["AC-2"],
    ["T1567.002"],
    "Uploading sensitive data to unauthorized cloud storage services",
    "Monitor network traffic for unauthorized cloud storage API calls and large uploads"
)

_register_indicator(
    "IND-020", "Screenshot or Screen Recording",
    "data_movement", 6.5, ["AC-2"],
    ["T1113"],
    "Excessive use of screenshot or screen recording tools on sensitive systems",
    "Monitor screen capture tool invocations; compare against baseline frequency"
)

_register_indicator(
    "IND-021", "Encryption of Non-Standard Data",
    "evasion", 7.5, ["AC-2"],
    ["T1027", "T1560"],
    "Encrypting data using personal keys or non-standard encryption tools",
    "Detect use of personal PGP keys, unauthorized encryption utilities on work systems"
)

_register_indicator(
    "IND-022", "Unauthorized Software Installation",
    "evasion", 6.0, ["AC-2"],
    ["T1204", "T1059"],
    "Installing unauthorized software including hacking tools or data exfiltration utilities",
    "Compare installed software against approved software baseline"
)

_register_indicator(
    "IND-023", "Network Scanning Activity",
    "reconnaissance", 8.0, ["AC-2"],
    ["T1046", "T1018"],
    "Internal network scanning or reconnaissance activity from user workstation",
    "Detect port scanning, service enumeration, or subnet sweeps from non-IT endpoints"
)

_register_indicator(
    "IND-024", "Data Staging",
    "data_movement", 7.5, ["AC-2"],
    ["T1074"],
    "Aggregating or staging data in unusual locations prior to potential exfiltration",
    "Monitor for large file collections in temp directories or non-standard locations"
)

_register_indicator(
    "IND-025", "Post-Termination Access",
    "access", 9.5, ["PS-4"],
    ["T1078"],
    "System access attempts after employment termination or clearance revocation",
    "Cross-reference authentication attempts against HR termination database in real-time"
)

_register_indicator(
    "IND-026", "Credential Sharing",
    "access", 7.0, ["AC-2", "PS-6"],
    ["T1078"],
    "Sharing credentials with unauthorized individuals or concurrent session anomalies",
    "Detect simultaneous sessions from different IPs; monitor for credential handoff patterns"
)

_register_indicator(
    "IND-027", "Clearance Scope Violation",
    "access", 8.5, ["PS-3", "AC-2"],
    ["T1078"],
    "Accessing information above current clearance level or outside compartment",
    "Real-time comparison of data classification against user clearance and SCI access"
)

_register_indicator(
    "IND-028", "Covert Channel Usage",
    "evasion", 8.5, ["AC-2"],
    ["T1071", "T1572"],
    "Using covert channels or steganography to exfiltrate data",
    "Deep packet inspection for DNS tunneling, ICMP covert channels, steganographic content"
)


# =============================================================================
# User Activity Profile
# =============================================================================

class UserActivityProfile:
    """Maintains a rolling activity profile for a single user."""

    def __init__(self, user_id: str, window_days: int = 30):
        self.user_id = user_id
        self.window_days = window_days
        self._lock = threading.Lock()

        # Activity baselines (rolling windows)
        self.login_hours: deque = deque(maxlen=1000)
        self.data_volumes: deque = deque(maxlen=500)
        self.access_resources: deque = deque(maxlen=2000)
        self.vpn_locations: deque = deque(maxlen=200)
        self.print_events: deque = deque(maxlen=200)
        self.failed_logins: deque = deque(maxlen=500)

        # Indicator trigger history
        self.triggered_indicators: List[Dict[str, Any]] = []
        self.risk_score_history: deque = deque(maxlen=100)

        # Metadata
        self.role: str = "unknown"
        self.department: str = "unknown"
        self.clearance_level: str = "UNCLASSIFIED"
        self.authorized_resources: List[str] = []
        self.work_hours: Tuple[int, int] = (7, 19)  # 7am-7pm default
        self.created_at: str = datetime.now(timezone.utc).isoformat()
        self.last_assessed: Optional[str] = None

    def record_login(self, hour: int, ip_address: str, success: bool) -> None:
        """Record a login event."""
        with self._lock:
            self.login_hours.append({
                "hour": hour,
                "ip": ip_address,
                "success": success,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            if not success:
                self.failed_logins.append({
                    "ip": ip_address,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })

    def record_data_transfer(self, volume_bytes: int, destination: str) -> None:
        """Record a data transfer event."""
        with self._lock:
            self.data_volumes.append({
                "volume_bytes": volume_bytes,
                "destination": destination,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

    def record_resource_access(self, resource_id: str, classification: str) -> None:
        """Record a resource access event."""
        with self._lock:
            self.access_resources.append({
                "resource_id": resource_id,
                "classification": classification,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

    def record_vpn_connection(self, source_ip: str, geo_location: str) -> None:
        """Record a VPN connection event."""
        with self._lock:
            self.vpn_locations.append({
                "source_ip": source_ip,
                "geo_location": geo_location,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

    def record_print_event(self, document_classification: str, page_count: int) -> None:
        """Record a print event."""
        with self._lock:
            self.print_events.append({
                "classification": document_classification,
                "page_count": page_count,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

    def get_average_data_volume(self) -> float:
        """Calculate average daily data transfer volume."""
        with self._lock:
            if not self.data_volumes:
                return 0.0
            total = sum(e["volume_bytes"] for e in self.data_volumes)
            return total / max(1, len(self.data_volumes))

    def get_login_hour_distribution(self) -> Dict[int, int]:
        """Get distribution of login hours."""
        with self._lock:
            dist: Dict[int, int] = defaultdict(int)
            for entry in self.login_hours:
                dist[entry["hour"]] += 1
            return dict(dist)

    def get_recent_failed_logins(self, minutes: int = 15) -> int:
        """Count recent failed login attempts."""
        with self._lock:
            cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
            count = 0
            for entry in self.failed_logins:
                try:
                    ts = datetime.fromisoformat(entry["timestamp"].replace("Z", "+00:00"))
                    if ts > cutoff:
                        count += 1
                except (ValueError, KeyError):
                    continue
            return count

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the profile for reporting."""
        return {
            "user_id": self.user_id,
            "role": self.role,
            "department": self.department,
            "clearance_level": self.clearance_level,
            "work_hours": list(self.work_hours),
            "login_events_tracked": len(self.login_hours),
            "data_transfer_events_tracked": len(self.data_volumes),
            "resource_access_events_tracked": len(self.access_resources),
            "triggered_indicator_count": len(self.triggered_indicators),
            "last_assessed": self.last_assessed,
            "created_at": self.created_at,
        }


# =============================================================================
# Insider Threat Assessor (Core Engine)
# =============================================================================

class InsiderThreatAssessor:
    """
    Core insider threat assessment engine per EO 13587 and NITTF guidance.

    Evaluates user activity against 28 behavioral indicators, calculates
    weighted risk scores, determines threat levels, and generates case
    referral reports. Thread-safe for concurrent assessments.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._profiles: Dict[str, UserActivityProfile] = {}
        self._assessment_history: deque = deque(maxlen=10000)
        self._case_referrals: List[Dict[str, Any]] = []
        self._alert_thresholds = {
            ThreatLevel.GUARDED: 21,
            ThreatLevel.ELEVATED: 41,
            ThreatLevel.HIGH: 61,
            ThreatLevel.SEVERE: 81,
        }
        logger.info(
            "InsiderThreatAssessor initialized with %d behavioral indicators",
            len(BEHAVIORAL_INDICATORS)
        )

    def get_or_create_profile(self, user_id: str) -> UserActivityProfile:
        """Get existing profile or create new one."""
        with self._lock:
            if user_id not in self._profiles:
                self._profiles[user_id] = UserActivityProfile(user_id)
            return self._profiles[user_id]

    def update_profile(
        self,
        user_id: str,
        role: Optional[str] = None,
        department: Optional[str] = None,
        clearance_level: Optional[str] = None,
        authorized_resources: Optional[List[str]] = None,
        work_hours: Optional[Tuple[int, int]] = None,
    ) -> UserActivityProfile:
        """Update a user's profile metadata."""
        profile = self.get_or_create_profile(user_id)
        with profile._lock:
            if role is not None:
                profile.role = role
            if department is not None:
                profile.department = department
            if clearance_level is not None:
                profile.clearance_level = clearance_level
            if authorized_resources is not None:
                profile.authorized_resources = authorized_resources
            if work_hours is not None:
                profile.work_hours = work_hours
        return profile

    def _evaluate_indicator(
        self,
        indicator: BehavioralIndicator,
        activity_data: Dict[str, Any],
        profile: UserActivityProfile,
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluate a single behavioral indicator against user activity data.

        Returns a trigger dict if the indicator fires, None otherwise.
        """
        triggered = False
        confidence = 0.0
        details = ""
        evidence: List[str] = []

        ind_id = indicator.indicator_id

        # --- Access-Based Evaluations ---
        if ind_id == "IND-001":
            # Unauthorized classified access
            accessed = activity_data.get("accessed_classification", "")
            user_clearance = profile.clearance_level
            classification_hierarchy = [
                "UNCLASSIFIED", "CUI", "CONFIDENTIAL", "SECRET", "TOP SECRET", "SCI"
            ]
            user_level = classification_hierarchy.index(user_clearance) \
                if user_clearance in classification_hierarchy else 0
            accessed_level = classification_hierarchy.index(accessed) \
                if accessed in classification_hierarchy else -1
            if accessed_level > user_level:
                triggered = True
                confidence = 0.95
                details = f"User with {user_clearance} clearance accessed {accessed} data"
                evidence.append(f"Classification breach: {user_clearance} -> {accessed}")

        elif ind_id == "IND-002":
            # After-hours access
            login_hour = activity_data.get("login_hour")
            if login_hour is not None:
                start, end = profile.work_hours
                if login_hour < start or login_hour > end:
                    hour_dist = profile.get_login_hour_distribution()
                    total = sum(hour_dist.values()) or 1
                    off_hours_freq = sum(
                        v for k, v in hour_dist.items() if k < start or k > end
                    )
                    ratio = off_hours_freq / total
                    if ratio < 0.15:  # Less than 15% off-hours logins historically
                        triggered = True
                        confidence = min(0.9, 0.5 + (0.15 - ratio) * 3)
                        details = f"Login at hour {login_hour} outside work hours ({start}-{end})"
                        evidence.append(f"Off-hours ratio: {ratio:.2%} (threshold: 15%)")

        elif ind_id == "IND-003":
            # Mass data download
            volume = activity_data.get("data_volume_bytes", 0)
            avg = profile.get_average_data_volume()
            if avg > 0 and volume > avg * 3:
                triggered = True
                multiplier = volume / avg
                confidence = min(0.95, 0.5 + (multiplier - 3) * 0.1)
                details = f"Data volume {volume:,} bytes is {multiplier:.1f}x above average"
                evidence.append(f"Average: {avg:,.0f} bytes, Current: {volume:,} bytes")

        elif ind_id == "IND-004":
            # Access outside job scope
            accessed_resource = activity_data.get("resource_id", "")
            if accessed_resource and accessed_resource not in profile.authorized_resources:
                triggered = True
                confidence = 0.80
                details = f"Access to resource '{accessed_resource}' outside authorized scope"
                evidence.append(f"Authorized: {profile.authorized_resources[:5]}")

        elif ind_id == "IND-005":
            # Unauthorized removable media
            device_type = activity_data.get("removable_media_type", "")
            approved_devices = activity_data.get("approved_devices", [])
            device_id = activity_data.get("device_id", "")
            if device_type and device_id not in approved_devices:
                triggered = True
                confidence = 0.90
                details = f"Unauthorized {device_type} device connected: {device_id}"
                evidence.append(f"Device not in approved list of {len(approved_devices)} devices")

        elif ind_id == "IND-006":
            # Security control bypass
            bypass_actions = activity_data.get("security_bypass_actions", [])
            if bypass_actions:
                triggered = True
                confidence = 0.92
                details = f"Security control bypass detected: {', '.join(bypass_actions[:3])}"
                evidence.extend(bypass_actions)

        elif ind_id == "IND-007":
            # Unusual foreign travel
            travel_destination = activity_data.get("travel_destination", "")
            adversary_nations = activity_data.get(
                "adversary_nations",
                ["CN", "RU", "IR", "KP", "CU", "SY", "VE"]
            )
            reported_travel = activity_data.get("reported_travel", [])
            if travel_destination:
                country_code = travel_destination.upper()[:2]
                if country_code in adversary_nations:
                    triggered = True
                    confidence = 0.85
                    details = f"Travel to country of concern: {travel_destination}"
                    evidence.append(f"Destination in adversary list")
                elif travel_destination not in reported_travel:
                    triggered = True
                    confidence = 0.70
                    details = f"Unreported foreign travel to: {travel_destination}"
                    evidence.append("Travel not in reporting database")

        elif ind_id == "IND-008":
            # Foreign contact violation
            foreign_contacts = activity_data.get("foreign_contacts_detected", [])
            reported_contacts = activity_data.get("reported_foreign_contacts", [])
            unreported = [c for c in foreign_contacts if c not in reported_contacts]
            if unreported:
                triggered = True
                confidence = 0.80
                details = f"{len(unreported)} unreported foreign contacts detected"
                evidence.extend(unreported[:3])

        elif ind_id == "IND-009":
            # Financial stress
            financial_flags = activity_data.get("financial_indicators", {})
            flag_count = sum(1 for v in financial_flags.values() if v)
            if flag_count >= 2:
                triggered = True
                confidence = min(0.85, 0.4 + flag_count * 0.15)
                active_flags = [k for k, v in financial_flags.items() if v]
                details = f"Financial stress indicators: {', '.join(active_flags)}"
                evidence.extend(active_flags)

        elif ind_id == "IND-010":
            # Disgruntlement
            disgruntlement_score = activity_data.get("disgruntlement_score", 0)
            hr_incidents = activity_data.get("hr_incidents", 0)
            if disgruntlement_score > 0.6 or hr_incidents >= 3:
                triggered = True
                confidence = min(0.75, max(disgruntlement_score, hr_incidents * 0.2))
                details = f"Disgruntlement score: {disgruntlement_score:.2f}, HR incidents: {hr_incidents}"
                evidence.append(f"Score threshold: 0.6, Incident threshold: 3")

        elif ind_id == "IND-011":
            # Counter-intelligence indicators
            ci_patterns = activity_data.get("ci_indicators", [])
            if ci_patterns:
                triggered = True
                confidence = 0.85
                details = f"Counter-intelligence patterns detected: {len(ci_patterns)} indicators"
                evidence.extend(ci_patterns[:3])

        elif ind_id == "IND-012":
            # Badge tailgating
            badge_anomalies = activity_data.get("badge_anomalies", [])
            if badge_anomalies:
                triggered = True
                confidence = 0.75
                details = f"Physical access anomalies: {len(badge_anomalies)} events"
                evidence.extend(badge_anomalies[:3])

        elif ind_id == "IND-013":
            # Email to personal account
            personal_email_forwards = activity_data.get("personal_email_forwards", 0)
            sensitive_attachments = activity_data.get("sensitive_attachments_forwarded", 0)
            if personal_email_forwards > 0 and sensitive_attachments > 0:
                triggered = True
                confidence = 0.88
                details = (
                    f"{sensitive_attachments} sensitive attachments forwarded "
                    f"to personal email ({personal_email_forwards} total forwards)"
                )
                evidence.append(f"Forwards: {personal_email_forwards}, Sensitive: {sensitive_attachments}")

        elif ind_id == "IND-014":
            # After-hours printing
            print_hour = activity_data.get("print_hour")
            doc_classification = activity_data.get("print_classification", "UNCLASSIFIED")
            if print_hour is not None:
                start, end = profile.work_hours
                if (print_hour < start or print_hour > end) and doc_classification != "UNCLASSIFIED":
                    triggered = True
                    confidence = 0.78
                    details = (
                        f"Printing {doc_classification} document at hour {print_hour} "
                        f"(outside {start}-{end})"
                    )
                    evidence.append(f"Classification: {doc_classification}")

        elif ind_id == "IND-015":
            # Unusual VPN location
            vpn_location = activity_data.get("vpn_location", "")
            known_locations = [e["geo_location"] for e in profile.vpn_locations]
            if vpn_location and known_locations and vpn_location not in known_locations:
                triggered = True
                confidence = 0.72
                details = f"VPN from unknown location: {vpn_location}"
                evidence.append(f"Known locations: {list(set(known_locations))[:5]}")

        elif ind_id == "IND-016":
            # Privilege escalation
            escalation_attempts = activity_data.get("privilege_escalation_attempts", [])
            if escalation_attempts:
                triggered = True
                confidence = 0.90
                details = f"Privilege escalation: {len(escalation_attempts)} attempts"
                evidence.extend(escalation_attempts[:3])

        elif ind_id == "IND-017":
            # Excessive failed logins
            recent_failures = profile.get_recent_failed_logins(minutes=15)
            current_failures = activity_data.get("failed_login_count", 0)
            total = recent_failures + current_failures
            if total >= 5:
                triggered = True
                confidence = min(0.90, 0.5 + total * 0.05)
                details = f"{total} failed login attempts in 15-minute window"
                evidence.append(f"Threshold: 5, Actual: {total}")

        elif ind_id == "IND-018":
            # Dormant account reactivation
            days_inactive = activity_data.get("days_since_last_login", 0)
            if days_inactive > 90:
                triggered = True
                confidence = 0.82
                details = f"Account reactivated after {days_inactive} days of inactivity"
                evidence.append(f"Dormancy threshold: 90 days")

        elif ind_id == "IND-019":
            # Cloud storage upload
            cloud_uploads = activity_data.get("unauthorized_cloud_uploads", [])
            if cloud_uploads:
                triggered = True
                confidence = 0.85
                total_size = sum(u.get("size_bytes", 0) for u in cloud_uploads)
                details = f"{len(cloud_uploads)} unauthorized cloud uploads ({total_size:,} bytes)"
                evidence.extend([u.get("service", "unknown") for u in cloud_uploads[:3]])

        elif ind_id == "IND-020":
            # Screenshot or screen recording
            capture_count = activity_data.get("screen_capture_count", 0)
            baseline_count = activity_data.get("screen_capture_baseline", 5)
            if capture_count > baseline_count * 3:
                triggered = True
                confidence = 0.70
                details = f"{capture_count} screen captures (baseline: {baseline_count})"
                evidence.append(f"Multiplier: {capture_count / max(1, baseline_count):.1f}x")

        elif ind_id == "IND-021":
            # Non-standard encryption
            encryption_tools = activity_data.get("unauthorized_encryption_tools", [])
            if encryption_tools:
                triggered = True
                confidence = 0.82
                details = f"Unauthorized encryption tools used: {', '.join(encryption_tools[:3])}"
                evidence.extend(encryption_tools)

        elif ind_id == "IND-022":
            # Unauthorized software
            unauthorized_sw = activity_data.get("unauthorized_software", [])
            if unauthorized_sw:
                triggered = True
                confidence = 0.78
                details = f"Unauthorized software installed: {', '.join(unauthorized_sw[:3])}"
                evidence.extend(unauthorized_sw)

        elif ind_id == "IND-023":
            # Network scanning
            scan_detected = activity_data.get("network_scanning_detected", False)
            scan_targets = activity_data.get("scan_targets", 0)
            if scan_detected:
                triggered = True
                confidence = 0.88
                details = f"Network scanning activity targeting {scan_targets} hosts"
                evidence.append(f"Scan targets: {scan_targets}")

        elif ind_id == "IND-024":
            # Data staging
            staging_detected = activity_data.get("data_staging_detected", False)
            staged_volume = activity_data.get("staged_data_volume_bytes", 0)
            if staging_detected:
                triggered = True
                confidence = 0.80
                details = f"Data staging detected: {staged_volume:,} bytes in temp locations"
                evidence.append(f"Staged volume: {staged_volume:,} bytes")

        elif ind_id == "IND-025":
            # Post-termination access
            employment_status = activity_data.get("employment_status", "active")
            if employment_status in ("terminated", "revoked", "suspended"):
                triggered = True
                confidence = 0.98
                details = f"Access attempt from {employment_status} personnel"
                evidence.append(f"Status: {employment_status}")

        elif ind_id == "IND-026":
            # Credential sharing
            concurrent_sessions = activity_data.get("concurrent_session_ips", [])
            if len(concurrent_sessions) > 1:
                unique_ips = set(concurrent_sessions)
                if len(unique_ips) > 1:
                    triggered = True
                    confidence = 0.82
                    details = f"Concurrent sessions from {len(unique_ips)} different IPs"
                    evidence.extend(list(unique_ips)[:3])

        elif ind_id == "IND-027":
            # Clearance scope violation
            accessed_compartments = activity_data.get("accessed_compartments", [])
            authorized_compartments = activity_data.get("authorized_compartments", [])
            violations = [c for c in accessed_compartments if c not in authorized_compartments]
            if violations:
                triggered = True
                confidence = 0.93
                details = f"Compartment access violations: {', '.join(violations[:3])}"
                evidence.extend(violations)

        elif ind_id == "IND-028":
            # Covert channel usage
            covert_indicators = activity_data.get("covert_channel_indicators", [])
            if covert_indicators:
                triggered = True
                confidence = 0.87
                details = f"Covert channel indicators: {', '.join(covert_indicators[:3])}"
                evidence.extend(covert_indicators)

        if triggered:
            return {
                "indicator_id": indicator.indicator_id,
                "indicator_name": indicator.name,
                "category": indicator.category,
                "weight": indicator.weight,
                "confidence": confidence,
                "details": details,
                "evidence": evidence,
                "nist_controls": indicator.nist_controls,
                "mitre_techniques": indicator.mitre_techniques,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        return None

    def assess_user(
        self,
        user_id: str,
        activity_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Run a full insider threat assessment against all behavioral indicators.

        Args:
            user_id: The user identifier to assess
            activity_data: Dictionary of current activity data with fields matching
                          indicator detection logic requirements

        Returns:
            Complete assessment with risk score, threat level, triggered indicators,
            NIST control violations, and recommended actions
        """
        assessment_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        profile = self.get_or_create_profile(user_id)

        # Ingest activity data into the profile
        if "login_hour" in activity_data:
            profile.record_login(
                activity_data["login_hour"],
                activity_data.get("source_ip", "unknown"),
                activity_data.get("login_success", True),
            )
        if "data_volume_bytes" in activity_data:
            profile.record_data_transfer(
                activity_data["data_volume_bytes"],
                activity_data.get("data_destination", "unknown"),
            )
        if "resource_id" in activity_data:
            profile.record_resource_access(
                activity_data["resource_id"],
                activity_data.get("accessed_classification", "UNCLASSIFIED"),
            )
        if "vpn_location" in activity_data:
            profile.record_vpn_connection(
                activity_data.get("source_ip", "unknown"),
                activity_data["vpn_location"],
            )
        if "print_hour" in activity_data:
            profile.record_print_event(
                activity_data.get("print_classification", "UNCLASSIFIED"),
                activity_data.get("page_count", 1),
            )

        # Evaluate all indicators
        triggered_indicators: List[Dict[str, Any]] = []
        for indicator in BEHAVIORAL_INDICATORS.values():
            result = self._evaluate_indicator(indicator, activity_data, profile)
            if result is not None:
                triggered_indicators.append(result)

        # Calculate weighted risk score (0-100)
        raw_score = 0.0
        max_possible = sum(ind.weight for ind in BEHAVIORAL_INDICATORS.values())
        if triggered_indicators:
            raw_score = sum(
                t["weight"] * t["confidence"] for t in triggered_indicators
            )
            # Normalize to 0-100 but cap with a sigmoid-like curve
            # This prevents single high-weight indicators from dominating
            normalized = (raw_score / max_possible) * 100
            # Apply a compression curve: sharper at edges
            risk_score = 100 * (1 - math.exp(-normalized / 25))
        else:
            risk_score = 0.0

        risk_score = round(min(100.0, max(0.0, risk_score)), 2)
        threat_level = score_to_threat_level(risk_score)

        # Identify violated NIST controls
        violated_controls: Dict[str, List[str]] = {}
        for trigger in triggered_indicators:
            for ctrl in trigger["nist_controls"]:
                if ctrl not in violated_controls:
                    violated_controls[ctrl] = []
                violated_controls[ctrl].append(trigger["indicator_id"])

        nist_violations = []
        for ctrl_id, indicator_ids in violated_controls.items():
            ctrl_info = NIST_CONTROLS.get(ctrl_id, {})
            nist_violations.append({
                "control_id": ctrl_id,
                "control_name": ctrl_info.get("name", "Unknown"),
                "family": ctrl_info.get("family", "Unknown"),
                "triggered_by": indicator_ids,
            })

        # Determine recommended actions
        recommended_actions = self._determine_actions(
            threat_level, triggered_indicators, profile
        )

        # Build assessment result
        assessment = {
            "assessment_id": assessment_id,
            "user_id": user_id,
            "timestamp": timestamp,
            "risk_score": risk_score,
            "threat_level": threat_level.value,
            "threat_level_description": self._threat_level_description(threat_level),
            "indicators_evaluated": len(BEHAVIORAL_INDICATORS),
            "indicators_triggered": len(triggered_indicators),
            "triggered_indicators": triggered_indicators,
            "nist_control_violations": nist_violations,
            "recommended_actions": recommended_actions,
            "user_profile_summary": profile.to_dict(),
            "assessment_metadata": {
                "framework": "NITTF Insider Threat Guide",
                "executive_order": "EO 13587",
                "uam_standard": "CNSSD 504",
                "indicator_count": len(BEHAVIORAL_INDICATORS),
                "scoring_method": "weighted_indicator_aggregation",
            },
        }

        # Record assessment in history
        with self._lock:
            profile.triggered_indicators.extend(triggered_indicators)
            profile.risk_score_history.append(risk_score)
            profile.last_assessed = timestamp
            self._assessment_history.append({
                "assessment_id": assessment_id,
                "user_id": user_id,
                "risk_score": risk_score,
                "threat_level": threat_level.value,
                "indicators_triggered": len(triggered_indicators),
                "timestamp": timestamp,
            })

        logger.info(
            "Insider threat assessment %s for user %s: score=%.2f level=%s triggered=%d",
            assessment_id, user_id, risk_score, threat_level.value,
            len(triggered_indicators)
        )

        return assessment

    def _threat_level_description(self, level: ThreatLevel) -> str:
        """Human-readable description of threat level."""
        descriptions = {
            ThreatLevel.LOW: "Normal activity. No significant indicators of insider threat.",
            ThreatLevel.GUARDED: "Minor anomalies detected. Continue routine monitoring.",
            ThreatLevel.ELEVATED: (
                "Notable behavioral concerns. Recommend enhanced monitoring "
                "and supervisor notification."
            ),
            ThreatLevel.HIGH: (
                "Serious insider threat indicators present. Recommend immediate "
                "investigation and access review."
            ),
            ThreatLevel.SEVERE: (
                "Critical insider threat assessment. Immediate action required: "
                "suspend access, initiate formal investigation, notify security officer."
            ),
        }
        return descriptions.get(level, "Unknown threat level")

    def _determine_actions(
        self,
        threat_level: ThreatLevel,
        triggered: List[Dict[str, Any]],
        profile: UserActivityProfile,
    ) -> List[Dict[str, Any]]:
        """Determine recommended actions based on threat level and indicators."""
        actions: List[Dict[str, Any]] = []

        if threat_level == ThreatLevel.LOW:
            actions.append({
                "action": "continue_monitoring",
                "priority": "routine",
                "description": "Continue routine UAM monitoring per CNSSD 504",
            })
        elif threat_level == ThreatLevel.GUARDED:
            actions.append({
                "action": "enhanced_monitoring",
                "priority": "low",
                "description": "Increase monitoring frequency for flagged indicators",
            })
        elif threat_level == ThreatLevel.ELEVATED:
            actions.append({
                "action": "supervisor_notification",
                "priority": "medium",
                "description": "Notify user's supervisor of behavioral anomalies",
            })
            actions.append({
                "action": "access_review",
                "priority": "medium",
                "description": "Review and validate current access permissions",
            })
        elif threat_level == ThreatLevel.HIGH:
            actions.append({
                "action": "formal_investigation",
                "priority": "high",
                "description": "Initiate formal insider threat investigation",
            })
            actions.append({
                "action": "access_restriction",
                "priority": "high",
                "description": "Restrict access to sensitive systems pending review",
            })
            actions.append({
                "action": "security_officer_briefing",
                "priority": "high",
                "description": "Brief the Facility Security Officer (FSO)",
            })
        elif threat_level == ThreatLevel.SEVERE:
            actions.append({
                "action": "immediate_access_suspension",
                "priority": "critical",
                "description": "Immediately suspend all system access",
            })
            actions.append({
                "action": "law_enforcement_referral",
                "priority": "critical",
                "description": "Consider referral to law enforcement / CI",
            })
            actions.append({
                "action": "evidence_preservation",
                "priority": "critical",
                "description": "Preserve all logs, communications, and digital evidence",
            })
            actions.append({
                "action": "executive_notification",
                "priority": "critical",
                "description": "Notify CISO/CSO and senior leadership",
            })

        # Add indicator-specific actions
        categories = set(t["category"] for t in triggered)
        if "data_movement" in categories:
            actions.append({
                "action": "dlp_policy_review",
                "priority": "high" if threat_level.value in ("HIGH", "SEVERE") else "medium",
                "description": "Review and tighten DLP policies for this user",
            })
        if "foreign_nexus" in categories:
            actions.append({
                "action": "ci_referral",
                "priority": "high",
                "description": "Refer to Counterintelligence for foreign nexus indicators",
            })
        if "physical" in categories:
            actions.append({
                "action": "physical_security_review",
                "priority": "medium",
                "description": "Coordinate with physical security for access review",
            })

        return actions

    def generate_case_referral(
        self,
        user_id: str,
        assessment_id: Optional[str] = None,
        additional_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a formal insider threat case referral report.

        Args:
            user_id: The user to generate a referral for
            assessment_id: Optional specific assessment to reference
            additional_context: Optional narrative context from the referring analyst

        Returns:
            Structured case referral report suitable for security team action
        """
        referral_id = f"ITCR-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"
        timestamp = datetime.now(timezone.utc).isoformat()

        profile = self.get_or_create_profile(user_id)

        # Gather all triggered indicators for this user
        all_triggers = list(profile.triggered_indicators)

        # Calculate aggregate statistics
        risk_scores = list(profile.risk_score_history)
        avg_risk = sum(risk_scores) / len(risk_scores) if risk_scores else 0
        max_risk = max(risk_scores) if risk_scores else 0
        trend = "stable"
        if len(risk_scores) >= 3:
            recent = risk_scores[-3:]
            if recent[-1] > recent[0] * 1.2:
                trend = "escalating"
            elif recent[-1] < recent[0] * 0.8:
                trend = "declining"

        # Categorize indicators
        category_counts: Dict[str, int] = defaultdict(int)
        for t in all_triggers:
            category_counts[t.get("category", "unknown")] += 1

        # Build referral
        referral = {
            "referral_id": referral_id,
            "classification": "FOR OFFICIAL USE ONLY",
            "generated_at": timestamp,
            "subject_user_id": user_id,
            "subject_profile": profile.to_dict(),
            "referral_type": "INSIDER_THREAT_CASE",
            "executive_summary": (
                f"Insider threat referral for user {user_id}. "
                f"Average risk score: {avg_risk:.1f}/100 ({score_to_threat_level(avg_risk).value}). "
                f"Peak risk score: {max_risk:.1f}/100. "
                f"Trend: {trend}. "
                f"Total indicators triggered: {len(all_triggers)} across "
                f"{len(category_counts)} categories."
            ),
            "risk_summary": {
                "current_threat_level": score_to_threat_level(
                    risk_scores[-1] if risk_scores else 0
                ).value,
                "average_risk_score": round(avg_risk, 2),
                "peak_risk_score": round(max_risk, 2),
                "risk_trend": trend,
                "assessments_conducted": len(risk_scores),
            },
            "indicator_summary": {
                "total_triggers": len(all_triggers),
                "by_category": dict(category_counts),
                "unique_indicators": len(set(t["indicator_id"] for t in all_triggers)),
                "high_weight_triggers": [
                    t for t in all_triggers if t["weight"] >= 8.0
                ],
            },
            "nist_control_impact": self._aggregate_nist_impacts(all_triggers),
            "mitre_techniques": self._aggregate_mitre_techniques(all_triggers),
            "timeline": self._build_indicator_timeline(all_triggers),
            "additional_context": additional_context or "No additional context provided.",
            "recommended_actions": self._determine_actions(
                score_to_threat_level(max_risk),
                all_triggers,
                profile,
            ),
            "legal_notice": (
                "This referral is generated pursuant to Executive Order 13587 and "
                "CNSSD 504 requirements. All information contained herein is "
                "FOR OFFICIAL USE ONLY and must be handled in accordance with "
                "applicable privacy laws, regulations, and organizational policies. "
                "Distribution is limited to authorized insider threat program personnel."
            ),
        }

        with self._lock:
            self._case_referrals.append({
                "referral_id": referral_id,
                "user_id": user_id,
                "timestamp": timestamp,
                "threat_level": referral["risk_summary"]["current_threat_level"],
            })

        logger.info(
            "Generated insider threat case referral %s for user %s",
            referral_id, user_id
        )

        return referral

    def _aggregate_nist_impacts(
        self, triggers: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Aggregate NIST control impacts across all triggered indicators."""
        control_map: Dict[str, List[str]] = defaultdict(list)
        for t in triggers:
            for ctrl in t.get("nist_controls", []):
                control_map[ctrl].append(t["indicator_id"])

        results = []
        for ctrl_id, indicator_ids in sorted(control_map.items()):
            ctrl_info = NIST_CONTROLS.get(ctrl_id, {})
            results.append({
                "control_id": ctrl_id,
                "control_name": ctrl_info.get("name", "Unknown"),
                "family": ctrl_info.get("family", "Unknown"),
                "trigger_count": len(indicator_ids),
                "triggered_by": list(set(indicator_ids)),
            })
        return results

    def _aggregate_mitre_techniques(
        self, triggers: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Aggregate MITRE ATT&CK techniques across triggered indicators."""
        technique_map: Dict[str, List[str]] = defaultdict(list)
        for t in triggers:
            for tech in t.get("mitre_techniques", []):
                technique_map[tech].append(t["indicator_id"])

        results = []
        for tech_id, indicator_ids in sorted(technique_map.items()):
            results.append({
                "technique_id": tech_id,
                "triggered_by": list(set(indicator_ids)),
                "occurrence_count": len(indicator_ids),
            })
        return results

    def _build_indicator_timeline(
        self, triggers: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Build a chronological timeline of indicator triggers."""
        sorted_triggers = sorted(
            triggers, key=lambda t: t.get("timestamp", ""), reverse=True
        )
        timeline = []
        for t in sorted_triggers[:50]:  # Last 50 events
            timeline.append({
                "timestamp": t.get("timestamp", "unknown"),
                "indicator_id": t["indicator_id"],
                "indicator_name": t["indicator_name"],
                "category": t["category"],
                "weight": t["weight"],
                "confidence": t["confidence"],
                "details": t["details"],
            })
        return timeline

    def get_assessment_stats(self) -> Dict[str, Any]:
        """Get aggregate statistics about all assessments."""
        with self._lock:
            history = list(self._assessment_history)

        if not history:
            return {
                "total_assessments": 0,
                "unique_users_assessed": 0,
                "threat_level_distribution": {},
                "average_risk_score": 0,
                "case_referrals_generated": len(self._case_referrals),
            }

        users = set(h["user_id"] for h in history)
        level_dist: Dict[str, int] = defaultdict(int)
        for h in history:
            level_dist[h["threat_level"]] += 1

        scores = [h["risk_score"] for h in history]

        return {
            "total_assessments": len(history),
            "unique_users_assessed": len(users),
            "threat_level_distribution": dict(level_dist),
            "average_risk_score": round(sum(scores) / len(scores), 2),
            "max_risk_score": max(scores),
            "min_risk_score": min(scores),
            "case_referrals_generated": len(self._case_referrals),
            "indicator_registry_size": len(BEHAVIORAL_INDICATORS),
        }

    def list_indicators(self) -> List[Dict[str, Any]]:
        """List all registered behavioral indicators."""
        return [
            {
                "indicator_id": ind.indicator_id,
                "name": ind.name,
                "category": ind.category,
                "weight": ind.weight,
                "nist_controls": ind.nist_controls,
                "mitre_techniques": ind.mitre_techniques,
                "description": ind.description,
            }
            for ind in BEHAVIORAL_INDICATORS.values()
        ]
