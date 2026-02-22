"""
SIEM Integration Module

Defense-grade security event correlation and forwarding.
Supports Common Event Format (CEF), Log Event Extended Format (LEEF),
and Syslog RFC 5424 structured data output.

Maps fraud and insider threat indicators to MITRE ATT&CK technique IDs.
Alert priority classification per DoD 8570/8140 incident categories.

References:
- ArcSight CEF Standard
- IBM QRadar LEEF Format
- RFC 5424: The Syslog Protocol
- MITRE ATT&CK Enterprise Matrix
- DoD 8570.01-M / DoD 8140
- CVSS v3.1 Scoring
"""

import logging
import re
import socket
import ssl
import threading
import uuid
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Event Severity (CVSS-aligned)
# =============================================================================

class EventSeverity(Enum):
    """CVSS v3.1-aligned severity levels."""
    INFORMATIONAL = 0   # CVSS 0.0
    LOW = 1             # CVSS 0.1-3.9
    MEDIUM = 2          # CVSS 4.0-6.9
    HIGH = 3            # CVSS 7.0-8.9
    CRITICAL = 4        # CVSS 9.0-10.0

    @classmethod
    def from_risk_score(cls, score: float) -> "EventSeverity":
        """Map a 0-100 risk score to CVSS-aligned severity."""
        if score <= 0:
            return cls.INFORMATIONAL
        elif score <= 25:
            return cls.LOW
        elif score <= 50:
            return cls.MEDIUM
        elif score <= 75:
            return cls.HIGH
        else:
            return cls.CRITICAL


# =============================================================================
# DoD 8570/8140 Incident Categories
# =============================================================================

DOD_INCIDENT_CATEGORIES = {
    "CAT-1": {
        "name": "Root Level Intrusion",
        "description": "Unauthorized privileged access to a DoD system",
        "response_time": "1 hour",
        "priority": "critical",
        "indicators": [
            "privilege_escalation", "unauthorized_admin_access",
            "rootkit_detection", "system_compromise"
        ],
    },
    "CAT-2": {
        "name": "User Level Intrusion",
        "description": "Unauthorized non-privileged access to a DoD system",
        "response_time": "4 hours",
        "priority": "high",
        "indicators": [
            "unauthorized_access", "credential_theft",
            "session_hijacking", "account_compromise"
        ],
    },
    "CAT-3": {
        "name": "Unsuccessful Activity Attempt",
        "description": "Attempt to gain unauthorized access that was blocked",
        "response_time": "24 hours",
        "priority": "medium",
        "indicators": [
            "failed_login_brute_force", "blocked_exploit",
            "firewall_block", "ids_alert"
        ],
    },
    "CAT-4": {
        "name": "Denial of Service",
        "description": "Activity that impairs normal system functionality",
        "response_time": "2 hours",
        "priority": "high",
        "indicators": [
            "dos_attack", "resource_exhaustion",
            "service_degradation", "bandwidth_flood"
        ],
    },
    "CAT-5": {
        "name": "Non-Compliance Activity",
        "description": "Activity violating security policies",
        "response_time": "72 hours",
        "priority": "low",
        "indicators": [
            "policy_violation", "configuration_drift",
            "unauthorized_software", "removable_media"
        ],
    },
    "CAT-6": {
        "name": "Reconnaissance",
        "description": "Probing or scanning activity",
        "response_time": "24 hours",
        "priority": "medium",
        "indicators": [
            "port_scan", "network_mapping",
            "service_enumeration", "vulnerability_scan"
        ],
    },
    "CAT-7": {
        "name": "Malicious Logic",
        "description": "Installation of malicious software",
        "response_time": "4 hours",
        "priority": "high",
        "indicators": [
            "malware_detection", "trojan",
            "ransomware", "worm"
        ],
    },
}


# =============================================================================
# MITRE ATT&CK Technique Mapping
# =============================================================================

MITRE_ATTACK_MAP: Dict[str, Dict[str, str]] = {
    # Initial Access
    "T1078": {"tactic": "Initial Access", "name": "Valid Accounts", "url": "https://attack.mitre.org/techniques/T1078/"},
    "T1078.004": {"tactic": "Initial Access", "name": "Valid Accounts: Cloud Accounts", "url": "https://attack.mitre.org/techniques/T1078/004/"},
    "T1133": {"tactic": "Initial Access", "name": "External Remote Services", "url": "https://attack.mitre.org/techniques/T1133/"},
    "T1200": {"tactic": "Initial Access", "name": "Hardware Additions", "url": "https://attack.mitre.org/techniques/T1200/"},
    # Execution
    "T1059": {"tactic": "Execution", "name": "Command and Scripting Interpreter", "url": "https://attack.mitre.org/techniques/T1059/"},
    "T1204": {"tactic": "Execution", "name": "User Execution", "url": "https://attack.mitre.org/techniques/T1204/"},
    # Persistence
    "T1098": {"tactic": "Persistence", "name": "Account Manipulation", "url": "https://attack.mitre.org/techniques/T1098/"},
    # Privilege Escalation
    "T1548": {"tactic": "Privilege Escalation", "name": "Abuse Elevation Control Mechanism", "url": "https://attack.mitre.org/techniques/T1548/"},
    "T1068": {"tactic": "Privilege Escalation", "name": "Exploitation for Privilege Escalation", "url": "https://attack.mitre.org/techniques/T1068/"},
    # Defense Evasion
    "T1562": {"tactic": "Defense Evasion", "name": "Impair Defenses", "url": "https://attack.mitre.org/techniques/T1562/"},
    "T1036": {"tactic": "Defense Evasion", "name": "Masquerading", "url": "https://attack.mitre.org/techniques/T1036/"},
    "T1027": {"tactic": "Defense Evasion", "name": "Obfuscated Files or Information", "url": "https://attack.mitre.org/techniques/T1027/"},
    # Discovery
    "T1083": {"tactic": "Discovery", "name": "File and Directory Discovery", "url": "https://attack.mitre.org/techniques/T1083/"},
    "T1135": {"tactic": "Discovery", "name": "Network Share Discovery", "url": "https://attack.mitre.org/techniques/T1135/"},
    "T1046": {"tactic": "Discovery", "name": "Network Service Discovery", "url": "https://attack.mitre.org/techniques/T1046/"},
    "T1018": {"tactic": "Discovery", "name": "Remote System Discovery", "url": "https://attack.mitre.org/techniques/T1018/"},
    # Collection
    "T1005": {"tactic": "Collection", "name": "Data from Local System", "url": "https://attack.mitre.org/techniques/T1005/"},
    "T1039": {"tactic": "Collection", "name": "Data from Network Shared Drive", "url": "https://attack.mitre.org/techniques/T1039/"},
    "T1119": {"tactic": "Collection", "name": "Automated Collection", "url": "https://attack.mitre.org/techniques/T1119/"},
    "T1074": {"tactic": "Collection", "name": "Data Staged", "url": "https://attack.mitre.org/techniques/T1074/"},
    "T1113": {"tactic": "Collection", "name": "Screen Capture", "url": "https://attack.mitre.org/techniques/T1113/"},
    "T1560": {"tactic": "Collection", "name": "Archive Collected Data", "url": "https://attack.mitre.org/techniques/T1560/"},
    # Command and Control
    "T1071": {"tactic": "Command and Control", "name": "Application Layer Protocol", "url": "https://attack.mitre.org/techniques/T1071/"},
    "T1572": {"tactic": "Command and Control", "name": "Protocol Tunneling", "url": "https://attack.mitre.org/techniques/T1572/"},
    # Exfiltration
    "T1048": {"tactic": "Exfiltration", "name": "Exfiltration Over Alternative Protocol", "url": "https://attack.mitre.org/techniques/T1048/"},
    "T1052": {"tactic": "Exfiltration", "name": "Exfiltration Over Physical Medium", "url": "https://attack.mitre.org/techniques/T1052/"},
    "T1567": {"tactic": "Exfiltration", "name": "Exfiltration Over Web Service", "url": "https://attack.mitre.org/techniques/T1567/"},
    "T1567.002": {"tactic": "Exfiltration", "name": "Exfiltration to Cloud Storage", "url": "https://attack.mitre.org/techniques/T1567/002/"},
    "T1091": {"tactic": "Lateral Movement", "name": "Replication Through Removable Media", "url": "https://attack.mitre.org/techniques/T1091/"},
    # Credential Access
    "T1110": {"tactic": "Credential Access", "name": "Brute Force", "url": "https://attack.mitre.org/techniques/T1110/"},
    # Resource Development
    "T1583": {"tactic": "Resource Development", "name": "Acquire Infrastructure", "url": "https://attack.mitre.org/techniques/T1583/"},
    "T1589": {"tactic": "Resource Development", "name": "Gather Victim Identity Information", "url": "https://attack.mitre.org/techniques/T1589/"},
}


# =============================================================================
# Correlation Rule Engine
# =============================================================================

class CorrelationRule:
    """A rule that detects multi-indicator patterns within a time window."""

    def __init__(
        self,
        rule_id: str,
        name: str,
        description: str,
        required_indicators: List[str],
        time_window_minutes: int,
        min_match_count: int,
        severity: EventSeverity,
        dod_category: str,
        mitre_techniques: List[str],
    ):
        self.rule_id = rule_id
        self.name = name
        self.description = description
        self.required_indicators = required_indicators
        self.time_window_minutes = time_window_minutes
        self.min_match_count = min_match_count
        self.severity = severity
        self.dod_category = dod_category
        self.mitre_techniques = mitre_techniques


# Built-in correlation rules
CORRELATION_RULES: List[CorrelationRule] = [
    CorrelationRule(
        "CR-001", "Data Exfiltration Sequence",
        "Detects reconnaissance followed by data staging and exfiltration within a time window",
        ["IND-023", "IND-024", "IND-003", "IND-013"],
        time_window_minutes=480,  # 8 hours
        min_match_count=3,
        severity=EventSeverity.CRITICAL,
        dod_category="CAT-1",
        mitre_techniques=["T1046", "T1074", "T1005", "T1048"],
    ),
    CorrelationRule(
        "CR-002", "Credential Compromise Chain",
        "Multiple failed logins followed by successful access and privilege escalation",
        ["IND-017", "IND-018", "IND-016", "IND-006"],
        time_window_minutes=60,
        min_match_count=2,
        severity=EventSeverity.HIGH,
        dod_category="CAT-2",
        mitre_techniques=["T1110", "T1078", "T1548"],
    ),
    CorrelationRule(
        "CR-003", "Foreign Intelligence Pattern",
        "Foreign contact violations combined with classified access and data movement",
        ["IND-008", "IND-001", "IND-003", "IND-011"],
        time_window_minutes=1440,  # 24 hours
        min_match_count=2,
        severity=EventSeverity.CRITICAL,
        dod_category="CAT-1",
        mitre_techniques=["T1583", "T1589", "T1005"],
    ),
    CorrelationRule(
        "CR-004", "Pre-Departure Exfiltration",
        "Signs of data collection and removal before employment termination",
        ["IND-003", "IND-005", "IND-019", "IND-024", "IND-014"],
        time_window_minutes=10080,  # 7 days
        min_match_count=3,
        severity=EventSeverity.HIGH,
        dod_category="CAT-1",
        mitre_techniques=["T1005", "T1052", "T1567", "T1074"],
    ),
    CorrelationRule(
        "CR-005", "Evasion and Exfiltration",
        "Security control bypass combined with covert channel or encrypted exfiltration",
        ["IND-006", "IND-021", "IND-028"],
        time_window_minutes=240,  # 4 hours
        min_match_count=2,
        severity=EventSeverity.CRITICAL,
        dod_category="CAT-1",
        mitre_techniques=["T1562", "T1027", "T1071", "T1572"],
    ),
    CorrelationRule(
        "CR-006", "After-Hours Suspicious Activity",
        "After-hours access combined with printing, data downloads, or email forwarding",
        ["IND-002", "IND-014", "IND-003", "IND-013"],
        time_window_minutes=480,  # 8 hours
        min_match_count=2,
        severity=EventSeverity.MEDIUM,
        dod_category="CAT-5",
        mitre_techniques=["T1078.004", "T1005"],
    ),
    CorrelationRule(
        "CR-007", "Insider Recon and Staging",
        "Network scanning or file discovery followed by data staging",
        ["IND-023", "IND-004", "IND-024"],
        time_window_minutes=360,  # 6 hours
        min_match_count=2,
        severity=EventSeverity.HIGH,
        dod_category="CAT-6",
        mitre_techniques=["T1046", "T1083", "T1074"],
    ),
    CorrelationRule(
        "CR-008", "Compartment Breach Sequence",
        "Clearance scope violations combined with unauthorized access patterns",
        ["IND-027", "IND-001", "IND-004"],
        time_window_minutes=1440,  # 24 hours
        min_match_count=2,
        severity=EventSeverity.CRITICAL,
        dod_category="CAT-1",
        mitre_techniques=["T1078", "T1083"],
    ),
]


# =============================================================================
# Event Formatters
# =============================================================================

def _escape_cef_value(value: str) -> str:
    """Escape special characters for CEF format."""
    value = value.replace("\\", "\\\\")
    value = value.replace("|", "\\|")
    value = value.replace("=", "\\=")
    value = value.replace("\n", "\\n")
    value = value.replace("\r", "\\r")
    return value


def _escape_leef_value(value: str) -> str:
    """Escape special characters for LEEF format."""
    value = value.replace("\\", "\\\\")
    value = value.replace("\t", "\\t")
    value = value.replace("\n", "\\n")
    value = value.replace("\r", "\\r")
    return value


class EventFormatter:
    """Formats security events in CEF, LEEF, and Syslog RFC 5424 formats."""

    CEF_VERSION = "0"
    DEVICE_VENDOR = "2AcreStudios"
    DEVICE_PRODUCT = "FraudDetectionMCP"
    DEVICE_VERSION = "2.0.0"

    LEEF_VERSION = "2.0"

    @classmethod
    def to_cef(
        cls,
        event_id: str,
        name: str,
        severity: EventSeverity,
        extensions: Dict[str, Any],
    ) -> str:
        """
        Format event as Common Event Format (CEF) for ArcSight.

        CEF:Version|Device Vendor|Device Product|Device Version|Device Event Class ID|Name|Severity|[Extension]
        """
        # CEF severity: 0-3 Low, 4-6 Medium, 7-8 High, 9-10 Critical
        severity_map = {
            EventSeverity.INFORMATIONAL: 0,
            EventSeverity.LOW: 2,
            EventSeverity.MEDIUM: 5,
            EventSeverity.HIGH: 8,
            EventSeverity.CRITICAL: 10,
        }
        sev_val = severity_map.get(severity, 0)

        header = (
            f"CEF:{cls.CEF_VERSION}|{cls.DEVICE_VENDOR}|{cls.DEVICE_PRODUCT}|"
            f"{cls.DEVICE_VERSION}|{_escape_cef_value(event_id)}|"
            f"{_escape_cef_value(name)}|{sev_val}"
        )

        # Build extension key-value pairs
        ext_parts = []
        for key, value in extensions.items():
            safe_key = re.sub(r'[^a-zA-Z0-9]', '', key)
            safe_val = _escape_cef_value(str(value))
            ext_parts.append(f"{safe_key}={safe_val}")

        ext_str = " ".join(ext_parts)
        return f"{header}|{ext_str}"

    @classmethod
    def to_leef(
        cls,
        event_id: str,
        name: str,
        severity: EventSeverity,
        attributes: Dict[str, Any],
    ) -> str:
        """
        Format event as Log Event Extended Format (LEEF) for IBM QRadar.

        LEEF:Version|Vendor|Product|Version|EventID|delimiter|Attributes
        """
        severity_map = {
            EventSeverity.INFORMATIONAL: 1,
            EventSeverity.LOW: 3,
            EventSeverity.MEDIUM: 5,
            EventSeverity.HIGH: 8,
            EventSeverity.CRITICAL: 10,
        }
        sev_val = severity_map.get(severity, 1)

        # LEEF uses tab as default delimiter
        delimiter = "\t"

        header = (
            f"LEEF:{cls.LEEF_VERSION}|{cls.DEVICE_VENDOR}|{cls.DEVICE_PRODUCT}|"
            f"{cls.DEVICE_VERSION}|{event_id}|"
        )

        # Add severity to attributes
        attrs = {"sev": sev_val, "eventName": name}
        attrs.update(attributes)

        attr_parts = []
        for key, value in attrs.items():
            safe_val = _escape_leef_value(str(value))
            attr_parts.append(f"{key}={safe_val}")

        return header + delimiter.join(attr_parts)

    @classmethod
    def to_syslog_rfc5424(
        cls,
        event_id: str,
        name: str,
        severity: EventSeverity,
        structured_data: Dict[str, Any],
        hostname: str = "fraud-detection-mcp",
        app_name: str = "insider-threat",
    ) -> str:
        """
        Format event as RFC 5424 Syslog message with structured data.

        <PRI>VERSION TIMESTAMP HOSTNAME APP-NAME PROCID MSGID [SD-ID SD-PARAM...] MSG
        """
        # RFC 5424 severity mapping (0=Emergency ... 7=Debug)
        severity_map = {
            EventSeverity.CRITICAL: 2,     # Critical
            EventSeverity.HIGH: 3,         # Error
            EventSeverity.MEDIUM: 4,       # Warning
            EventSeverity.LOW: 5,          # Notice
            EventSeverity.INFORMATIONAL: 6,  # Informational
        }
        sev_val = severity_map.get(severity, 6)

        # Facility 4 = security/authorization, Priority = Facility * 8 + Severity
        facility = 4
        pri = facility * 8 + sev_val

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        procid = "-"
        msgid = event_id

        # Build structured data elements
        sd_parts = []
        sd_id = "insider-threat@2acrestudios"
        params = []
        for key, value in structured_data.items():
            safe_key = re.sub(r'[^a-zA-Z0-9._-]', '', str(key))[:32]
            safe_val = str(value).replace("\\", "\\\\").replace('"', '\\"').replace("]", "\\]")
            params.append(f'{safe_key}="{safe_val}"')
        sd_element = f"[{sd_id} {' '.join(params)}]"
        sd_parts.append(sd_element)

        sd_string = "".join(sd_parts) if sd_parts else "-"

        return (
            f"<{pri}>1 {timestamp} {hostname} {app_name} {procid} "
            f"{msgid} {sd_string} {name}"
        )


# =============================================================================
# SIEM Integration Engine
# =============================================================================

class SIEMIntegration:
    """
    Defense-grade SIEM integration engine.

    Generates security events in multiple formats, runs correlation rules,
    enriches events with MITRE ATT&CK mappings, and manages event forwarding.
    Thread-safe for concurrent event generation.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._event_buffer: deque = deque(maxlen=50000)
        self._correlation_buffer: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )
        self._alert_history: deque = deque(maxlen=10000)
        self._forwarding_destinations: List[Dict[str, Any]] = []
        self._stats = {
            "events_generated": 0,
            "events_forwarded": 0,
            "correlation_alerts": 0,
            "cef_events": 0,
            "leef_events": 0,
            "syslog_events": 0,
        }
        logger.info("SIEMIntegration engine initialized")

    def add_forwarding_destination(
        self,
        name: str,
        destination_type: str,
        host: str,
        port: int,
        protocol: str = "tcp",
        format_type: str = "cef",
        enabled: bool = True,
    ) -> Dict[str, Any]:
        """
        Register a forwarding destination for SIEM events.

        The destination is configured but not connected until events are
        forwarded. Each call to generate_events() will attempt to send
        formatted events to all enabled destinations via TCP or TLS sockets.

        Args:
            name: Human-readable destination name
            destination_type: e.g. "syslog", "siem", "splunk"
            host: Destination hostname or IP
            port: Destination port
            protocol: "tcp" or "tls"
            format_type: Event format to send: "cef", "leef", or "syslog"
            enabled: Whether this destination is active

        Returns:
            The registered destination configuration
        """
        dest = {
            "name": name,
            "type": destination_type,
            "host": host,
            "port": port,
            "protocol": protocol,
            "format": format_type,
            "enabled": enabled,
            "connected": False,
            "last_error": None,
            "events_sent": 0,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with self._lock:
            self._forwarding_destinations.append(dest)
        logger.info(
            "Added forwarding destination '%s' (%s:%d via %s, format=%s)",
            name, host, port, protocol, format_type,
        )
        return dest

    def _forward_events(self, events: Dict[str, List[str]]) -> int:
        """
        Forward formatted events to all enabled destinations.

        Opens a TCP or TLS socket for each destination, sends events in the
        destination's configured format, and closes the connection.

        Args:
            events: Dict mapping format name ("cef", "leef", "syslog") to
                    lists of formatted event strings.

        Returns:
            Total number of events successfully forwarded across all destinations.
        """
        total_forwarded = 0

        with self._lock:
            destinations = list(self._forwarding_destinations)

        for dest in destinations:
            if not dest.get("enabled", False):
                continue

            fmt = dest.get("format", "cef")
            event_list = events.get(fmt, [])
            if not event_list:
                continue

            host = dest["host"]
            port = dest["port"]
            protocol = dest.get("protocol", "tcp")
            sock = None

            try:
                raw_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                raw_sock.settimeout(10.0)

                if protocol == "tls":
                    ctx = ssl.create_default_context()
                    # Allow self-signed certs in defense environments where
                    # internal CAs are common; production deployments should
                    # configure proper CA bundles via ssl context.
                    ctx.check_hostname = False
                    ctx.verify_mode = ssl.CERT_NONE
                    sock = ctx.wrap_socket(raw_sock, server_hostname=host)
                else:
                    sock = raw_sock

                sock.connect((host, port))
                dest["connected"] = True

                for event_str in event_list:
                    payload = event_str + "\n"
                    sock.sendall(payload.encode("utf-8"))
                    total_forwarded += 1
                    dest["events_sent"] = dest.get("events_sent", 0) + 1

                dest["last_error"] = None

            except (socket.error, ssl.SSLError, OSError) as exc:
                dest["connected"] = False
                dest["last_error"] = str(exc)
                logger.warning(
                    "Failed to forward events to '%s' (%s:%d): %s",
                    dest["name"], host, port, exc,
                )
            finally:
                if sock is not None:
                    try:
                        sock.close()
                    except OSError:
                        pass

        return total_forwarded

    def generate_events(
        self,
        assessment_result: Dict[str, Any],
        output_formats: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Generate SIEM events from an insider threat or fraud assessment.

        Args:
            assessment_result: Result dict from InsiderThreatAssessor.assess_user()
                              or similar assessment output
            output_formats: List of formats to generate: "cef", "leef", "syslog"
                           Defaults to all three.

        Returns:
            Generated events in requested formats with enrichment data
        """
        if output_formats is None:
            output_formats = ["cef", "leef", "syslog"]

        timestamp = datetime.now(timezone.utc).isoformat()
        user_id = assessment_result.get("user_id", "unknown")
        risk_score = assessment_result.get("risk_score", 0)
        threat_level = assessment_result.get("threat_level", "LOW")
        triggered = assessment_result.get("triggered_indicators", [])

        severity = EventSeverity.from_risk_score(risk_score)

        # Determine DoD incident category
        dod_category = self._classify_dod_category(triggered)

        # Collect all MITRE techniques
        all_techniques: List[str] = []
        for t in triggered:
            all_techniques.extend(t.get("mitre_techniques", []))
        unique_techniques = list(set(all_techniques))

        # Enrich with MITRE ATT&CK data
        mitre_enrichment = self._enrich_mitre(unique_techniques)

        # Build base event data
        base_extensions = {
            "src": user_id,
            "duser": user_id,
            "rt": timestamp,
            "cs1": threat_level,
            "cs1Label": "ThreatLevel",
            "cs2": str(risk_score),
            "cs2Label": "RiskScore",
            "cs3": dod_category.get("category_id", "UNCAT"),
            "cs3Label": "DoDCategory",
            "cs4": ",".join(unique_techniques[:10]),
            "cs4Label": "MITRETechniques",
            "cn1": len(triggered),
            "cn1Label": "IndicatorsTriggered",
            "cat": "InsiderThreat",
            "msg": assessment_result.get("threat_level_description", ""),
        }

        events: Dict[str, List[str]] = {"cef": [], "leef": [], "syslog": []}

        # Generate summary event
        summary_event_id = f"IT-{uuid.uuid4().hex[:12].upper()}"
        event_name = f"Insider Threat Assessment: {threat_level}"

        if "cef" in output_formats:
            cef = EventFormatter.to_cef(
                summary_event_id, event_name, severity, base_extensions
            )
            events["cef"].append(cef)
            with self._lock:
                self._stats["cef_events"] += 1

        if "leef" in output_formats:
            leef_attrs = {
                "usrName": user_id,
                "identSrc": user_id,
                "devTime": timestamp,
                "threatLevel": threat_level,
                "riskScore": str(risk_score),
                "dodCategory": dod_category.get("category_id", "UNCAT"),
                "mitreTechniques": ",".join(unique_techniques[:10]),
                "indicatorCount": str(len(triggered)),
            }
            leef = EventFormatter.to_leef(
                summary_event_id, event_name, severity, leef_attrs
            )
            events["leef"].append(leef)
            with self._lock:
                self._stats["leef_events"] += 1

        if "syslog" in output_formats:
            sd = {
                "userId": user_id,
                "threatLevel": threat_level,
                "riskScore": str(risk_score),
                "dodCategory": dod_category.get("category_id", "UNCAT"),
                "indicatorCount": str(len(triggered)),
                "mitre": ",".join(unique_techniques[:5]),
            }
            syslog = EventFormatter.to_syslog_rfc5424(
                summary_event_id, event_name, severity, sd
            )
            events["syslog"].append(syslog)
            with self._lock:
                self._stats["syslog_events"] += 1

        # Generate per-indicator events for HIGH/CRITICAL
        if severity in (EventSeverity.HIGH, EventSeverity.CRITICAL):
            for indicator in triggered:
                ind_event_id = f"IT-IND-{uuid.uuid4().hex[:8].upper()}"
                ind_name = f"Indicator: {indicator['indicator_name']}"
                ind_severity = EventSeverity.from_risk_score(
                    indicator["weight"] * indicator["confidence"] * 10
                )

                ind_extensions = {
                    "src": user_id,
                    "duser": user_id,
                    "rt": indicator.get("timestamp", timestamp),
                    "cs1": indicator["indicator_id"],
                    "cs1Label": "IndicatorID",
                    "cs2": indicator["category"],
                    "cs2Label": "IndicatorCategory",
                    "cs3": str(indicator["weight"]),
                    "cs3Label": "IndicatorWeight",
                    "cs4": str(indicator["confidence"]),
                    "cs4Label": "Confidence",
                    "msg": indicator.get("details", ""),
                    "cat": "InsiderThreatIndicator",
                }

                if "cef" in output_formats:
                    events["cef"].append(
                        EventFormatter.to_cef(
                            ind_event_id, ind_name, ind_severity, ind_extensions
                        )
                    )
                if "leef" in output_formats:
                    events["leef"].append(
                        EventFormatter.to_leef(
                            ind_event_id, ind_name, ind_severity, {
                                "indicatorId": indicator["indicator_id"],
                                "indicatorName": indicator["indicator_name"],
                                "category": indicator["category"],
                                "weight": str(indicator["weight"]),
                                "confidence": str(indicator["confidence"]),
                                "details": indicator.get("details", ""),
                            }
                        )
                    )
                if "syslog" in output_formats:
                    events["syslog"].append(
                        EventFormatter.to_syslog_rfc5424(
                            ind_event_id, ind_name, ind_severity, {
                                "indicatorId": indicator["indicator_id"],
                                "category": indicator["category"],
                                "weight": str(indicator["weight"]),
                                "confidence": str(indicator["confidence"]),
                            }
                        )
                    )

        # Store in buffer and check correlation rules
        event_record = {
            "event_id": summary_event_id,
            "user_id": user_id,
            "risk_score": risk_score,
            "threat_level": threat_level,
            "severity": severity.name,
            "dod_category": dod_category,
            "indicator_ids": [t["indicator_id"] for t in triggered],
            "mitre_techniques": unique_techniques,
            "timestamp": timestamp,
        }

        with self._lock:
            self._event_buffer.append(event_record)
            self._stats["events_generated"] += 1

        # Feed correlation engine
        correlation_alerts = self._check_correlations(user_id, triggered, timestamp)

        # Forward events to registered SIEM destinations
        forwarded_count = self._forward_events(events)
        with self._lock:
            self._stats["events_forwarded"] += forwarded_count

        result = {
            "event_id": summary_event_id,
            "timestamp": timestamp,
            "user_id": user_id,
            "severity": severity.name,
            "dod_incident_category": dod_category,
            "mitre_enrichment": mitre_enrichment,
            "events": {
                fmt: evts for fmt, evts in events.items() if evts
            },
            "event_counts": {
                fmt: len(evts) for fmt, evts in events.items() if evts
            },
            "correlation_alerts": correlation_alerts,
            "total_events_generated": sum(len(evts) for evts in events.values()),
            "events_forwarded": forwarded_count,
        }

        return result

    def _classify_dod_category(
        self, triggered_indicators: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Classify the incident per DoD 8570/8140 categories."""
        if not triggered_indicators:
            return {
                "category_id": "UNCAT",
                "name": "Uncategorized",
                "priority": "low",
            }

        # Check categories by priority (CAT-1 highest)
        indicator_ids = set(t["indicator_id"] for t in triggered_indicators)
        categories_set = set(t.get("category", "") for t in triggered_indicators)

        # CAT-1: Root level intrusion indicators
        cat1_indicators = {"IND-016", "IND-006", "IND-025"}
        if indicator_ids & cat1_indicators:
            cat = DOD_INCIDENT_CATEGORIES["CAT-1"]
            return {"category_id": "CAT-1", **cat}

        # CAT-2: User level intrusion
        cat2_indicators = {"IND-001", "IND-018", "IND-026", "IND-027"}
        if indicator_ids & cat2_indicators:
            cat = DOD_INCIDENT_CATEGORIES["CAT-2"]
            return {"category_id": "CAT-2", **cat}

        # CAT-7: Malicious logic
        if "IND-022" in indicator_ids:
            cat = DOD_INCIDENT_CATEGORIES["CAT-7"]
            return {"category_id": "CAT-7", **cat}

        # CAT-6: Reconnaissance
        if "IND-023" in indicator_ids:
            cat = DOD_INCIDENT_CATEGORIES["CAT-6"]
            return {"category_id": "CAT-6", **cat}

        # CAT-3: Unsuccessful attempts
        if "IND-017" in indicator_ids:
            cat = DOD_INCIDENT_CATEGORIES["CAT-3"]
            return {"category_id": "CAT-3", **cat}

        # CAT-5: Non-compliance (default for policy violations)
        cat = DOD_INCIDENT_CATEGORIES["CAT-5"]
        return {"category_id": "CAT-5", **cat}

    def _enrich_mitre(self, technique_ids: List[str]) -> List[Dict[str, Any]]:
        """Enrich event with MITRE ATT&CK technique details."""
        enriched = []
        for tech_id in technique_ids:
            if tech_id in MITRE_ATTACK_MAP:
                info = MITRE_ATTACK_MAP[tech_id]
                enriched.append({
                    "technique_id": tech_id,
                    "technique_name": info["name"],
                    "tactic": info["tactic"],
                    "reference_url": info["url"],
                })
            else:
                enriched.append({
                    "technique_id": tech_id,
                    "technique_name": "Unknown",
                    "tactic": "Unknown",
                    "reference_url": f"https://attack.mitre.org/techniques/{tech_id}/",
                })
        return enriched

    def _check_correlations(
        self,
        user_id: str,
        triggered: List[Dict[str, Any]],
        timestamp: str,
    ) -> List[Dict[str, Any]]:
        """Check triggered indicators against correlation rules."""
        alerts: List[Dict[str, Any]] = []

        indicator_ids = [t["indicator_id"] for t in triggered]

        with self._lock:
            # Add to per-user correlation buffer
            for ind_id in indicator_ids:
                self._correlation_buffer[user_id].append({
                    "indicator_id": ind_id,
                    "timestamp": timestamp,
                })

            # Check each correlation rule
            for rule in CORRELATION_RULES:
                matched = self._evaluate_correlation_rule(
                    rule, user_id, timestamp
                )
                if matched:
                    alert = {
                        "alert_id": f"CORR-{uuid.uuid4().hex[:8].upper()}",
                        "rule_id": rule.rule_id,
                        "rule_name": rule.name,
                        "description": rule.description,
                        "user_id": user_id,
                        "severity": rule.severity.name,
                        "dod_category": rule.dod_category,
                        "matched_indicators": matched,
                        "mitre_techniques": rule.mitre_techniques,
                        "time_window_minutes": rule.time_window_minutes,
                        "timestamp": timestamp,
                    }
                    alerts.append(alert)
                    self._alert_history.append(alert)
                    self._stats["correlation_alerts"] += 1
                    logger.warning(
                        "Correlation alert %s fired for user %s: %s",
                        alert["alert_id"], user_id, rule.name
                    )

        return alerts

    def _evaluate_correlation_rule(
        self,
        rule: CorrelationRule,
        user_id: str,
        current_timestamp: str,
    ) -> Optional[List[str]]:
        """Evaluate a single correlation rule against the user's event buffer."""
        buffer = self._correlation_buffer.get(user_id, deque())
        if not buffer:
            return None

        try:
            current_time = datetime.fromisoformat(
                current_timestamp.replace("Z", "+00:00")
            )
        except ValueError:
            current_time = datetime.now(timezone.utc)

        window_start = current_time - timedelta(minutes=rule.time_window_minutes)

        # Find all indicators in the time window that match the rule's requirements
        matched: List[str] = []
        for entry in buffer:
            try:
                entry_time = datetime.fromisoformat(
                    entry["timestamp"].replace("Z", "+00:00")
                )
            except ValueError:
                continue

            if entry_time >= window_start:
                if entry["indicator_id"] in rule.required_indicators:
                    if entry["indicator_id"] not in matched:
                        matched.append(entry["indicator_id"])

        if len(matched) >= rule.min_match_count:
            return matched
        return None

    def batch_export(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        user_id: Optional[str] = None,
        min_severity: Optional[str] = None,
        output_format: str = "json",
    ) -> Dict[str, Any]:
        """
        Export events for offline analysis.

        Args:
            start_time: ISO format start time filter
            end_time: ISO format end time filter
            user_id: Filter by specific user
            min_severity: Minimum severity filter
            output_format: "json" or "csv"

        Returns:
            Exported events with metadata
        """
        with self._lock:
            events = list(self._event_buffer)

        # Apply filters
        filtered = events
        if user_id:
            filtered = [e for e in filtered if e.get("user_id") == user_id]
        if start_time:
            filtered = [e for e in filtered if e.get("timestamp", "") >= start_time]
        if end_time:
            filtered = [e for e in filtered if e.get("timestamp", "") <= end_time]
        if min_severity:
            severity_order = ["INFORMATIONAL", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
            try:
                min_idx = severity_order.index(min_severity.upper())
                filtered = [
                    e for e in filtered
                    if severity_order.index(e.get("severity", "INFORMATIONAL")) >= min_idx
                ]
            except ValueError:
                pass

        if output_format == "csv":
            # Generate CSV representation
            if not filtered:
                csv_content = "event_id,user_id,risk_score,threat_level,severity,timestamp\n"
            else:
                lines = ["event_id,user_id,risk_score,threat_level,severity,dod_category,timestamp"]
                for e in filtered:
                    lines.append(
                        f"{e.get('event_id', '')},{e.get('user_id', '')},"
                        f"{e.get('risk_score', 0)},{e.get('threat_level', '')},"
                        f"{e.get('severity', '')},{e.get('dod_category', {}).get('category_id', '')},"
                        f"{e.get('timestamp', '')}"
                    )
                csv_content = "\n".join(lines)
            return {
                "format": "csv",
                "event_count": len(filtered),
                "content": csv_content,
            }
        else:
            return {
                "format": "json",
                "event_count": len(filtered),
                "events": filtered,
                "export_metadata": {
                    "exported_at": datetime.now(timezone.utc).isoformat(),
                    "filters_applied": {
                        "start_time": start_time,
                        "end_time": end_time,
                        "user_id": user_id,
                        "min_severity": min_severity,
                    },
                    "total_events_in_buffer": len(events),
                },
            }

    def get_stats(self) -> Dict[str, Any]:
        """Get SIEM integration statistics."""
        with self._lock:
            stats = dict(self._stats)
            stats["event_buffer_size"] = len(self._event_buffer)
            stats["correlation_rules_count"] = len(CORRELATION_RULES)
            stats["alert_history_size"] = len(self._alert_history)
            stats["forwarding_destinations"] = len(self._forwarding_destinations)
            stats["users_tracked"] = len(self._correlation_buffer)
        return stats

    def list_correlation_rules(self) -> List[Dict[str, Any]]:
        """List all active correlation rules."""
        return [
            {
                "rule_id": r.rule_id,
                "name": r.name,
                "description": r.description,
                "required_indicators": r.required_indicators,
                "min_match_count": r.min_match_count,
                "time_window_minutes": r.time_window_minutes,
                "severity": r.severity.name,
                "dod_category": r.dod_category,
                "mitre_techniques": r.mitre_techniques,
            }
            for r in CORRELATION_RULES
        ]
