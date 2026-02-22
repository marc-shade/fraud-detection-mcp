"""
Compliance Dashboard Metrics Module

Defense compliance metrics generation for executive reporting.
Implements NITTF maturity model scoring, Key Risk Indicators (KRIs),
MTTD/MTTR tracking, and compliance posture assessment against
NIST 800-53 PS/PE/AC control families.

References:
- NITTF Insider Threat Program Maturity Framework
- NIST SP 800-53 Rev 5: PS, PE, AC Control Families
- CISA Cybersecurity Performance Goals
"""

import csv
import io
import json
import logging
import math
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# NITTF Maturity Model
# =============================================================================

class MaturityLevel(Enum):
    """NITTF Insider Threat Program Maturity Levels."""
    INITIAL = 1
    MANAGED = 2
    DEFINED = 3
    QUANTITATIVELY_MANAGED = 4
    OPTIMIZING = 5


MATURITY_CRITERIA = {
    MaturityLevel.INITIAL: {
        "name": "Initial",
        "description": "Ad hoc insider threat processes; reactive posture",
        "criteria": [
            "insider_threat_program_exists",
            "designated_senior_official",
            "basic_access_controls",
        ],
    },
    MaturityLevel.MANAGED: {
        "name": "Managed",
        "description": "Basic monitoring in place; some standardized processes",
        "criteria": [
            "user_activity_monitoring_deployed",
            "incident_reporting_process",
            "security_awareness_training",
            "access_review_periodic",
            "audit_logging_enabled",
        ],
    },
    MaturityLevel.DEFINED: {
        "name": "Defined",
        "description": "Documented policies and procedures; integrated monitoring",
        "criteria": [
            "insider_threat_policy_documented",
            "uam_integrated_with_siem",
            "behavioral_indicator_library",
            "cross_functional_team",
            "continuous_evaluation_program",
            "risk_scoring_implemented",
            "referral_process_established",
        ],
    },
    MaturityLevel.QUANTITATIVELY_MANAGED: {
        "name": "Quantitatively Managed",
        "description": "Metrics-driven program; data-informed decisions",
        "criteria": [
            "kri_tracking_active",
            "mttd_mttr_measured",
            "false_positive_rate_tracked",
            "model_performance_monitored",
            "compliance_posture_scored",
            "trend_analysis_conducted",
            "executive_reporting_regular",
            "correlation_rules_engine",
            "mitre_attack_mapping",
        ],
    },
    MaturityLevel.OPTIMIZING: {
        "name": "Optimizing",
        "description": "Continuous improvement; adaptive threat detection",
        "criteria": [
            "automated_indicator_tuning",
            "ml_model_drift_detection",
            "feedback_loop_from_investigations",
            "peer_benchmarking",
            "advanced_analytics_ai_ml",
            "red_team_exercises",
            "supply_chain_insider_coverage",
            "cross_agency_information_sharing",
            "zero_trust_integration",
            "predictive_analytics",
        ],
    },
}


# =============================================================================
# NIST 800-53 Control Families for Compliance Posture
# =============================================================================

COMPLIANCE_CONTROLS = {
    "PS": {
        "family_name": "Personnel Security",
        "controls": {
            "PS-1": {"name": "Policy and Procedures", "weight": 1.0},
            "PS-2": {"name": "Position Risk Designation", "weight": 1.0},
            "PS-3": {"name": "Personnel Screening", "weight": 2.0},
            "PS-4": {"name": "Personnel Termination", "weight": 2.0},
            "PS-5": {"name": "Personnel Transfer", "weight": 1.5},
            "PS-6": {"name": "Access Agreements", "weight": 1.5},
            "PS-7": {"name": "External Personnel Security", "weight": 1.0},
            "PS-8": {"name": "Personnel Sanctions", "weight": 1.0},
            "PS-9": {"name": "Position Descriptions", "weight": 0.5},
        },
    },
    "PE": {
        "family_name": "Physical and Environmental Protection",
        "controls": {
            "PE-1": {"name": "Policy and Procedures", "weight": 1.0},
            "PE-2": {"name": "Physical Access Authorizations", "weight": 2.0},
            "PE-3": {"name": "Physical Access Control", "weight": 2.0},
            "PE-4": {"name": "Access Control for Transmission", "weight": 1.0},
            "PE-5": {"name": "Access Control for Output Devices", "weight": 1.0},
            "PE-6": {"name": "Monitoring Physical Access", "weight": 1.5},
            "PE-8": {"name": "Visitor Access Records", "weight": 1.0},
        },
    },
    "AC": {
        "family_name": "Access Control",
        "controls": {
            "AC-1": {"name": "Policy and Procedures", "weight": 1.0},
            "AC-2": {"name": "Account Management", "weight": 2.0},
            "AC-3": {"name": "Access Enforcement", "weight": 2.0},
            "AC-5": {"name": "Separation of Duties", "weight": 1.5},
            "AC-6": {"name": "Least Privilege", "weight": 2.0},
            "AC-7": {"name": "Unsuccessful Logon Attempts", "weight": 1.0},
            "AC-11": {"name": "Device Lock", "weight": 0.5},
            "AC-12": {"name": "Session Termination", "weight": 1.0},
            "AC-17": {"name": "Remote Access", "weight": 1.5},
            "AC-19": {"name": "Access Control for Mobile Devices", "weight": 1.0},
            "AC-20": {"name": "Use of External Systems", "weight": 1.0},
            "AC-25": {"name": "Reference Monitor", "weight": 1.5},
        },
    },
}


# =============================================================================
# Compliance Dashboard Engine
# =============================================================================

class ComplianceDashboard:
    """
    Defense compliance metrics engine.

    Generates maturity scores, KRIs, MTTD/MTTR metrics,
    compliance posture assessments, and executive reports.
    Thread-safe for concurrent metric generation.
    """

    def __init__(self):
        self._lock = threading.Lock()

        # Detection and response time tracking
        self._detection_times: deque = deque(maxlen=10000)
        self._response_times: deque = deque(maxlen=10000)

        # False positive tracking
        self._alert_outcomes: deque = deque(maxlen=10000)

        # Model performance tracking
        self._model_metrics: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )

        # KRI history for trend analysis
        self._kri_history: deque = deque(maxlen=5000)

        # Compliance control assessments
        self._control_assessments: Dict[str, Dict[str, Any]] = {}

        # Maturity criteria status
        self._maturity_criteria_status: Dict[str, bool] = {}

        # Executive report history
        self._report_history: deque = deque(maxlen=100)

        logger.info("ComplianceDashboard initialized")

    # =========================================================================
    # Data Recording Methods
    # =========================================================================

    def record_detection_time(
        self,
        event_id: str,
        detection_seconds: float,
        event_type: str = "insider_threat",
    ) -> None:
        """Record time to detect an event (for MTTD calculation)."""
        with self._lock:
            self._detection_times.append({
                "event_id": event_id,
                "detection_seconds": detection_seconds,
                "event_type": event_type,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

    def record_response_time(
        self,
        event_id: str,
        response_seconds: float,
        event_type: str = "insider_threat",
    ) -> None:
        """Record time to respond to an event (for MTTR calculation)."""
        with self._lock:
            self._response_times.append({
                "event_id": event_id,
                "response_seconds": response_seconds,
                "event_type": event_type,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

    def record_alert_outcome(
        self,
        alert_id: str,
        outcome: str,
        alert_type: str = "insider_threat",
    ) -> None:
        """
        Record the outcome of an alert for false positive tracking.

        Args:
            alert_id: Alert identifier
            outcome: One of "true_positive", "false_positive", "indeterminate"
            alert_type: Type of alert
        """
        with self._lock:
            self._alert_outcomes.append({
                "alert_id": alert_id,
                "outcome": outcome,
                "alert_type": alert_type,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

    def record_model_metric(
        self,
        model_name: str,
        metric_name: str,
        value: float,
    ) -> None:
        """Record a model performance metric for drift detection."""
        with self._lock:
            self._model_metrics[f"{model_name}.{metric_name}"].append({
                "value": value,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

    def set_maturity_criterion(self, criterion: str, met: bool) -> None:
        """Set whether a specific maturity criterion is met."""
        with self._lock:
            self._maturity_criteria_status[criterion] = met

    def set_control_assessment(
        self,
        control_id: str,
        status: str,
        evidence: Optional[str] = None,
        assessor: Optional[str] = None,
    ) -> None:
        """
        Set the assessment status of a NIST 800-53 control.

        Args:
            control_id: Control identifier (e.g., "PS-3", "AC-2")
            status: One of "implemented", "partially_implemented",
                   "planned", "not_implemented", "not_applicable"
            evidence: Evidence supporting the assessment
            assessor: Name of the assessor
        """
        with self._lock:
            self._control_assessments[control_id] = {
                "control_id": control_id,
                "status": status,
                "evidence": evidence or "",
                "assessor": assessor or "system",
                "assessed_at": datetime.now(timezone.utc).isoformat(),
            }

    # =========================================================================
    # Metric Calculation Methods
    # =========================================================================

    def calculate_maturity_score(self) -> Dict[str, Any]:
        """
        Calculate the insider threat program maturity level per NITTF framework.

        Returns:
            Maturity assessment with level, score, and per-level analysis.
            If no maturity criteria have been seeded, returns assessment_status
            "NOT_ASSESSED" with data_seeded=False to distinguish from an
            actual 0% / INITIAL score.
        """
        with self._lock:
            criteria_status = dict(self._maturity_criteria_status)

        data_seeded = len(criteria_status) > 0

        levels_assessment: Dict[str, Dict[str, Any]] = {}
        achieved_level = MaturityLevel.INITIAL
        total_criteria = 0
        met_criteria = 0

        for level in MaturityLevel:
            level_info = MATURITY_CRITERIA[level]
            criteria = level_info["criteria"]
            total_criteria += len(criteria)

            met_for_level = sum(
                1 for c in criteria if criteria_status.get(c, False)
            )
            met_criteria += met_for_level

            completion = met_for_level / len(criteria) if criteria else 0
            level_met = completion >= 0.80  # 80% threshold for level achievement

            levels_assessment[level.name] = {
                "level_number": level.value,
                "name": level_info["name"],
                "description": level_info["description"],
                "criteria_count": len(criteria),
                "criteria_met": met_for_level,
                "completion_percentage": round(completion * 100, 1),
                "level_achieved": level_met,
                "criteria_detail": {
                    c: criteria_status.get(c, False) for c in criteria
                },
            }

            if level_met:
                achieved_level = level

        overall_score = (met_criteria / total_criteria * 100) if total_criteria > 0 else 0

        return {
            "maturity_level": achieved_level.name,
            "maturity_level_number": achieved_level.value,
            "maturity_level_name": MATURITY_CRITERIA[achieved_level]["name"],
            "overall_score": round(overall_score, 1),
            "total_criteria": total_criteria,
            "criteria_met": met_criteria,
            "framework": "NITTF Insider Threat Program Maturity",
            "levels": levels_assessment,
            "data_seeded": data_seeded,
            "assessment_status": "ASSESSED" if data_seeded else "NOT_ASSESSED",
        }

    def calculate_kris(self) -> Dict[str, Any]:
        """
        Calculate Key Risk Indicators with trend analysis.

        Returns:
            KRI values with current state, trend, and historical data
        """
        with self._lock:
            detection_times = list(self._detection_times)
            response_times = list(self._response_times)
            alert_outcomes = list(self._alert_outcomes)

        now = datetime.now(timezone.utc)
        period_30d = now - timedelta(days=30)
        period_7d = now - timedelta(days=7)

        kris: Dict[str, Any] = {}

        # KRI 1: MTTD (Mean Time to Detect)
        mttd_30d = self._calc_mean_metric(detection_times, "detection_seconds", period_30d)
        mttd_7d = self._calc_mean_metric(detection_times, "detection_seconds", period_7d)
        mttd_trend = self._determine_trend(mttd_30d, mttd_7d)
        kris["mttd"] = {
            "name": "Mean Time to Detect",
            "unit": "seconds",
            "current_30d": round(mttd_30d, 1),
            "current_7d": round(mttd_7d, 1),
            "trend": mttd_trend,
            "target": 3600,  # 1 hour target
            "status": "ON_TARGET" if mttd_7d <= 3600 else "ABOVE_TARGET",
        }

        # KRI 2: MTTR (Mean Time to Respond)
        mttr_30d = self._calc_mean_metric(response_times, "response_seconds", period_30d)
        mttr_7d = self._calc_mean_metric(response_times, "response_seconds", period_7d)
        mttr_trend = self._determine_trend(mttr_30d, mttr_7d)
        kris["mttr"] = {
            "name": "Mean Time to Respond",
            "unit": "seconds",
            "current_30d": round(mttr_30d, 1),
            "current_7d": round(mttr_7d, 1),
            "trend": mttr_trend,
            "target": 14400,  # 4 hour target
            "status": "ON_TARGET" if mttr_7d <= 14400 else "ABOVE_TARGET",
        }

        # KRI 3: False Positive Rate
        fp_30d = self._calc_false_positive_rate(alert_outcomes, period_30d)
        fp_7d = self._calc_false_positive_rate(alert_outcomes, period_7d)
        fp_trend = self._determine_trend(fp_30d, fp_7d)
        kris["false_positive_rate"] = {
            "name": "False Positive Rate",
            "unit": "percentage",
            "current_30d": round(fp_30d, 2),
            "current_7d": round(fp_7d, 2),
            "trend": fp_trend,
            "target": 5.0,  # 5% target
            "status": "ON_TARGET" if fp_7d <= 5.0 else "ABOVE_TARGET",
        }

        # KRI 4: Alert Volume
        alerts_30d = len([
            a for a in alert_outcomes
            if self._in_period(a.get("timestamp", ""), period_30d)
        ])
        alerts_7d = len([
            a for a in alert_outcomes
            if self._in_period(a.get("timestamp", ""), period_7d)
        ])
        # Normalize to daily rate
        daily_30d = alerts_30d / 30 if alerts_30d > 0 else 0
        daily_7d = alerts_7d / 7 if alerts_7d > 0 else 0
        kris["alert_volume"] = {
            "name": "Daily Alert Volume",
            "unit": "alerts/day",
            "current_30d": round(daily_30d, 1),
            "current_7d": round(daily_7d, 1),
            "trend": self._determine_trend(daily_30d, daily_7d),
            "total_30d": alerts_30d,
            "total_7d": alerts_7d,
        }

        # KRI 5: True Positive Rate
        tp_30d = self._calc_true_positive_rate(alert_outcomes, period_30d)
        tp_7d = self._calc_true_positive_rate(alert_outcomes, period_7d)
        kris["true_positive_rate"] = {
            "name": "True Positive Rate",
            "unit": "percentage",
            "current_30d": round(tp_30d, 2),
            "current_7d": round(tp_7d, 2),
            "trend": self._determine_trend(tp_30d, tp_7d, higher_is_better=True),
            "target": 85.0,
            "status": "ON_TARGET" if tp_7d >= 85.0 else "BELOW_TARGET",
        }

        # Record KRI snapshot for history
        kri_snapshot = {
            "timestamp": now.isoformat(),
            "mttd": mttd_7d,
            "mttr": mttr_7d,
            "false_positive_rate": fp_7d,
            "true_positive_rate": tp_7d,
            "daily_alert_volume": daily_7d,
        }
        with self._lock:
            self._kri_history.append(kri_snapshot)

        return {
            "calculated_at": now.isoformat(),
            "kris": kris,
            "overall_health": self._assess_kri_health(kris),
        }

    def calculate_compliance_posture(self) -> Dict[str, Any]:
        """
        Score compliance posture against NIST 800-53 PS/PE/AC families.

        Returns:
            Compliance posture with per-family and per-control scores.
            If no controls have been assessed, returns assessment_status
            "NOT_ASSESSED" with data_seeded=False to distinguish from an
            actual 0% / NON_COMPLIANT score.
        """
        with self._lock:
            assessments = dict(self._control_assessments)

        data_seeded = len(assessments) > 0

        status_scores = {
            "implemented": 1.0,
            "partially_implemented": 0.5,
            "planned": 0.2,
            "not_implemented": 0.0,
            "not_applicable": None,  # Excluded from scoring
        }

        family_results: Dict[str, Dict[str, Any]] = {}
        total_weighted_score = 0
        total_weight = 0

        for family_id, family_info in COMPLIANCE_CONTROLS.items():
            controls = family_info["controls"]
            family_score = 0
            family_weight = 0
            control_details: Dict[str, Dict[str, Any]] = {}

            for ctrl_id, ctrl_info in controls.items():
                assessment = assessments.get(ctrl_id, {})
                status = assessment.get("status", "not_implemented")
                score_val = status_scores.get(status)

                if score_val is not None:
                    weight = ctrl_info["weight"]
                    weighted = score_val * weight
                    family_score += weighted
                    family_weight += weight
                    total_weighted_score += weighted
                    total_weight += weight

                control_details[ctrl_id] = {
                    "name": ctrl_info["name"],
                    "weight": ctrl_info["weight"],
                    "status": status,
                    "score": score_val if score_val is not None else "N/A",
                    "evidence": assessment.get("evidence", ""),
                    "assessor": assessment.get("assessor", ""),
                    "assessed_at": assessment.get("assessed_at", ""),
                }

            family_pct = (family_score / family_weight * 100) if family_weight > 0 else 0

            family_results[family_id] = {
                "family_name": family_info["family_name"],
                "score_percentage": round(family_pct, 1),
                "controls_assessed": sum(
                    1 for c in control_details.values() if c["status"] != "not_implemented"
                ),
                "controls_total": len(controls),
                "controls": control_details,
                "compliance_level": self._score_to_compliance_level(family_pct),
            }

        overall_pct = (total_weighted_score / total_weight * 100) if total_weight > 0 else 0

        return {
            "overall_compliance_score": round(overall_pct, 1),
            "overall_compliance_level": (
                self._score_to_compliance_level(overall_pct)
                if data_seeded else "NOT_ASSESSED"
            ),
            "families": family_results,
            "assessed_at": datetime.now(timezone.utc).isoformat(),
            "framework": "NIST SP 800-53 Rev 5",
            "families_evaluated": list(COMPLIANCE_CONTROLS.keys()),
            "data_seeded": data_seeded,
            "assessment_status": "ASSESSED" if data_seeded else "NOT_ASSESSED",
        }

    def detect_model_drift(self) -> Dict[str, Any]:
        """
        Detect model performance drift by comparing recent metrics to historical baselines.

        Returns:
            Drift detection results per model/metric combination
        """
        with self._lock:
            all_metrics = {k: list(v) for k, v in self._model_metrics.items()}

        drift_results: Dict[str, Dict[str, Any]] = {}

        for metric_key, values in all_metrics.items():
            if len(values) < 10:
                drift_results[metric_key] = {
                    "status": "INSUFFICIENT_DATA",
                    "data_points": len(values),
                    "minimum_required": 10,
                }
                continue

            all_vals = [v["value"] for v in values]
            n = len(all_vals)

            # Split into baseline (first 70%) and recent (last 30%)
            split = max(1, int(n * 0.7))
            baseline = all_vals[:split]
            recent = all_vals[split:]

            if not recent:
                drift_results[metric_key] = {
                    "status": "INSUFFICIENT_RECENT_DATA",
                    "data_points": n,
                }
                continue

            baseline_mean = sum(baseline) / len(baseline)
            recent_mean = sum(recent) / len(recent)
            baseline_std = math.sqrt(
                sum((x - baseline_mean) ** 2 for x in baseline) / len(baseline)
            ) if len(baseline) > 1 else 0.001

            # Z-score test for drift
            z_score = (recent_mean - baseline_mean) / max(baseline_std, 0.001)
            drift_detected = abs(z_score) > 2.0

            # Calculate magnitude
            pct_change = (
                (recent_mean - baseline_mean) / max(abs(baseline_mean), 0.001) * 100
            )

            drift_results[metric_key] = {
                "status": "DRIFT_DETECTED" if drift_detected else "STABLE",
                "baseline_mean": round(baseline_mean, 4),
                "recent_mean": round(recent_mean, 4),
                "z_score": round(z_score, 3),
                "percentage_change": round(pct_change, 2),
                "direction": "increasing" if z_score > 0 else "decreasing",
                "baseline_points": len(baseline),
                "recent_points": len(recent),
                "threshold": 2.0,
            }

        drifting_count = sum(
            1 for r in drift_results.values() if r.get("status") == "DRIFT_DETECTED"
        )

        return {
            "models_monitored": len(drift_results),
            "drift_detected_count": drifting_count,
            "overall_status": "DRIFT_DETECTED" if drifting_count > 0 else "STABLE",
            "metrics": drift_results,
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }

    # =========================================================================
    # Report Generation
    # =========================================================================

    def generate_executive_summary(
        self,
        insider_threat_stats: Optional[Dict[str, Any]] = None,
        siem_stats: Optional[Dict[str, Any]] = None,
        personnel_stats: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate an executive summary report suitable for CSO/CISO briefings.

        Args:
            insider_threat_stats: Stats from InsiderThreatAssessor
            siem_stats: Stats from SIEMIntegration
            personnel_stats: Stats from ClearedPersonnelAnalyzer

        Returns:
            Executive summary with all compliance metrics
        """
        report_id = (
            f"EXEC-{datetime.now(timezone.utc).strftime('%Y%m%d')}-"
            f"{hash(time.time()) % 10000:04d}"
        )
        timestamp = datetime.now(timezone.utc).isoformat()

        maturity = self.calculate_maturity_score()
        kris = self.calculate_kris()
        compliance = self.calculate_compliance_posture()
        drift = self.detect_model_drift()

        report = {
            "report_id": report_id,
            "report_type": "EXECUTIVE_SUMMARY",
            "classification": "FOR OFFICIAL USE ONLY",
            "generated_at": timestamp,
            "reporting_period": {
                "start": (
                    datetime.now(timezone.utc) - timedelta(days=30)
                ).isoformat(),
                "end": timestamp,
            },
            "program_maturity": {
                "level": maturity["maturity_level"],
                "level_name": maturity["maturity_level_name"],
                "overall_score": maturity["overall_score"],
                "criteria_met": f"{maturity['criteria_met']}/{maturity['total_criteria']}",
            },
            "key_risk_indicators": {
                "overall_health": kris.get("overall_health", "UNKNOWN"),
                "mttd": kris["kris"].get("mttd", {}),
                "mttr": kris["kris"].get("mttr", {}),
                "false_positive_rate": kris["kris"].get("false_positive_rate", {}),
                "true_positive_rate": kris["kris"].get("true_positive_rate", {}),
                "alert_volume": kris["kris"].get("alert_volume", {}),
            },
            "compliance_posture": {
                "overall_score": compliance["overall_compliance_score"],
                "overall_level": compliance["overall_compliance_level"],
                "family_scores": {
                    fid: {
                        "name": fdata["family_name"],
                        "score": fdata["score_percentage"],
                        "level": fdata["compliance_level"],
                    }
                    for fid, fdata in compliance["families"].items()
                },
            },
            "model_health": {
                "overall_status": drift["overall_status"],
                "models_monitored": drift["models_monitored"],
                "drift_detected_count": drift["drift_detected_count"],
            },
            "component_stats": {
                "insider_threat": insider_threat_stats or {},
                "siem": siem_stats or {},
                "cleared_personnel": personnel_stats or {},
            },
            "recommendations": self._generate_recommendations(
                maturity, kris, compliance, drift
            ),
        }

        with self._lock:
            self._report_history.append({
                "report_id": report_id,
                "timestamp": timestamp,
                "maturity_level": maturity["maturity_level"],
                "compliance_score": compliance["overall_compliance_score"],
            })

        logger.info("Generated executive summary report %s", report_id)
        return report

    def export_metrics(
        self,
        output_format: str = "json",
        include_history: bool = False,
    ) -> Dict[str, Any]:
        """
        Export all metrics in JSON or CSV format.

        Args:
            output_format: "json" or "csv"
            include_history: Include historical KRI data

        Returns:
            Exported metrics data
        """
        maturity = self.calculate_maturity_score()
        kris = self.calculate_kris()
        compliance = self.calculate_compliance_posture()
        drift = self.detect_model_drift()

        data = {
            "maturity": maturity,
            "kris": kris,
            "compliance": compliance,
            "model_drift": drift,
            "exported_at": datetime.now(timezone.utc).isoformat(),
        }

        if include_history:
            with self._lock:
                data["kri_history"] = list(self._kri_history)

        if output_format == "csv":
            csv_output = io.StringIO()
            writer = csv.writer(csv_output)

            # Header
            writer.writerow([
                "metric_category", "metric_name", "value",
                "unit", "target", "status", "timestamp"
            ])

            # Maturity
            writer.writerow([
                "maturity", "level", maturity["maturity_level_name"],
                "level", "", "", data["exported_at"]
            ])
            writer.writerow([
                "maturity", "overall_score", maturity["overall_score"],
                "percentage", "", "", data["exported_at"]
            ])

            # KRIs
            for kri_key, kri_data in kris.get("kris", {}).items():
                writer.writerow([
                    "kri", kri_data.get("name", kri_key),
                    kri_data.get("current_7d", ""),
                    kri_data.get("unit", ""),
                    kri_data.get("target", ""),
                    kri_data.get("status", ""),
                    data["exported_at"]
                ])

            # Compliance
            for fid, fdata in compliance.get("families", {}).items():
                writer.writerow([
                    "compliance", fdata["family_name"],
                    fdata["score_percentage"],
                    "percentage", "100", fdata["compliance_level"],
                    data["exported_at"]
                ])

            # KRI History
            if include_history:
                for entry in data.get("kri_history", []):
                    writer.writerow([
                        "kri_history", "snapshot",
                        json.dumps(entry), "", "", "",
                        entry.get("timestamp", "")
                    ])

            return {
                "format": "csv",
                "content": csv_output.getvalue(),
                "rows": csv_output.getvalue().count("\n"),
            }
        else:
            return {
                "format": "json",
                "data": data,
            }

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _calc_mean_metric(
        self,
        entries: List[Dict[str, Any]],
        metric_key: str,
        since: datetime,
    ) -> float:
        """Calculate mean of a metric for entries since a given time."""
        values = []
        for entry in entries:
            if self._in_period(entry.get("timestamp", ""), since):
                val = entry.get(metric_key, 0)
                if isinstance(val, (int, float)):
                    values.append(val)
        return sum(values) / len(values) if values else 0

    def _calc_false_positive_rate(
        self,
        outcomes: List[Dict[str, Any]],
        since: datetime,
    ) -> float:
        """Calculate false positive rate for alerts since a given time."""
        total = 0
        false_positives = 0
        for outcome in outcomes:
            if self._in_period(outcome.get("timestamp", ""), since):
                if outcome.get("outcome") != "indeterminate":
                    total += 1
                    if outcome.get("outcome") == "false_positive":
                        false_positives += 1
        return (false_positives / total * 100) if total > 0 else 0

    def _calc_true_positive_rate(
        self,
        outcomes: List[Dict[str, Any]],
        since: datetime,
    ) -> float:
        """Calculate true positive rate for alerts since a given time."""
        total = 0
        true_positives = 0
        for outcome in outcomes:
            if self._in_period(outcome.get("timestamp", ""), since):
                if outcome.get("outcome") != "indeterminate":
                    total += 1
                    if outcome.get("outcome") == "true_positive":
                        true_positives += 1
        return (true_positives / total * 100) if total > 0 else 0

    def _in_period(self, timestamp_str: str, since: datetime) -> bool:
        """Check if a timestamp is within the given period."""
        try:
            ts = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            return ts >= since
        except (ValueError, AttributeError):
            return False

    def _determine_trend(
        self,
        value_30d: float,
        value_7d: float,
        higher_is_better: bool = False,
    ) -> str:
        """Determine trend direction and desirability."""
        if value_30d == 0 and value_7d == 0:
            return "STABLE"
        if value_30d == 0:
            return "INCREASING"

        pct_change = (value_7d - value_30d) / max(abs(value_30d), 0.001) * 100

        if abs(pct_change) < 5:
            return "STABLE"
        elif pct_change > 0:
            return "IMPROVING" if higher_is_better else "DEGRADING"
        else:
            return "DEGRADING" if higher_is_better else "IMPROVING"

    def _assess_kri_health(self, kris: Dict[str, Any]) -> str:
        """Assess overall KRI health status."""
        statuses = []
        for kri_data in kris.get("kris", {}).values():
            status = kri_data.get("status", "")
            trend = kri_data.get("trend", "")
            if status == "ABOVE_TARGET" or status == "BELOW_TARGET":
                if trend == "DEGRADING":
                    statuses.append("critical")
                else:
                    statuses.append("warning")
            elif trend == "DEGRADING":
                statuses.append("warning")
            else:
                statuses.append("healthy")

        if "critical" in statuses:
            return "CRITICAL"
        elif "warning" in statuses:
            return "WARNING"
        else:
            return "HEALTHY"

    def _score_to_compliance_level(self, score: float) -> str:
        """Convert compliance percentage to a level."""
        if score >= 90:
            return "COMPLIANT"
        elif score >= 70:
            return "SUBSTANTIALLY_COMPLIANT"
        elif score >= 50:
            return "PARTIALLY_COMPLIANT"
        elif score >= 25:
            return "MINIMALLY_COMPLIANT"
        else:
            return "NON_COMPLIANT"

    def _generate_recommendations(
        self,
        maturity: Dict[str, Any],
        kris: Dict[str, Any],
        compliance: Dict[str, Any],
        drift: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Generate actionable recommendations from metric analysis."""
        recommendations: List[Dict[str, Any]] = []

        # Maturity recommendations
        level_num = maturity.get("maturity_level_number", 1)
        if level_num < 3:
            recommendations.append({
                "area": "Program Maturity",
                "priority": "HIGH",
                "recommendation": (
                    f"Current maturity at Level {level_num} "
                    f"({maturity.get('maturity_level_name', 'Unknown')}). "
                    f"Target Level 3 (Defined) by implementing documented policies, "
                    f"SIEM integration, and behavioral indicator library."
                ),
            })

        # KRI recommendations
        kri_data = kris.get("kris", {})
        mttd = kri_data.get("mttd", {})
        if mttd.get("status") == "ABOVE_TARGET":
            recommendations.append({
                "area": "Detection Speed",
                "priority": "HIGH",
                "recommendation": (
                    f"MTTD ({mttd.get('current_7d', 0):.0f}s) exceeds target "
                    f"({mttd.get('target', 3600)}s). Improve automated detection "
                    f"rules and reduce manual triage bottlenecks."
                ),
            })

        fp_rate = kri_data.get("false_positive_rate", {})
        if fp_rate.get("status") == "ABOVE_TARGET":
            recommendations.append({
                "area": "Alert Quality",
                "priority": "MEDIUM",
                "recommendation": (
                    f"False positive rate ({fp_rate.get('current_7d', 0):.1f}%) "
                    f"exceeds target ({fp_rate.get('target', 5)}%). "
                    f"Tune detection thresholds and correlation rules."
                ),
            })

        # Compliance recommendations
        for fid, fdata in compliance.get("families", {}).items():
            if fdata.get("compliance_level") in (
                "NON_COMPLIANT", "MINIMALLY_COMPLIANT"
            ):
                recommendations.append({
                    "area": f"Compliance: {fdata['family_name']}",
                    "priority": "HIGH",
                    "recommendation": (
                        f"{fdata['family_name']} ({fid}) compliance at "
                        f"{fdata['score_percentage']}% ({fdata['compliance_level']}). "
                        f"Prioritize implementation of unmet controls."
                    ),
                })

        # Drift recommendations
        if drift.get("overall_status") == "DRIFT_DETECTED":
            recommendations.append({
                "area": "Model Health",
                "priority": "MEDIUM",
                "recommendation": (
                    f"Model drift detected in {drift.get('drift_detected_count', 0)} "
                    f"metric(s). Investigate root cause and consider model retraining."
                ),
            })

        if not recommendations:
            recommendations.append({
                "area": "General",
                "priority": "LOW",
                "recommendation": "All metrics within targets. Continue monitoring.",
            })

        return recommendations

    def get_stats(self) -> Dict[str, Any]:
        """Get dashboard statistics."""
        with self._lock:
            return {
                "detection_times_recorded": len(self._detection_times),
                "response_times_recorded": len(self._response_times),
                "alert_outcomes_recorded": len(self._alert_outcomes),
                "model_metrics_tracked": len(self._model_metrics),
                "control_assessments": len(self._control_assessments),
                "maturity_criteria_set": len(self._maturity_criteria_status),
                "kri_history_points": len(self._kri_history),
                "reports_generated": len(self._report_history),
            }
