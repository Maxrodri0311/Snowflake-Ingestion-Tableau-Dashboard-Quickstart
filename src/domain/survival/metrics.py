"""
src/domain/survival/metrics.py - Métricas de Negocio, ECRL y Prescripción Operativa

Traduce las probabilidades y hazards estadísticos en indicadores financieros
y acciones operativas directas para los directores de cuenta de Inetum:
1. Expected Contract Residual Life (ECRL / RMST) mediante integración numérica.
2. Gross ARR at Risk ponderado por la probabilidad condicional de rescisión.
3. Prescriptive Action Engine que mapea los drivers de riesgo a runbooks operativos.
"""

from typing import Dict, Any, Callable, List
import numpy as np
from src.domain.contracts import RiskBand, PrescriptiveAction


def compute_risk_band(hazard_score: float) -> RiskBand:
    """Clasifica el hazard multiplicativo relativo en bandas operativas de riesgo."""
    if hazard_score < 0.80:
        return RiskBand.LOW
    elif hazard_score < 1.50:
        return RiskBand.MEDIUM
    elif hazard_score < 2.50:
        return RiskBand.HIGH
    else:
        return RiskBand.CRITICAL


def compute_ecrl(
    survival_evaluator: Callable[[float], float],
    current_age_days: float,
    horizon_days: float = 180.0,
    n_steps: int = 36,
) -> float:
    """
    Calcula la Vida Residual Media Restringida (ECRL / RMST) en días:
        ECRL(s, tau) = [ Integral_s^tau S(u) du ] / S(s)
        
    Args:
        survival_evaluator: Función que retorna S(t) para cualquier tiempo t.
        current_age_days: Antigüedad observada del contrato (s).
        horizon_days: Ventana de horizonte temporal (tau - s, típicamente 180 días).
        n_steps: Pasos de integración numérica trapezoidal.
        
    Returns:
        Días de vida residual esperados dentro de la ventana de horizonte [0, horizon_days].
    """
    s = max(0.0, current_age_days)
    tau = s + horizon_days
    s_at_s = survival_evaluator(s)

    if s_at_s <= 1e-6:
        return 0.0

    eval_times = np.linspace(s, tau, n_steps)
    surv_values = np.array([survival_evaluator(t) for t in eval_times])

    # Integración numérica por regla trapezoidal (compatible con NumPy 2.0+ y SciPy)
    try:
        from scipy.integrate import trapezoid
        area = float(trapezoid(surv_values, eval_times))
    except (ImportError, AttributeError):
        trap_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
        area = float(trap_fn(surv_values, eval_times))

    ecrl = area / s_at_s

    # Debe estar acotado entre 0 y el horizonte máximo
    return float(np.clip(ecrl, 0.0, horizon_days))


def compute_arr_at_risk(
    arr: float,
    current_survival: float,
    horizon_survival: float,
) -> float:
    """
    Calcula el ARR Bruto en riesgo para el horizonte evaluado:
        P(T <= t + H | T > t) = 1 - S(t + H) / S(t)
        ARR_at_Risk = ARR * P(rescisión condicional)
    """
    if current_survival <= 1e-6:
        return arr

    prob_churn_conditional = 1.0 - (horizon_survival / current_survival)
    prob_churn_conditional = np.clip(prob_churn_conditional, 0.0, 1.0)
    return float(np.round(arr * prob_churn_conditional, 2))


def prescribe_action(
    contract_id: str,
    hazard_score: float,
    row: Dict[str, Any],
    arr: float,
) -> PrescriptiveAction:
    """
    Evalúa los factores de riesgo del contrato y selecciona el runbook de remediación
    específico más efectivo para Inetum.
    """
    sla_breach = float(row.get("sla_breach_rate", 0.0))
    turnover = float(row.get("engineer_turnover_ratio", 0.0))
    budget_burn = float(row.get("budget_burn_variance", 0.0))
    escalation = float(row.get("ticket_escalation_velocity", 0.0))
    stack_align = float(row.get("tech_stack_alignment_score", 1.0))
    payment_delay = float(row.get("payment_delay_days", 0.0))

    risk_band = compute_risk_band(hazard_score)

    if risk_band == RiskBand.LOW:
        return PrescriptiveAction(
            contract_id=contract_id,
            action_code="ACT-01-MONITOR",
            severity="INFO",
            primary_driver="Healthy Operational State",
            recommended_playbook="Monitorización rutinaria mensual de métricas SLA y balance presupuestario.",
            target_stakeholder="Service Delivery Manager",
        )

    # Identificar el driver dominante de riesgo
    if sla_breach > 0.12:
        return PrescriptiveAction(
            contract_id=contract_id,
            action_code="ACT-02-SLA-REMED",
            severity="CRITICAL" if hazard_score > 2.0 else "HIGH",
            primary_driver=f"SLA Breach Rate Elevado ({sla_breach*100:.1f}%)",
            recommended_playbook="Auditoría técnica de entregables, revisión de dependencias de cliente y acuerdo de exención de penalizaciones.",
            target_stakeholder="Service Delivery Lead & CTO Cliente",
        )
    elif turnover > 0.22:
        return PrescriptiveAction(
            contract_id=contract_id,
            action_code="ACT-03-STAFF-RETAIN",
            severity="HIGH",
            primary_driver=f"Rotación Crítica de Personal ({turnover*100:.1f}%)",
            recommended_playbook="Inyección inmediata de Tech Leads senior de retén y plan de transferencia de conocimiento formal.",
            target_stakeholder="Resource Manager & Talent Lead",
        )
    elif escalation > 4.5:
        return PrescriptiveAction(
            contract_id=contract_id,
            action_code="ACT-04-WAR-ROOM",
            severity="CRITICAL",
            primary_driver=f"Alta Velocidad de Escalamientos ({escalation:.1f} tickets/mes)",
            recommended_playbook="Apertura de mesa de crisis técnica ejecutiva ('War Room') con standups diarios y desbloqueo de tickets P1/P2.",
            target_stakeholder="Delivery Director",
        )
    elif budget_burn > 0.15:
        return PrescriptiveAction(
            contract_id=contract_id,
            action_code="ACT-05-BUDGET-REALIGN",
            severity="MEDIUM" if hazard_score < 2.0 else "HIGH",
            primary_driver=f"Desviación Presupuestaria ({budget_burn*100:.1f}%)",
            recommended_playbook="Reunión de gobernanza de alcance (Scope Control) y re-estimación de horas de consultoría.",
            target_stakeholder="Project Manager & Finance Controller",
        )
    elif stack_align < 0.55:
        return PrescriptiveAction(
            contract_id=contract_id,
            action_code="ACT-06-TECH-ENABLE",
            severity="MEDIUM",
            primary_driver=f"Fricción de Stack Tecnológico ({stack_align*100:.1f}%)",
            recommended_playbook="Capacitación acelerada en arquitectura del cliente y simplificación de herramientas operativas.",
            target_stakeholder="Principal Cloud Architect",
        )
    elif payment_delay > 25.0:
        return PrescriptiveAction(
            contract_id=contract_id,
            action_code="ACT-07-COMMERCIAL-ALIGN",
            severity="HIGH",
            primary_driver=f"Retraso Recurrente en Facturación ({payment_delay:.0f} días)",
            recommended_playbook="Revisión ejecutiva de términos de pago y alineación con la dirección de compras del cliente.",
            target_stakeholder="Account Executive & CFO",
        )
    else:
        return PrescriptiveAction(
            contract_id=contract_id,
            action_code="ACT-08-EXEC-SPONSOR",
            severity="CRITICAL" if risk_band == RiskBand.CRITICAL else "HIGH",
            primary_driver=f"Riesgo Acumulado Multinivel (Hazard {hazard_score:.2f}x)",
            recommended_playbook="Reunión estratégica ejecutiva ('Executive Check-in') con el sponsor C-Level del cliente.",
            target_stakeholder="VP of Client Accounts",
        )
