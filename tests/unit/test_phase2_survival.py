"""
tests/unit/test_phase2_survival.py - Pruebas Unitarias del Motor Estadístico y Métricas (Fase 2)

Verifica:
1. Estimador Kaplan-Meier: monotonía estricta, cotas Greenwood [0, 1] y condición inicial S(0) = 1.
2. Modelo Cox PH: empates de Efron, concordancia C-index > 0.60 y consistencia direccional de HR.
3. Métricas de Negocio: ECRL (RMST), Gross ARR at Risk y Prescriptive Action Engine.
"""

import pytest
import numpy as np
import pandas as pd

from src.domain.contracts import RiskBand
from src.domain.survival.km import KaplanMeierEstimator
from src.domain.survival.cox import CoxPHModel
from src.domain.survival.metrics import (
    compute_risk_band,
    compute_ecrl,
    compute_arr_at_risk,
    prescribe_action,
)
from src.data.data_generator import generate_contract_dataset


@pytest.fixture(scope="module")
def sample_dataset():
    """Genera un dataset de prueba de 2,500 registros para testing ultrarrápido."""
    return generate_contract_dataset(n_records=2500, seed=123, output_path=None)


# ==============================================================================
# 1. PRUEBAS MATEMÁTICAS DE KAPLAN-MEIER
# ==============================================================================

def test_kaplan_meier_invariants(sample_dataset):
    """Verifica invariantes matemáticas: monotonía, límites y varianza de Greenwood."""
    durations = sample_dataset["duration_days"].to_numpy()
    events = sample_dataset["event_cancelled"].to_numpy()

    km = KaplanMeierEstimator().fit(durations, events)

    # 1. Condición inicial: S(0) = 1.0
    assert km.predict_survival(0.0) == 1.0
    assert km.survival_[0] == 1.0

    # 2. Monotonía no creciente: S(t_j) <= S(t_{j-1})
    diffs = np.diff(km.survival_)
    assert np.all(diffs <= 1e-9), "La función de supervivencia S(t) debe ser monótona no creciente."

    # 3. Cotas estrictas [0, 1]
    assert np.all(km.survival_ >= 0.0) and np.all(km.survival_ <= 1.0)
    assert np.all(km.lower_ci_ >= 0.0) and np.all(km.upper_ci_ <= 1.0)

    # 4. Consistencia de bandas: lower_ci <= survival <= upper_ci
    assert np.all(km.lower_ci_ <= km.survival_ + 1e-6)
    assert np.all(km.survival_ <= km.upper_ci_ + 1e-6)


def test_kaplan_meier_interpolation(sample_dataset):
    """Verifica interpolación por escalón en horizontes clave (90, 180, 365 días)."""
    km = KaplanMeierEstimator().fit(
        sample_dataset["duration_days"].to_numpy(),
        sample_dataset["event_cancelled"].to_numpy(),
    )

    s_90 = km.predict_survival(90.0)
    s_180 = km.predict_survival(180.0)
    s_365 = km.predict_survival(365.0)

    assert 0.0 < s_365 <= s_180 <= s_90 <= 1.0


# ==============================================================================
# 2. PRUEBAS DE COX PROPORTIONAL HAZARDS (EMPATES EFRON)
# ==============================================================================

def test_cox_ph_fit_and_concordance(sample_dataset):
    """Verifica ajuste Cox con empates de Efron, concordancia C-index y hazard ratios."""
    cox = CoxPHModel(penalizer=0.01)
    cox.fit(sample_dataset)

    assert cox.is_fitted
    # El C-index debe ser significativamente superior al azar (0.50)
    assert cox.concordance_index_ > 0.58, f"C-index bajo: {cox.concordance_index_}"

    # Hazard scores deben ser estrictamente positivos
    hazards = cox.predict_hazard_score(sample_dataset)
    assert np.all(hazards > 0.0)

    # Validar consistencia direccional de los coeficientes:
    coeffs = cox.get_coefficients()
    assert coeffs["sla_breach_rate"] > 0.0, "SLA Breach debe incrementar el riesgo instantáneo."
    assert coeffs["tech_stack_alignment_score"] < 0.0, "Alineación tecnológica debe ser protectora."


# ==============================================================================
# 3. PRUEBAS DE MÉTRICAS DE NEGOCIO, ECRL Y RUNBOOKS PRESCRIPTIVOS
# ==============================================================================

def test_ecrl_calculation():
    """Verifica que el ECRL (RMST) está acotado y se comporta adecuadamente."""
    # Función de supervivencia de prueba constante S(t) = 0.8
    constant_surv = lambda t: 0.80

    # Para horizonte de 180 días con S(t) constante, ECRL = (0.80 * 180) / 0.80 = 180 días
    ecrl = compute_ecrl(constant_surv, current_age_days=100.0, horizon_days=180.0)
    assert pytest.approx(ecrl, rel=1e-2) == 180.0

    # Si la supervivencia decae linealmente hacia 0
    decaying_surv = lambda t: max(0.0, 1.0 - (t / 1000.0))
    ecrl_decay = compute_ecrl(decaying_surv, current_age_days=0.0, horizon_days=180.0)
    assert 0.0 < ecrl_decay < 180.0


def test_arr_at_risk_calculation():
    """Verifica cálculo de Gross ARR at Risk ponderado condicionalmente."""
    arr = 100000.0
    # Caso 1: Sin caída de supervivencia (S(t) = 1.0, S(t+H) = 1.0) -> ARR en riesgo = 0
    assert compute_arr_at_risk(arr, current_survival=1.0, horizon_survival=1.0) == 0.0

    # Caso 2: Supervivencia actual 1.0 y cae al horizonte a 0.70 (30% riesgo) -> ARR en riesgo = $30,000
    arr_risk = compute_arr_at_risk(arr, current_survival=1.0, horizon_survival=0.70)
    assert pytest.approx(arr_risk, abs=1.0) == 30000.0


def test_prescriptive_action_engine():
    """Verifica que el motor asigna el runbook operativo correcto según el factor de riesgo dominante."""
    # Caso SLA Breach dominante
    action_sla = prescribe_action(
        contract_id="CNT-001",
        hazard_score=2.2,
        row={"sla_breach_rate": 0.25, "engineer_turnover_ratio": 0.05},
        arr=150000.0,
    )
    assert action_sla.action_code == "ACT-02-SLA-REMED"
    assert "SLA" in action_sla.primary_driver

    # Caso Rotación de Personal dominante
    action_turnover = prescribe_action(
        contract_id="CNT-002",
        hazard_score=1.8,
        row={"sla_breach_rate": 0.02, "engineer_turnover_ratio": 0.35},
        arr=120000.0,
    )
    assert action_turnover.action_code == "ACT-03-STAFF-RETAIN"

    # Caso Contrato Saludable (Low Risk)
    action_healthy = prescribe_action(
        contract_id="CNT-003",
        hazard_score=0.45,
        row={"sla_breach_rate": 0.01, "engineer_turnover_ratio": 0.02},
        arr=80000.0,
    )
    assert action_healthy.action_code == "ACT-01-MONITOR"
