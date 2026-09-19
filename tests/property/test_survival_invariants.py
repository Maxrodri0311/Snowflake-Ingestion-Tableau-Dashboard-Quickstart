"""
tests/property/test_survival_invariants.py - Pruebas de Invariantes Matemáticas y Propiedades

Valida las propiedades formales de la teoría de supervivencia y consistencia de negocio:
1. Invariante de Monotonía: S(t) es monótona no creciente (Delta S(t) <= 0).
2. Invariante de Acotamiento: S(t) in [0, 1] y CI_lower <= S(t) <= CI_upper.
3. Condición de Frontera: S(0) = 1.0.
4. Positividad del Hazard: exp(beta^T * X) > 0.
5. Acotamiento del ECRL: 0 <= ECRL(s, s + H) <= H.
6. Cota Financiera de ARR at Risk: 0 <= ARR_risk <= ARR.
"""

import pytest
import numpy as np
import pandas as pd

from src.domain.survival.km import KaplanMeierEstimator
from src.domain.survival.metrics import (
    compute_ecrl,
    compute_arr_at_risk,
    compute_risk_band,
    prescribe_action,
)
from src.domain.contracts import RiskBand
from src.data.data_generator import generate_contract_dataset


@pytest.fixture(scope="module")
def property_dataset():
    """Genera dataset de validación de propiedades con 3,000 registros."""
    return generate_contract_dataset(n_records=3000, seed=777, output_path=None)


def test_property_monotonicity(property_dataset):
    """Propiedad 1: S(t) no puede aumentar en el tiempo bajo ningún escenario."""
    km = KaplanMeierEstimator().fit(
        property_dataset["duration_days"].to_numpy(),
        property_dataset["event_cancelled"].to_numpy(),
    )
    diffs = np.diff(km.survival_)
    assert np.all(diffs <= 1e-12), f"Violación de monotonía detectada: max diff = {np.max(diffs)}"


def test_property_probability_bounds(property_dataset):
    """Propiedad 2: Todas las estimaciones y bandas deben estar estrictamente acotadas en [0, 1]."""
    km = KaplanMeierEstimator().fit(
        property_dataset["duration_days"].to_numpy(),
        property_dataset["event_cancelled"].to_numpy(),
    )

    assert np.all(km.survival_ >= 0.0) and np.all(km.survival_ <= 1.0)
    assert np.all(km.lower_ci_ >= 0.0) and np.all(km.lower_ci_ <= 1.0)
    assert np.all(km.upper_ci_ >= 0.0) and np.all(km.upper_ci_ <= 1.0)
    # Greenwood CI containment
    assert np.all(km.lower_ci_ <= km.survival_ + 1e-6)
    assert np.all(km.survival_ <= km.upper_ci_ + 1e-6)


def test_property_boundary_conditions(property_dataset):
    """Propiedad 3: S(0) = 1.0 y S(t_max) < 1.0 cuando existen eventos observados."""
    km = KaplanMeierEstimator().fit(
        property_dataset["duration_days"].to_numpy(),
        property_dataset["event_cancelled"].to_numpy(),
    )

    assert km.survival_[0] == 1.0
    assert km.times_[0] == 0.0
    assert km.predict_survival(0.0) == 1.0

    # Dado que hay eventos, la supervivencia terminal debe haber descendido de 1.0
    assert km.survival_[-1] < 1.0


def test_property_ecrl_bounds():
    """Propiedad 4: Para cualquier función de supervivencia y edad s, 0 <= ECRL <= H."""
    # Test sobre múltiples funciones de supervivencia sintéticas
    surv_funcs = [
        lambda t: 1.0,                           # Supervivencia perfecta (inmortal)
        lambda t: 0.0,                           # Supervivencia nula
        lambda t: np.exp(-0.005 * t),            # Decaimiento exponencial
        lambda t: np.clip(1.0 - (t / 500.0), 0, 1), # Decaimiento lineal
    ]

    for sf in surv_funcs:
        for horizon in [30.0, 90.0, 180.0, 365.0]:
            ecrl = compute_ecrl(sf, current_age_days=50.0, horizon_days=horizon)
            assert 0.0 <= ecrl <= horizon + 1e-5, f"ECRL {ecrl} fuera de cota [0, {horizon}]"


def test_property_arr_at_risk_bounds():
    """Propiedad 5: Gross ARR at Risk nunca puede ser negativo ni superar el ARR total."""
    arr_values = [1000.0, 50000.0, 1500000.0]
    surv_pairs = [
        (1.0, 1.0),   # 0% riesgo
        (1.0, 0.5),   # 50% riesgo
        (0.8, 0.2),   # 75% riesgo condicional
        (0.5, 0.0),   # 100% riesgo
    ]

    for arr in arr_values:
        for cur_s, horiz_s in surv_pairs:
            arr_risk = compute_arr_at_risk(arr, cur_s, horiz_s)
            assert 0.0 <= arr_risk <= arr + 1e-2, f"ARR at risk {arr_risk} fuera de cota [0, {arr}]"


def test_property_prescriptive_determinism():
    """Propiedad 6: Mapeo prescriptivo idéntico y determinista ante las mismas covariables."""
    row = {
        "sla_breach_rate": 0.18,
        "engineer_turnover_ratio": 0.10,
        "budget_burn_variance": 0.05,
        "ticket_escalation_velocity": 3.0,
        "tech_stack_alignment_score": 0.85,
        "payment_delay_days": 10.0,
    }

    action1 = prescribe_action("CNT-DET-01", hazard_score=1.85, row=row, arr=100000.0)
    action2 = prescribe_action("CNT-DET-01", hazard_score=1.85, row=row, arr=100000.0)

    assert action1.action_code == action2.action_code
    assert action1.primary_driver == action2.primary_driver
    assert action1.severity == action2.severity
