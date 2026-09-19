"""
tests/unit/test_phase1_domain.py - Pruebas Unitarias de Dominio Puro, DIP y Generador (Fase 1)

Verifica:
1. Inmutabilidad estricta de los contratos Pydantic v2 (frozen=True).
2. Conformidad estructural de los protocolos DIP (runtime_checkable).
3. Integridad estadística y determinismo del generador estocástico de Inetum.
"""

import os
import pytest
import numpy as np
import pandas as pd
from pydantic import ValidationError

from src.domain.contracts import (
    ContractTier,
    ServiceLine,
    RiskBand,
    ContractRecord,
    SurvivalPoint,
    PrescriptiveAction,
    RiskPrediction,
)
from src.domain.protocols import (
    ContractRepository,
    SurvivalEstimator,
    PrescriptiveEngine,
    ExtractPublisher,
)
from src.data.data_generator import generate_contract_dataset


# ==============================================================================
# 1. PRUEBAS DE INMUTABILIDAD Y MODELOS DE DOMINIO (PYDANTIC V2 FROZEN)
# ==============================================================================

def test_contract_record_immutability():
    """Verifica que ContractRecord es inmutable y lanza ValidationError ante mutaciones."""
    record = ContractRecord(
        contract_id="CNT-001",
        customer_id="CUST-001",
        contract_tier=ContractTier.ENTERPRISE,
        service_line=ServiceLine.CLOUD_INFRASTRUCTURE,
        duration_days=450.5,
        event_cancelled=0,
        arr=85000.0,
        sla_breach_rate=0.05,
        engineer_turnover_ratio=0.10,
        budget_burn_variance=0.02,
        ticket_escalation_velocity=2.0,
        tech_stack_alignment_score=0.90,
        payment_delay_days=5.0,
        renewal_count=1,
    )
    assert record.contract_id == "CNT-001"
    assert record.duration_days == 450.5

    with pytest.raises(ValidationError):
        # Intentar mutar un modelo frozen debe fallar
        record.duration_days = 999.0  # type: ignore


def test_survival_point_validation_bounds():
    """Verifica validación de cotas de probabilidad [0, 1] en SurvivalPoint."""
    point = SurvivalPoint(
        time_days=180.0,
        survival_probability=0.885,
        lower_ci=0.850,
        upper_ci=0.920,
        at_risk_count=4500,
        event_count=120,
    )
    assert 0.0 <= point.survival_probability <= 1.0

    # Probabilidad fuera de cota debe fallar
    with pytest.raises(ValidationError):
        SurvivalPoint(
            time_days=180.0,
            survival_probability=1.25,  # Inválido
            lower_ci=0.85,
            upper_ci=0.92,
            at_risk_count=100,
            event_count=10,
        )


# ==============================================================================
# 2. PRUEBAS DE PROTOCOLOS DE INVERSIÓN DE DEPENDENCIAS (DIP)
# ==============================================================================

class MockContractRepository:
    """Implementación mock que satisface el protocolo ContractRepository."""
    def read_contracts(self, limit=None) -> pd.DataFrame:
        return pd.DataFrame({"contract_id": ["C1"]})

    def write_marts(self, risk_mart, survival_curve_mart) -> None:
        pass


class MockSurvivalEstimator:
    """Implementación mock que satisface el protocolo SurvivalEstimator."""
    def fit(self, durations, events, covariates=None):
        return self

    def predict_survival_curve(self):
        return []

    def predict_hazard_score(self, covariates):
        return [1.0]


def test_dip_protocols_conformance():
    """Verifica que las implementaciones mock cumplen con los protocolos abstractos del dominio."""
    mock_repo = MockContractRepository()
    mock_estimator = MockSurvivalEstimator()

    assert isinstance(mock_repo, ContractRepository)
    assert isinstance(mock_estimator, SurvivalEstimator)


# ==============================================================================
# 3. PRUEBAS DEL GENERADOR ESTOCÁSTICO WEIBULL (FÍSICA DE CONTRATOS)
# ==============================================================================

def test_generator_schema_and_bounds(tmp_path):
    """Verifica que el generador produce el esquema correcto con valores en cotas válidas."""
    output_file = str(tmp_path / "test_contracts.parquet")
    df = generate_contract_dataset(n_records=2000, seed=42, output_path=output_file)

    assert len(df) == 2000
    assert os.path.exists(output_file)

    # Validaciones dimensionales y estadísticas
    assert set(df["event_cancelled"].unique()).issubset({0, 1})
    assert (df["duration_days"] > 0).all()
    assert (df["arr"] > 0).all()
    assert (df["sla_breach_rate"] >= 0).all()
    assert (df["tech_stack_alignment_score"].between(0.0, 1.0)).all()

    # Tasa de censura realista (entre 30% y 65%)
    censoring_rate = 1.0 - df["event_cancelled"].mean()
    assert 0.30 <= censoring_rate <= 0.65


def test_generator_determinism():
    """Verifica determinismo estricto: misma semilla produce exactamente los mismos datos."""
    df1 = generate_contract_dataset(n_records=500, seed=12345, output_path=None)
    df2 = generate_contract_dataset(n_records=500, seed=12345, output_path=None)

    pd.testing.assert_frame_equal(df1, df2)
