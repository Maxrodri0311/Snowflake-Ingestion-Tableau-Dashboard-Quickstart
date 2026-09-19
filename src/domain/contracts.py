"""
src/domain/contracts.py - Modelos de Dominio Inmutables (Pydantic v2 Frozen)

Define los DTOs y entidades de negocio fuertemente tipadas y protegidas contra
mutaciones en tiempo de ejecución. Representa contratos, puntos de supervivencia
y alertas prescriptivas del ecosistema Inetum.
"""

from enum import Enum
from typing import Optional, Dict
from pydantic import BaseModel, ConfigDict, Field


class ContractTier(str, Enum):
    """Categorización corporativa de clientes según escala y ARR."""
    STRATEGIC = "STRATEGIC"
    ENTERPRISE = "ENTERPRISE"
    MID_MARKET = "MID_MARKET"
    STANDARD = "STANDARD"


class ServiceLine(str, Enum):
    """Líneas de servicio tecnológicas principales en Inetum."""
    MANAGED_SERVICES = "MANAGED_SERVICES"
    CLOUD_INFRASTRUCTURE = "CLOUD_INFRASTRUCTURE"
    CYBERSECURITY = "CYBERSECURITY"
    DATA_AI = "DATA_AI"
    APPLICATION_DEV = "APPLICATION_DEV"


class RiskBand(str, Enum):
    """Segmentación operativa del riesgo de rescisión."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ContractRecord(BaseModel):
    """Registro inmutable de un contrato B2B observado."""
    model_config = ConfigDict(frozen=True)

    contract_id: str
    customer_id: str
    contract_tier: ContractTier
    service_line: ServiceLine
    duration_days: float = Field(..., gt=0, description="Días observados en riesgo o hasta evento")
    event_cancelled: int = Field(..., ge=0, le=1, description="1 si ocurrió rescisión, 0 si está censurado")
    arr: float = Field(..., gt=0, description="Annual Recurring Revenue en USD")
    sla_breach_rate: float = Field(..., ge=0.0, description="Ratio de penalizaciones o brechas SLA")
    engineer_turnover_ratio: float = Field(..., ge=0.0, description="Rotación del personal técnico asignado")
    budget_burn_variance: float = Field(..., description="Desviación de consumo presupuestario vs plan")
    ticket_escalation_velocity: float = Field(..., ge=0.0, description="Tickets críticos escalados por mes")
    tech_stack_alignment_score: float = Field(..., ge=0.0, le=1.0, description="Alineación tecnológica cliente/equipo")
    payment_delay_days: float = Field(..., ge=0.0, description="Días promedio de retraso en pagos")
    renewal_count: int = Field(..., ge=0, description="Número de renovaciones históricas previas")


class SurvivalPoint(BaseModel):
    """Coordenada puntual en la curva de supervivencia S(t)."""
    model_config = ConfigDict(frozen=True)

    time_days: float = Field(..., ge=0.0)
    survival_probability: float = Field(..., ge=0.0, le=1.0)
    lower_ci: float = Field(..., ge=0.0, le=1.0)
    upper_ci: float = Field(..., ge=0.0, le=1.0)
    at_risk_count: int = Field(..., ge=0)
    event_count: int = Field(..., ge=0)


class PrescriptiveAction(BaseModel):
    """Acción de remediación operativa prescriptiva para mitigar riesgo de ARR."""
    model_config = ConfigDict(frozen=True)

    contract_id: str
    action_code: str
    severity: str
    primary_driver: str
    recommended_playbook: str
    target_stakeholder: str


class RiskPrediction(BaseModel):
    """Predicción multidimensional de riesgo y exposición financiera."""
    model_config = ConfigDict(frozen=True)

    contract_id: str
    hazard_score: float = Field(..., gt=0.0, description="Riesgo relativo instantáneo exp(beta^T * X)")
    risk_band: RiskBand
    survival_90d: float = Field(..., ge=0.0, le=1.0)
    survival_180d: float = Field(..., ge=0.0, le=1.0)
    survival_365d: float = Field(..., ge=0.0, le=1.0)
    ecrl_days: float = Field(..., ge=0.0, description="Expected Contract Residual Life (RMST) en días")
    gross_arr_at_risk_180d: float = Field(..., ge=0.0, description="ARR bruto ponderado en riesgo a 180 días")
    prescriptive_action: PrescriptiveAction
