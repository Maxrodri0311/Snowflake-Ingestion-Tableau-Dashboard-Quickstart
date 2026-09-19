"""
src/domain/protocols.py - Contratos Abstractos e Interfaces (Clean Architecture / DIP)

Define las abstracciones e interfaces del dominio analítico sin acoplamiento a 
tecnologías de persistencia concretas (DuckDB, Snowflake, Parquet, Tableau).
Cualquier repositorio, estimador o publicador debe implementar estos protocolos.
"""

from typing import Protocol, List, Dict, Any, Optional, runtime_checkable
import pandas as pd
from src.domain.contracts import (
    ContractRecord,
    SurvivalPoint,
    RiskPrediction,
    PrescriptiveAction,
)


@runtime_checkable
class ContractRepository(Protocol):
    """Protocolo de acceso a datos para ingesta y persistencia dimensional."""

    def read_contracts(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Carga el dataset de contratos para entrenamiento o inferencia."""
        ...

    def write_marts(
        self,
        risk_mart: pd.DataFrame,
        survival_curve_mart: pd.DataFrame,
    ) -> None:
        """Persiste los data marts analíticos procesados."""
        ...


@runtime_checkable
class SurvivalEstimator(Protocol):
    """Protocolo para motores matemáticos de análisis de supervivencia."""

    def fit(
        self,
        durations: List[float],
        events: List[int],
        covariates: Optional[pd.DataFrame] = None,
    ) -> "SurvivalEstimator":
        """Ajusta el estimador a los datos observados de tiempo-a-evento."""
        ...

    def predict_survival_curve(self) -> List[SurvivalPoint]:
        """Retorna las coordenadas de la curva de supervivencia agregada."""
        ...

    def predict_hazard_score(self, covariates: pd.DataFrame) -> List[float]:
        """Calcula el hazard multiplicativo o riesgo relativo exp(beta^T * X)."""
        ...


@runtime_checkable
class PrescriptiveEngine(Protocol):
    """Protocolo para motores de prescripción operativa y runbooks de mitigación."""

    def prescribe(
        self,
        contract_id: str,
        hazard_score: float,
        drivers: Dict[str, float],
        arr: float,
    ) -> PrescriptiveAction:
        """Determina la acción operativa recomendada según los drivers de riesgo."""
        ...


@runtime_checkable
class ExtractPublisher(Protocol):
    """Protocolo para publicación y exportación de artefactos analíticos (Tableau/BI)."""

    def publish(self, df: pd.DataFrame, target_path: str) -> None:
        """Exporta un DataFrame optimizado hacia la capa de visualización."""
        ...
