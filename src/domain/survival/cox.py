"""
src/domain/survival/cox.py - Modelo de Riesgos Proporcionales de Cox con Empates Efron

Envuelve el modelo semi-paramétrico de Cox Proportional Hazards con aproximación 
de Efron para el manejo riguroso de empates temporales en contratos discretizados.
Aislado detrás del protocolo SurvivalEstimator para cumplir con Clean Architecture.

Fórmula:
    h(t | X) = h_0(t) * exp(beta_1 * X_1 + ... + beta_p * X_p)
    Hazard Ratio: HR_j = exp(beta_j)
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test


class CoxPHModel:
    """Modelo de Cox Proportional Hazards con aproximación de empates de Efron."""

    DEFAULT_COVARIATES = [
        "sla_breach_rate",
        "engineer_turnover_ratio",
        "budget_burn_variance",
        "ticket_escalation_velocity",
        "tech_stack_alignment_score",
        "payment_delay_days",
        "renewal_count",
    ]

    def __init__(self, penalizer: float = 0.01, ties: str = "efron"):
        self.penalizer = penalizer
        self.ties = ties
        self.fitter = CoxPHFitter(penalizer=penalizer, l1_ratio=0.0)
        self.covariate_cols: List[str] = []
        self.is_fitted: bool = False
        self.concordance_index_: float = 0.0

    def fit(
        self,
        df: pd.DataFrame,
        duration_col: str = "duration_days",
        event_col: str = "event_cancelled",
        covariates: Optional[List[str]] = None,
    ) -> "CoxPHModel":
        """
        Ajusta el modelo Cox PH sobre el DataFrame de contratos.
        
        Args:
            df: DataFrame con duración, evento y covariables.
            duration_col: Nombre de la columna de duración observada.
            event_col: Nombre de la columna del indicador binario de evento.
            covariates: Lista opcional de nombres de covariables a incluir.
        """
        self.covariate_cols = covariates or [c for c in self.DEFAULT_COVARIATES if c in df.columns]

        fit_df = df[[duration_col, event_col] + self.covariate_cols].copy()

        # Ajuste con aproximación de Efron (default estándar en lifelines)
        self.fitter.fit(
            fit_df,
            duration_col=duration_col,
            event_col=event_col,
            robust=True,
        )

        self.is_fitted = True
        self.concordance_index_ = float(self.fitter.concordance_index_)
        return self

    def predict_hazard_score(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calcula el Hazard Score multiplicativo relativo respecto a la media de la población:
            HS_i = exp(beta^T * (X_i - X_mean))
        """
        if not self.is_fitted:
            raise RuntimeError("El modelo CoxPH debe ser ajustado con fit() antes de predecir.")

        sub_df = df[self.covariate_cols]
        # predict_partial_hazard retorna exp(beta^T * (X - X_mean))
        partial_hazard = self.fitter.predict_partial_hazard(sub_df)
        return partial_hazard.to_numpy()

    def predict_survival_at_horizon(
        self,
        df: pd.DataFrame,
        horizon_days: float,
    ) -> np.ndarray:
        """Predice la probabilidad de supervivencia condicional S(horizon | X) para cada contrato."""
        if not self.is_fitted:
            raise RuntimeError("El modelo CoxPH debe ser ajustado antes de predecir supervivencia.")

        sub_df = df[self.covariate_cols]
        surv_df = self.fitter.predict_survival_function(sub_df, times=[horizon_days])
        # Retorna array 1D de probabilidades para el horizonte solicitado
        return surv_df.loc[horizon_days].to_numpy()

    def get_coefficients(self) -> Dict[str, float]:
        """Retorna el diccionario de coeficientes beta estimados."""
        if not self.is_fitted:
            raise RuntimeError("El modelo debe estar ajustado.")
        return {col: float(self.fitter.params_[col]) for col in self.covariate_cols}

    def get_hazard_ratios(self) -> Dict[str, float]:
        """Retorna los Hazard Ratios HR = exp(beta)."""
        if not self.is_fitted:
            raise RuntimeError("El modelo debe estar ajustado.")
        return {col: float(np.exp(self.fitter.params_[col])) for col in self.covariate_cols}

    def schoenfeld_test(self) -> pd.DataFrame:
        """
        Ejecuta el test de residuos de Schoenfeld para diagnosticar la asunción 
        de riesgos proporcionales (PH Assumption) para cada covariable.
        """
        if not self.is_fitted:
            raise RuntimeError("El modelo debe estar ajustado.")

        results = proportional_hazard_test(self.fitter, self.fitter._central_values)
        summary = results.summary.copy()
        summary["violates_ph"] = summary["p"] < 0.05
        return summary
