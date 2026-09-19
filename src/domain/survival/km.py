"""
src/domain/survival/km.py - Estimador No Paramétrico de Kaplan-Meier (Primeros Principios)

Implementa la estimación de la función de supervivencia S(t) = P(T > t), la varianza 
de Greenwood y las bandas de confianza al 95% mediante la transformación log-log 
para garantizar que las cotas permanezcan estrictamente en [0, 1].

Fórmulas Matemáticas:
    S_hat(t) = Prod_{t_j <= t} (1 - d_j / n_j)
    Var_hat(S_hat(t)) = S_hat(t)^2 * Sum_{t_j <= t} [ d_j / (n_j * (n_j - d_j)) ]
    CI_loglog = S_hat(t) ^ exp( +/- z_{1 - alpha/2} * sqrt(Var_hat) / (S_hat(t) * ln(S_hat(t))) )
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from src.domain.contracts import SurvivalPoint


class KaplanMeierEstimator:
    """Estimador de supervivencia de Kaplan-Meier puro en NumPy (Sub-5ms)."""

    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.z_score = 1.95996  # Para 95% de confianza (Normal estándar)
        self.times_: np.ndarray = np.array([])
        self.survival_: np.ndarray = np.array([])
        self.lower_ci_: np.ndarray = np.array([])
        self.upper_ci_: np.ndarray = np.array([])
        self.at_risk_: np.ndarray = np.array([])
        self.events_: np.ndarray = np.array([])
        self.is_fitted: bool = False

    def fit(self, durations: np.ndarray, events: np.ndarray) -> "KaplanMeierEstimator":
        """
        Ajusta el estimador no paramétrico a los pares observados (duración, evento).
        
        Args:
            durations: Array 1D de tiempos observados Y_i = min(T_i, C_i) > 0.
            events: Array 1D binario (1 si ocurrió rescisión, 0 si fue censurado).
        """
        durations = np.asarray(durations, dtype=float)
        events = np.asarray(events, dtype=int)

        if len(durations) == 0:
            raise ValueError("El conjunto de datos de duraciones no puede estar vacío.")
        if len(durations) != len(events):
            raise ValueError("Duraciones y eventos deben tener exactamente la misma longitud.")

        # 1. Identificar tiempos de evento únicos ordenados
        unique_times = np.sort(np.unique(durations))
        n_times = len(unique_times)

        # 2. Conteo vectorizado O(N log N) de sujetos en riesgo n_j y eventos d_j
        sorted_durations = np.sort(durations)
        left_idx = np.searchsorted(sorted_durations, unique_times, side="left")
        at_risk = len(durations) - left_idx

        event_mask = (events == 1)
        if np.any(event_mask):
            event_positions = np.searchsorted(unique_times, durations[event_mask])
            d_events = np.bincount(event_positions, minlength=n_times)[:n_times]
        else:
            d_events = np.zeros(n_times, dtype=int)

        # 3. Probabilidades condicionales de supervivencia p_j = 1 - (d_j / n_j)
        # Evitar divisiones por cero si n_j == 0
        valid_mask = at_risk > 0
        p_step = np.ones(n_times, dtype=float)
        p_step[valid_mask] = 1.0 - (d_events[valid_mask] / at_risk[valid_mask])

        # Función de supervivencia acumulada: S(t) = Prod p_j
        survival = np.cumprod(p_step)

        # 4. Varianza de Greenwood: Sum [ d_j / (n_j * (n_j - d_j)) ]
        denom = at_risk * (at_risk - d_events)
        var_sum_terms = np.zeros(n_times, dtype=float)
        valid_denom = (denom > 0) & (at_risk > d_events)
        var_sum_terms[valid_denom] = d_events[valid_denom] / denom[valid_denom]

        cumulative_greenwood_sum = np.cumsum(var_sum_terms)
        std_error = survival * np.sqrt(cumulative_greenwood_sum)

        # 5. Bandas de confianza al 95% con transformación Log-Log vectorizada
        # Previene que CI < 0 o CI > 1 y acelera la ejecución a sub-milisegundo
        lower_ci = np.where(survival >= 1.0, 1.0, 0.0)
        upper_ci = np.where(survival >= 1.0, 1.0, 0.0)

        valid_mask = (survival > 0.0) & (survival < 1.0) & (std_error > 0.0)
        if np.any(valid_mask):
            s_val = survival[valid_mask]
            se_val = std_error[valid_mask]
            log_s = np.log(s_val)
            sigma_loglog = se_val / (s_val * np.abs(log_s))
            exponent = np.exp(self.z_score * sigma_loglog)
            lower_ci[valid_mask] = np.clip(s_val ** exponent, 0.0, 1.0)
            upper_ci[valid_mask] = np.clip(s_val ** (1.0 / exponent), 0.0, 1.0)

        # Punto t = 0 (invariante inicial S(0) = 1.0)
        self.times_ = np.insert(unique_times, 0, 0.0)
        self.survival_ = np.insert(survival, 0, 1.0)
        self.lower_ci_ = np.insert(lower_ci, 0, 1.0)
        self.upper_ci_ = np.insert(upper_ci, 0, 1.0)
        self.at_risk_ = np.insert(at_risk, 0, len(durations))
        self.events_ = np.insert(d_events, 0, 0)
        self.is_fitted = True

        return self

    def predict_survival(self, t: float) -> float:
        """Retorna S(t) para cualquier tiempo arbitrario t mediante interpolación de escalón."""
        if not self.is_fitted:
            raise RuntimeError("El estimador KaplanMeier debe ajustarse con fit() antes de predecir.")
        if t <= 0.0:
            return 1.0
        # Buscar el mayor tiempo t_j <= t
        idx = np.searchsorted(self.times_, t, side="right") - 1
        idx = np.clip(idx, 0, len(self.survival_) - 1)
        return float(self.survival_[idx])

    def to_survival_points(self, max_points: int = 150) -> List[SurvivalPoint]:
        """
        Retorna la curva reducida a max_points coordenadas representativas
        optimizadas para almacenamiento y consumo visual en Tableau/Power BI.
        """
        if not self.is_fitted:
            raise RuntimeError("El estimador debe ajustarse antes de exportar puntos.")

        n_total = len(self.times_)
        if n_total <= max_points:
            indices = range(n_total)
        else:
            indices = np.linspace(0, n_total - 1, max_points, dtype=int)

        points = []
        for idx in indices:
            points.append(
                SurvivalPoint(
                    time_days=round(float(self.times_[idx]), 1),
                    survival_probability=round(float(self.survival_[idx]), 4),
                    lower_ci=round(float(self.lower_ci_[idx]), 4),
                    upper_ci=round(float(self.upper_ci_[idx]), 4),
                    at_risk_count=int(self.at_risk_[idx]),
                    event_count=int(self.events_[idx]),
                )
            )
        return points

    def to_dataframe(self) -> pd.DataFrame:
        """Exporta la curva completa a DataFrame para persistencia Lakehouse."""
        if not self.is_fitted:
            raise RuntimeError("El estimador debe ajustarse antes de exportar a DataFrame.")

        return pd.DataFrame({
            "time_days": self.times_,
            "survival_probability": np.round(self.survival_, 5),
            "lower_ci": np.round(self.lower_ci_, 5),
            "upper_ci": np.round(self.upper_ci_, 5),
            "at_risk_count": self.at_risk_,
            "event_count": self.events_,
        })
