"""
src/application/pipeline.py - Orquestador de Aplicación (Application Layer)

Coordina los casos de uso del sistema sin acoplamiento a detalles de infraestructura.
Conecta el repositorio de contratos, el motor de supervivencia (KM + Cox Efron),
el evaluador de métricas prescriptivas y el publicador de Data Marts para Tableau.
"""

import time
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from src.domain.protocols import ContractRepository, ExtractPublisher
from src.domain.survival.km import KaplanMeierEstimator
from src.domain.survival.cox import CoxPHModel
from src.domain.survival.metrics import (
    compute_risk_band,
    compute_ecrl,
    compute_arr_at_risk,
    prescribe_action,
)
from src.infrastructure.duckdb_repo import DuckDBContractRepository
from src.infrastructure.tableau_exporter import TableauExporter


class ContractSurvivalPipeline:
    """Orquestador de alto nivel para el análisis de supervivencia y prescripción de contratos."""

    def __init__(
        self,
        repository: Optional[ContractRepository] = None,
        exporter: Optional[ExtractPublisher] = None,
    ):
        self.repository = repository or DuckDBContractRepository()
        self.exporter = exporter or TableauExporter()
        self.global_km = KaplanMeierEstimator()
        self.cox_model = CoxPHModel(penalizer=0.01, ties="efron")

    def run(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Ejecuta el pipeline analítico end-to-end:
            1. Ingesta desde el Lakehouse
            2. Ajuste de Kaplan-Meier (Global y por Segmentos)
            3. Ajuste de Cox PH con Efron y extracción de Hazard Scores
            4. Cálculo de ECRL, ARR en Riesgo y Runbooks Prescriptivos
            5. Persistencia en DuckDB y exportación Dual Mart para Tableau
        """
        start_time = time.time()

        # 1. Cargar datos desde el Lakehouse
        df = self.repository.read_contracts(limit=limit)
        if df.empty:
            raise ValueError("No se encontraron contratos en el repositorio para procesar.")

        n_records = len(df)

        # 2. Ajuste del modelo Kaplan-Meier Global
        durations = df["duration_days"].to_numpy()
        events = df["event_cancelled"].to_numpy()
        self.global_km.fit(durations, events)

        # 3. Ajuste de Kaplan-Meier por Segmentos para Tableau
        curve_records: List[Dict[str, Any]] = []

        # Curva Global
        for pt in self.global_km.to_survival_points(max_points=100):
            curve_records.append({
                "segment_type": "GLOBAL",
                "segment_value": "ALL_PORTFOLIO",
                "time_days": pt.time_days,
                "survival_probability": pt.survival_probability,
                "lower_ci": pt.lower_ci,
                "upper_ci": pt.upper_ci,
                "at_risk_count": pt.at_risk_count,
                "event_count": pt.event_count,
            })

        # Curvas por Contract Tier
        for tier_val in df["contract_tier"].unique():
            tier_mask = df["contract_tier"] == tier_val
            sub_km = KaplanMeierEstimator().fit(
                df.loc[tier_mask, "duration_days"].to_numpy(),
                df.loc[tier_mask, "event_cancelled"].to_numpy(),
            )
            for pt in sub_km.to_survival_points(max_points=50):
                curve_records.append({
                    "segment_type": "TIER",
                    "segment_value": str(tier_val),
                    "time_days": pt.time_days,
                    "survival_probability": pt.survival_probability,
                    "lower_ci": pt.lower_ci,
                    "upper_ci": pt.upper_ci,
                    "at_risk_count": pt.at_risk_count,
                    "event_count": pt.event_count,
                })

        # Curvas por Service Line
        for srv_val in df["service_line"].unique():
            srv_mask = df["service_line"] == srv_val
            sub_km = KaplanMeierEstimator().fit(
                df.loc[srv_mask, "duration_days"].to_numpy(),
                df.loc[srv_mask, "event_cancelled"].to_numpy(),
            )
            for pt in sub_km.to_survival_points(max_points=50):
                curve_records.append({
                    "segment_type": "SERVICE_LINE",
                    "segment_value": str(srv_val),
                    "time_days": pt.time_days,
                    "survival_probability": pt.survival_probability,
                    "lower_ci": pt.lower_ci,
                    "upper_ci": pt.upper_ci,
                    "at_risk_count": pt.at_risk_count,
                    "event_count": pt.event_count,
                })

        df_survival_curve_mart = pd.DataFrame(curve_records)

        # 4. Ajuste del modelo de Cox PH con empates Efron
        self.cox_model.fit(df)
        hazard_scores = self.cox_model.predict_hazard_score(df)

        # Probabilidades condicionales a horizontes clave
        surv_90d = self.cox_model.predict_survival_at_horizon(df, horizon_days=90.0)
        surv_180d = self.cox_model.predict_survival_at_horizon(df, horizon_days=180.0)
        surv_365d = self.cox_model.predict_survival_at_horizon(df, horizon_days=365.0)

        # 5. Calcular métricas financieras y prescriptivas por contrato
        risk_bands: List[str] = []
        ecrl_list: List[float] = []
        arr_at_risk_180d_list: List[float] = []
        action_codes: List[str] = []
        primary_drivers: List[str] = []
        playbooks: List[str] = []
        stakeholders: List[str] = []

        for i, row in df.iterrows():
            hs = float(hazard_scores[i])
            risk_band = compute_risk_band(hs)
            risk_bands.append(risk_band.value)

            # ECRL en horizonte 180 días
            age = float(row["duration_days"])
            ecrl = compute_ecrl(self.global_km.predict_survival, current_age_days=age, horizon_days=180.0)
            ecrl_list.append(round(ecrl, 1))

            # ARR en riesgo
            arr_val = float(row["arr"])
            cur_surv = self.global_km.predict_survival(age)
            horiz_surv = self.global_km.predict_survival(age + 180.0)
            arr_risk = compute_arr_at_risk(arr_val, cur_surv, horiz_surv)
            arr_at_risk_180d_list.append(arr_risk)

            # Prescripción operativa
            act = prescribe_action(row["contract_id"], hs, row.to_dict(), arr_val)
            action_codes.append(act.action_code)
            primary_drivers.append(act.primary_driver)
            playbooks.append(act.recommended_playbook)
            stakeholders.append(act.target_stakeholder)

        # Ensamblar Data Mart de Riesgo
        df_risk_mart = df.copy()
        df_risk_mart["hazard_score"] = np.round(hazard_scores, 4)
        df_risk_mart["risk_band"] = risk_bands
        df_risk_mart["survival_90d"] = np.round(surv_90d, 4)
        df_risk_mart["survival_180d"] = np.round(surv_180d, 4)
        df_risk_mart["survival_365d"] = np.round(surv_365d, 4)
        df_risk_mart["ecrl_days"] = ecrl_list
        df_risk_mart["gross_arr_at_risk_180d"] = arr_at_risk_180d_list
        df_risk_mart["action_code"] = action_codes
        df_risk_mart["primary_driver"] = primary_drivers
        df_risk_mart["recommended_playbook"] = playbooks
        df_risk_mart["target_stakeholder"] = stakeholders

        # 6. Persistir en Lakehouse DuckDB y exportar para Tableau
        self.repository.write_marts(df_risk_mart, df_survival_curve_mart)
        export_stats = self.exporter.export_dual_marts(df_risk_mart, df_survival_curve_mart)

        elapsed = time.time() - start_time

        # 7. Resumen Ejecutivo de la Ejecución
        total_arr = float(df_risk_mart["arr"].sum())
        arr_at_risk = float(df_risk_mart["gross_arr_at_risk_180d"].sum())
        critical_count = int((df_risk_mart["risk_band"] == "CRITICAL").sum())
        high_count = int((df_risk_mart["risk_band"] == "HIGH").sum())

        summary = {
            "status": "SUCCESS",
            "execution_time_seconds": round(elapsed, 2),
            "contracts_processed": n_records,
            "total_monitored_arr_usd": total_arr,
            "total_arr_at_risk_180d_usd": arr_at_risk,
            "arr_at_risk_percentage": round((arr_at_risk / total_arr) * 100, 2),
            "critical_risk_contracts": critical_count,
            "high_risk_contracts": high_count,
            "cox_concordance_index": round(self.cox_model.concordance_index_, 4),
            "tableau_exports": export_stats,
        }

        return summary
