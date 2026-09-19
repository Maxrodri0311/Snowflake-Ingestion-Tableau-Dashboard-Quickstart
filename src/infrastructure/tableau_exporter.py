"""
src/infrastructure/tableau_exporter.py - Exportador de Data Marts Dual para Tableau

Implementa el protocolo ExtractPublisher para generar los datasets optimizados 
que consumirá el dashboard ejecutivo en Tableau:
1. mart_contract_risk.csv: Tabla ancha a nivel contrato con ARR en riesgo y runbooks.
2. mart_tableau_survival_curve.csv: Coordenadas de curvas Kaplan-Meier por segmento con bandas Greenwood.
"""

import os
from typing import Dict, Any, List
import pandas as pd
from src.domain.protocols import ExtractPublisher


class TableauExporter(ExtractPublisher):
    """Exportador de Data Marts formateados específicamente para Tableau Desktop / Server."""

    def __init__(self, output_dir: str = "data/tableau"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def publish(self, df: pd.DataFrame, target_filename: str) -> str:
        """Exporta un DataFrame a formato CSV optimizado para Tableau."""
        target_path = os.path.join(self.output_dir, target_filename)
        df.to_csv(target_path, index=False, encoding="utf-8")
        return target_path

    def export_dual_marts(
        self,
        risk_mart: pd.DataFrame,
        survival_curve_mart: pd.DataFrame,
    ) -> Dict[str, str]:
        """
        Exporta simultáneamente los dos Data Marts canónicos:
            - mart_contract_risk.csv (Métricas y cola prescriptiva por contrato)
            - mart_tableau_survival_curve.csv (Coordenadas de supervivencia por segmento)
        """
        # Formatear el mart de riesgo para Tableau
        formatted_risk = risk_mart.copy()
        float_cols_2dec = ["arr", "gross_arr_at_risk_180d"]
        float_cols_4dec = ["hazard_score", "survival_90d", "survival_180d", "survival_365d"]

        for col in float_cols_2dec:
            if col in formatted_risk.columns:
                formatted_risk[col] = formatted_risk[col].round(2)
        for col in float_cols_4dec:
            if col in formatted_risk.columns:
                formatted_risk[col] = formatted_risk[col].round(4)

        risk_path = self.publish(formatted_risk, "mart_contract_risk.csv")
        curve_path = self.publish(survival_curve_mart, "mart_tableau_survival_curve.csv")

        return {
            "risk_mart_path": risk_path,
            "curve_mart_path": curve_path,
            "total_risk_rows": len(formatted_risk),
            "total_curve_rows": len(survival_curve_mart),
        }
