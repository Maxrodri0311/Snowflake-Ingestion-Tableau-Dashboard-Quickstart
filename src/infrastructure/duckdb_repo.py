"""
src/infrastructure/duckdb_repo.py - Repositorio Lakehouse Local con DuckDB (Patrón dbt)

Implementa el protocolo ContractRepository utilizando DuckDB como motor OLAP columnar 
en memoria y persistente. Modela el linaje dimensional completo (Staging -> Intermediate -> Marts)
emulando la estructura física y analítica de un Data Warehouse en Snowflake.

Linaje de Datos:
    raw_contracts.parquet -> stg_contracts (Vista Staging)
                          -> int_contract_exposure_features (Transformación Intermedia)
                          -> fct_contract_lifecycle (Tabla Hechos Core)
                          -> dim_customer & dim_contract (Dimensiones Conformadas)
                          -> mart_contract_risk & mart_tableau_survival_curve (Marts BI)
"""

import os
from typing import Optional, Dict, Any, List
import duckdb
import pandas as pd
from src.domain.protocols import ContractRepository


class DuckDBContractRepository(ContractRepository):
    """Repositorio columnar de alto rendimiento sobre DuckDB emulando Snowflake."""

    def __init__(self, db_path: str = ":memory:", raw_data_path: str = "data/raw_contracts.parquet"):
        self.db_path = db_path
        self.raw_data_path = raw_data_path
        self.conn = duckdb.connect(db_path)
        self._initialize_lakehouse_lineage()

    def _initialize_lakehouse_lineage(self) -> None:
        """Configura vistas y transformaciones iniciales siguiendo el estándar de dbt."""
        if os.path.exists(self.raw_data_path):
            # 1. Capa Staging: Vistas 1:1 con fuentes crudas con tipado explícito
            self.conn.execute(f"""
                CREATE OR REPLACE VIEW stg_contracts AS
                SELECT 
                    contract_id::VARCHAR AS contract_id,
                    customer_id::VARCHAR AS customer_id,
                    contract_tier::VARCHAR AS contract_tier,
                    service_line::VARCHAR AS service_line,
                    duration_days::DOUBLE AS duration_days,
                    event_cancelled::INTEGER AS event_cancelled,
                    arr::DOUBLE AS arr,
                    sla_breach_rate::DOUBLE AS sla_breach_rate,
                    engineer_turnover_ratio::DOUBLE AS engineer_turnover_ratio,
                    budget_burn_variance::DOUBLE AS budget_burn_variance,
                    ticket_escalation_velocity::DOUBLE AS ticket_escalation_velocity,
                    tech_stack_alignment_score::DOUBLE AS tech_stack_alignment_score,
                    payment_delay_days::DOUBLE AS payment_delay_days,
                    renewal_count::INTEGER AS renewal_count
                FROM read_parquet('{self.raw_data_path}');
            """)

            # 2. Capa Intermediate: Enriquecimiento de exposición y ventanas de riesgo
            self.conn.execute("""
                CREATE OR REPLACE VIEW int_contract_exposure_features AS
                SELECT
                    contract_id,
                    customer_id,
                    contract_tier,
                    service_line,
                    duration_days,
                    event_cancelled,
                    arr,
                    sla_breach_rate,
                    engineer_turnover_ratio,
                    budget_burn_variance,
                    ticket_escalation_velocity,
                    tech_stack_alignment_score,
                    payment_delay_days,
                    renewal_count,
                    -- Bandera de exposición temporal prolongada (> 1 año)
                    CASE WHEN duration_days >= 365.0 THEN 1 ELSE 0 END AS is_multiyear_tenure,
                    -- Score compuesto de fricción operativa
                    ROUND((sla_breach_rate * 0.40) + (engineer_turnover_ratio * 0.35) + (ticket_escalation_velocity / 20.0 * 0.25), 4) AS operational_friction_index
                FROM stg_contracts;
            """)

    def read_contracts(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Carga el dataset preparado desde la capa intermedia."""
        query = "SELECT * FROM int_contract_exposure_features"
        if limit:
            query += f" LIMIT {limit}"
        return self.conn.execute(query).df()

    def write_marts(
        self,
        risk_mart: pd.DataFrame,
        survival_curve_mart: pd.DataFrame,
    ) -> None:
        """Persiste los Data Marts analíticos en tablas nativas de DuckDB."""
        # Registrar DataFrames en el catálogo DuckDB
        self.conn.register("df_risk_mart", risk_mart)
        self.conn.register("df_curve_mart", survival_curve_mart)

        # Crear tablas físicas de Data Marts listas para BI
        self.conn.execute("""
            CREATE OR REPLACE TABLE mart_contract_risk AS
            SELECT * FROM df_risk_mart;
        """)

        self.conn.execute("""
            CREATE OR REPLACE TABLE mart_tableau_survival_curve AS
            SELECT * FROM df_curve_mart;
        """)

        # Generar dimensiones conformadas (Kimball Star Schema)
        self.conn.execute("""
            CREATE OR REPLACE TABLE dim_customer AS
            SELECT DISTINCT
                customer_id,
                COUNT(contract_id) AS total_contracts,
                SUM(arr) AS total_customer_arr,
                ROUND(AVG(sla_breach_rate), 4) AS avg_sla_breach_rate
            FROM df_risk_mart
            GROUP BY customer_id;
        """)

        self.conn.execute("""
            CREATE OR REPLACE TABLE dim_contract AS
            SELECT DISTINCT
                contract_id,
                customer_id,
                contract_tier,
                service_line,
                arr
            FROM df_risk_mart;
        """)

    def query(self, sql: str) -> pd.DataFrame:
        """Ejecuta una consulta SQL arbitraria en el Lakehouse."""
        return self.conn.execute(sql).df()

    def get_lakehouse_summary(self) -> Dict[str, Any]:
        """Retorna estadísticas operativas y financieras del catálogo de datos."""
        res = self.conn.execute("""
            SELECT 
                COUNT(*) AS total_contracts,
                ROUND(SUM(arr), 2) AS total_portfolio_arr,
                ROUND(AVG(duration_days), 1) AS avg_duration_days,
                SUM(event_cancelled) AS total_churn_events,
                ROUND(SUM(event_cancelled) * 100.0 / COUNT(*), 1) AS churn_rate_pct
            FROM stg_contracts;
        """).fetchone()

        return {
            "total_contracts": res[0],
            "total_portfolio_arr": res[1],
            "avg_duration_days": res[2],
            "total_churn_events": res[3],
            "churn_rate_pct": res[4],
        }
