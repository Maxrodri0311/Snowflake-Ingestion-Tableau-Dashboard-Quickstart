"""
tests/unit/test_phase3_lakehouse.py - Pruebas Unitarias de Lakehouse, Snowflake y Tableau (Fase 3)

Verifica:
1. Repositorio DuckDB: linaje dbt (Staging -> Int -> Marts) y dimensiones Kimball.
2. Adaptador Snowflake: validación sintáctica del DDL de producción y clustering keys.
3. Tableau Exporter: generación atómica de Dual Marts (mart_contract_risk y mart_tableau_survival_curve).
4. Pipeline de Aplicación: ejecución end-to-end con métricas financieras y runbooks prescriptivos.
"""

import os
import pytest
import pandas as pd
import duckdb

from src.infrastructure.duckdb_repo import DuckDBContractRepository
from src.infrastructure.snowflake_adapter import SnowflakeAdapter, generate_snowflake_ddl
from src.infrastructure.tableau_exporter import TableauExporter
from src.application.pipeline import ContractSurvivalPipeline
from src.data.data_generator import generate_contract_dataset


@pytest.fixture(scope="module")
def prepared_parquet(tmp_path_factory):
    """Genera un archivo Parquet temporal de 1,500 registros para testing de integración rápida."""
    fn = str(tmp_path_factory.mktemp("data") / "test_contracts.parquet")
    generate_contract_dataset(n_records=1500, seed=999, output_path=fn)
    return fn


# ==============================================================================
# 1. PRUEBAS DE LAKEHOUSE DUCKDB (PATRÓN DBT)
# ==============================================================================

def test_duckdb_repository_lineage(prepared_parquet):
    """Verifica creación de vistas staging, intermediate y lectura de contratos."""
    repo = DuckDBContractRepository(db_path=":memory:", raw_data_path=prepared_parquet)

    # 1. Lectura desde la capa intermediate
    df = repo.read_contracts(limit=500)
    assert len(df) == 500
    assert "is_multiyear_tenure" in df.columns
    assert "operational_friction_index" in df.columns

    # 2. Resumen del Lakehouse
    summary = repo.get_lakehouse_summary()
    assert summary["total_contracts"] == 1500
    assert summary["total_portfolio_arr"] > 0
    assert summary["churn_rate_pct"] > 0


def test_duckdb_write_marts(prepared_parquet):
    """Verifica persistencia de Data Marts y dimensiones conformadas en DuckDB."""
    repo = DuckDBContractRepository(db_path=":memory:", raw_data_path=prepared_parquet)
    df = repo.read_contracts(limit=200)

    # Marts de prueba mínimos
    df_risk = df.copy()
    df_risk["hazard_score"] = 1.0
    df_risk["risk_band"] = "MEDIUM"

    df_curve = pd.DataFrame({
        "segment_type": ["GLOBAL"],
        "segment_value": ["ALL"],
        "time_days": [100.0],
        "survival_probability": [0.85],
    })

    repo.write_marts(df_risk, df_curve)

    # Verificar que las tablas existen y son consultables en SQL
    res_risk = repo.query("SELECT COUNT(*) FROM mart_contract_risk").iloc[0, 0]
    res_dim_cust = repo.query("SELECT COUNT(*) FROM dim_customer").iloc[0, 0]
    res_dim_cnt = repo.query("SELECT COUNT(*) FROM dim_contract").iloc[0, 0]

    assert res_risk == 200
    assert res_dim_cust > 0
    assert res_dim_cnt == 200


# ==============================================================================
# 2. PRUEBAS DEL ADAPTADOR SNOWFLAKE
# ==============================================================================

def test_snowflake_ddl_generation(tmp_path):
    """Verifica que el script DDL de Snowflake contiene las especificaciones Enterprise requeridas."""
    ddl = generate_snowflake_ddl()

    assert "CLUSTER BY (service_line, contract_tier)" in ddl
    assert "RAW_CONTRACT_TELEMETRY" in ddl
    assert "payload VARIANT" in ddl
    assert "FCT_CONTRACT_LIFECYCLE" in ddl
    assert "MART_CONTRACT_RISK" in ddl
    assert "MART_TABLEAU_SURVIVAL_CURVE" in ddl

    # Exportar archivo DDL
    adapter = SnowflakeAdapter()
    target_file = str(tmp_path / "test_snowflake.sql")
    out_path = adapter.export_ddl_to_file(target_file)
    assert os.path.exists(out_path)


# ==============================================================================
# 3. PRUEBAS DE TABLEAU DUAL MARTS EXPORTER
# ==============================================================================

def test_tableau_exporter(tmp_path):
    """Verifica exportación atómica de ambos Data Marts con columnas esperadas."""
    output_dir = str(tmp_path / "tableau")
    exporter = TableauExporter(output_dir=output_dir)

    df_risk = pd.DataFrame({
        "contract_id": ["CNT-01", "CNT-02"],
        "arr": [150000.555, 80000.333],
        "hazard_score": [1.45678, 0.82341],
        "risk_band": ["MEDIUM", "LOW"],
    })

    df_curve = pd.DataFrame({
        "segment_type": ["GLOBAL", "GLOBAL"],
        "time_days": [0.0, 180.0],
        "survival_probability": [1.0, 0.88],
    })

    stats = exporter.export_dual_marts(df_risk, df_curve)

    assert stats["total_risk_rows"] == 2
    assert stats["total_curve_rows"] == 2
    assert os.path.exists(stats["risk_mart_path"])
    assert os.path.exists(stats["curve_mart_path"])

    # Validar redondeo para Tableau
    exported_risk = pd.read_csv(stats["risk_mart_path"])
    assert exported_risk.loc[0, "arr"] == 150000.56
    assert exported_risk.loc[0, "hazard_score"] == 1.4568


# ==============================================================================
# 4. PRUEBA INTEGRAL DEL PIPELINE DE APLICACIÓN
# ==============================================================================

def test_application_pipeline_execution(prepared_parquet, tmp_path):
    """Verifica ejecución end-to-end del pipeline de aplicación orquestado."""
    output_dir = str(tmp_path / "tableau_app")
    repo = DuckDBContractRepository(db_path=":memory:", raw_data_path=prepared_parquet)
    exporter = TableauExporter(output_dir=output_dir)

    pipeline = ContractSurvivalPipeline(repository=repo, exporter=exporter)
    summary = pipeline.run(limit=1000)

    assert summary["status"] == "SUCCESS"
    assert summary["contracts_processed"] == 1000
    assert summary["total_monitored_arr_usd"] > 0
    assert summary["total_arr_at_risk_180d_usd"] > 0
    assert summary["cox_concordance_index"] > 0.55
    assert os.path.exists(summary["tableau_exports"]["risk_mart_path"])
    assert os.path.exists(summary["tableau_exports"]["curve_mart_path"])
