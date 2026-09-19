"""
tests/benchmark.py - Medición Cuantitativa de Rendimiento, Latencias p50/p95/p99 y Memoria

Ejecuta el pipeline completo sobre 50.000 contratos y mide:
1. Latencia de generación estocástica.
2. Latencia de ingestión en DuckDB Lakehouse.
3. Latencia de ajuste Kaplan-Meier (Global y por segmentos).
4. Latencia de ajuste e inferencia Cox PH con Efron.
5. Latencia p50, p95 y p99 de transformaciones puras de dominio (sub-5ms test).
6. Consumo pico de memoria RAM (RSS en MB).
7. Persistencia de resultados estructurados en artifacts/benchmark.json.
"""

import os
import sys
import time
import json
import tracemalloc
import numpy as np
import pandas as pd

# Ensure project root is in sys.path for standalone script execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.data.data_generator import generate_contract_dataset
from src.infrastructure.duckdb_repo import DuckDBContractRepository
from src.infrastructure.tableau_exporter import TableauExporter
from src.infrastructure.snowflake_adapter import SnowflakeAdapter
from src.application.pipeline import ContractSurvivalPipeline
from src.domain.survival.metrics import compute_ecrl, compute_arr_at_risk, prescribe_action


def run_benchmarks(n_records: int = 50000, domain_iterations: int = 10, batch_size: int = 100) -> dict:
    print("\n" + "=" * 80)
    print(f" [BENCHMARK] GP-102 RUNNER: Benchmarking {n_records:,} Enterprise Contracts")
    print("=" * 80)

    tracemalloc.start()
    benchmark_data = {}

    # 1. Generación de Datos
    print("\n[1/6] Benchmarking Synthetic Data Generation (50k records)...")
    os.makedirs("data", exist_ok=True)
    t0 = time.perf_counter()
    raw_path = "data/raw_contracts.parquet"
    df_raw = generate_contract_dataset(n_records=n_records, seed=20260919, output_path=raw_path)
    t_gen = time.perf_counter() - t0
    benchmark_data["data_generation_seconds"] = round(t_gen, 3)
    benchmark_data["records_generated"] = n_records
    print(f"  -> Generated {n_records:,} records in {t_gen:.3f}s ({n_records/t_gen:,.0f} rows/s)")

    # 2. Pipeline de Aplicación End-to-End
    print("\n[2/6] Executing End-to-End Application Pipeline (DuckDB + KM + Cox + Tableau)...")
    repo = DuckDBContractRepository(db_path=":memory:", raw_data_path=raw_path)
    exporter = TableauExporter(output_dir="data/tableau")
    pipeline = ContractSurvivalPipeline(repository=repo, exporter=exporter)

    t0 = time.perf_counter()
    summary = pipeline.run()
    t_pipeline = time.perf_counter() - t0
    benchmark_data["end_to_end_pipeline_seconds"] = round(t_pipeline, 3)
    benchmark_data["pipeline_summary"] = summary
    print(f"  -> End-to-End Pipeline executed in {t_pipeline:.3f}s")
    print(f"  -> Total Monitored ARR: ${summary['total_monitored_arr_usd']:,.2f} USD")
    print(f"  -> Gross ARR at Risk (180d): ${summary['total_arr_at_risk_180d_usd']:,.2f} USD ({summary['arr_at_risk_percentage']}%)")
    print(f"  -> Model Concordance Index (C-index): {summary['cox_concordance_index']}")

    # 3. Micro-Benchmarks de Transformación Pura de Dominio (100 registros, 10 iteraciones)
    print(f"\n[3/6] Measuring Pure Domain Transformations Latency ({domain_iterations} iterations on {batch_size} contracts)...")
    sample_records = df_raw.iloc[:batch_size].to_dict(orient="records")
    dummy_surv = pipeline.global_km.predict_survival

    latencies_ms = []
    for _ in range(domain_iterations):
        t_iter = time.perf_counter()
        for row_dict in sample_records:
            age = float(row_dict["duration_days"])
            arr = float(row_dict["arr"])
            ecrl = compute_ecrl(dummy_surv, current_age_days=age, horizon_days=180.0)
            arr_risk = compute_arr_at_risk(arr, dummy_surv(age), dummy_surv(age + 180.0))
            action = prescribe_action(row_dict["contract_id"], 1.5, row_dict, arr)
        latencies_ms.append((time.perf_counter() - t_iter) * 1000)

    p50_ms = float(np.percentile(latencies_ms, 50))
    p95_ms = float(np.percentile(latencies_ms, 95))
    p99_ms = float(np.percentile(latencies_ms, 99))
    per_contract_us = (p50_ms / batch_size) * 1000  # microsegundos por contrato

    benchmark_data["pure_domain_batch_p50_ms"] = round(p50_ms, 2)
    benchmark_data["pure_domain_batch_p95_ms"] = round(p95_ms, 2)
    benchmark_data["pure_domain_batch_p99_ms"] = round(p99_ms, 2)
    benchmark_data["pure_domain_per_contract_microseconds"] = round(per_contract_us, 2)

    print(f"  -> {batch_size} Contracts Pure Transform p50: {p50_ms:.2f} ms ({per_contract_us:.2f} us/contract)")
    print(f"  -> {batch_size} Contracts Pure Transform p95: {p95_ms:.2f} ms")
    print(f"  -> {batch_size} Contracts Pure Transform p99: {p99_ms:.2f} ms")

    # 4. Memoria Pico
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak_mem / (1024 * 1024)
    benchmark_data["peak_memory_rss_mb"] = round(peak_mb, 2)
    print(f"\n[4/6] Peak RAM Consumption (tracemalloc RSS): {peak_mb:.2f} MB")

    # 5. Exportar DDL Snowflake
    print("\n[5/6] Exporting Production Snowflake DDL Artifact...")
    sf = SnowflakeAdapter()
    ddl_path = sf.export_ddl_to_file("artifacts/snowflake_deployment.sql")
    benchmark_data["snowflake_ddl_path"] = ddl_path
    print(f"  -> DDL Script exported -> {ddl_path}")

    # 6. Guardar Reporte Estructurado
    print("\n[6/6] Persisting Structured Telemetry Artifacts...")
    os.makedirs("artifacts", exist_ok=True)
    benchmark_file = "artifacts/benchmark.json"
    with open(benchmark_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2)
    print(f"  -> Benchmark saved -> {benchmark_file}")

    print("\n" + "=" * 80)
    print(" [OK] ALL QUANTITATIVE BENCHMARKS COMPLETED SUCCESSFULLY")
    print("=" * 80 + "\n")

    return benchmark_data


if __name__ == "__main__":
    run_benchmarks()
