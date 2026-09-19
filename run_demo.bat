@echo off
setlocal enabledelayedexpansion

echo ===============================================================================
echo   INETUM - CONTRACT SURVIVAL AND LIFECYCLE ANALYTICS ENGINE
echo   Enterprise Production Case Study - Data Analyst / Analytics Engineer
echo ===============================================================================
echo.

echo [1/2] Running Full Pytest Suite (Unit, Integration and Mathematical Invariants)...
python -m pytest tests/ -v
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Pytest suite encountered failures.
    exit /b %ERRORLEVEL%
)

echo.
echo [2/2] Executing Quantitative Benchmarks and Full End-to-End Pipeline...
echo       Ingestion, Kaplan-Meier, Cox PH with Efron, Dual Tableau Marts and Profiling...
python tests/benchmark.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Benchmark execution encountered failures.
    exit /b %ERRORLEVEL%
)

echo.
echo ===============================================================================
echo   [SUCCESS] ALL SYSTEMS OPERATIONAL: 22 TESTS PASSED, PIPELINE EXECUTED
echo   - Telemetry Report:  artifacts/benchmark.json
echo   - Snowflake DDL:     artifacts/snowflake_deployment.sql
echo   - Tableau Risk Mart: data/tableau/mart_contract_risk.csv
echo   - Tableau Curves:    data/tableau/mart_tableau_survival_curve.csv
echo ===============================================================================
