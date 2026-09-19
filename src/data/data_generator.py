"""
src/data/data_generator.py - Generador de Física Estocástica de Contratos IT (Inetum)

Genera 50.000+ contratos B2B con exposición temporal, efectos aleatorios por cliente,
distribución latente de Weibull y censura administrativa a la derecha.

Matemática de Generación:
    T_i ~ Weibull(k_segment, lambda_0 * exp(-beta^T * X_i - u_customer - u_service))
    C_i ~ Uniform(c_min, c_max)   [Censura Administrativa]
    Y_i = min(T_i, C_i)           [Duración Observada]
    delta_i = 1[T_i <= C_i]       [Indicador de Evento de Rescisión]
"""

import os
import time
import argparse
from typing import Optional, Tuple
import numpy as np
import pandas as pd


def generate_contract_dataset(
    n_records: int = 50000,
    seed: int = 20260919,
    output_path: Optional[str] = "data/raw_contracts.parquet",
) -> pd.DataFrame:
    """
    Genera el dataset calibrado de telemetría de 50.000+ contratos B2B para Inetum.
    
    Args:
        n_records: Número total de contratos a simular.
        seed: Semilla determinista para reproducibilidad estricta.
        output_path: Ruta de guardado en Parquet (opcional).
        
    Returns:
        pd.DataFrame con el esquema canónico dimensional de contratos.
    """
    start_time = time.time()
    rng = np.random.default_rng(seed)

    # 1. Dimensiones de Cliente y Contrato
    n_customers = max(1000, n_records // 10)
    customer_ids = [f"CUST-{i:06d}" for i in range(1, n_customers + 1)]
    assigned_customers = rng.choice(customer_ids, size=n_records)

    # Efecto aleatorio por cliente (heterogeneidad latente no observada)
    customer_re = {cid: rng.normal(0.0, 0.15) for cid in customer_ids}
    customer_re_arr = np.array([customer_re[cid] for cid in assigned_customers])

    tiers = ["STRATEGIC", "ENTERPRISE", "MID_MARKET", "STANDARD"]
    tier_probs = [0.10, 0.25, 0.40, 0.25]
    contract_tiers = rng.choice(tiers, size=n_records, p=tier_probs)

    services = [
        "MANAGED_SERVICES",
        "CLOUD_INFRASTRUCTURE",
        "CYBERSECURITY",
        "DATA_AI",
        "APPLICATION_DEV",
    ]
    service_probs = [0.30, 0.25, 0.15, 0.20, 0.10]
    service_lines = rng.choice(services, size=n_records, p=service_probs)

    # Efecto aleatorio por línea de servicio
    service_re_map = {
        "MANAGED_SERVICES": 0.08,
        "CLOUD_INFRASTRUCTURE": -0.05,
        "CYBERSECURITY": -0.10,
        "DATA_AI": 0.05,
        "APPLICATION_DEV": 0.12,
    }
    service_re_arr = np.array([service_re_map[s] for s in service_lines])

    # 2. ARR Modelado por Tier (Calibrado a la escala de Inetum: ~2.5B EUR/USD total)
    arr_means = {
        "STRATEGIC": (180000, 35000),
        "ENTERPRISE": (80000, 15000),
        "MID_MARKET": (32000, 7000),
        "STANDARD": (14000, 3000),
    }
    arr_values = np.zeros(n_records)
    for t in tiers:
        mask = contract_tiers == t
        mean, std = arr_means[t]
        arr_values[mask] = np.clip(rng.normal(mean, std, size=np.sum(mask)), 15000, 2500000)

    # 3. Covariables Operativas de Consultoría IT (Correlacionadas)
    # SLA Breach Rate (mayor propensión en Standard y Mid-Market)
    sla_shape_a = np.where(np.isin(contract_tiers, ["STANDARD", "MID_MARKET"]), 2.2, 1.4)
    sla_breach_rate = np.clip(rng.beta(sla_shape_a, 12.0, size=n_records), 0.0, 0.45)

    # Engineer Turnover Ratio (rotación de personal en el proyecto)
    engineer_turnover_ratio = np.clip(rng.beta(1.4, 4.5, size=n_records), 0.0, 0.65)

    # Budget Burn Variance (desviación presupuestaria: negativa = sub-ejecución, positiva = sobrecosto)
    budget_burn_variance = np.clip(rng.normal(0.06, 0.12, size=n_records), -0.25, 0.55)

    # Ticket Escalation Velocity (velocidad mensual de tickets escalados a dirección)
    escalation_velocity = rng.negative_binomial(n=4, p=0.45, size=n_records) + (sla_breach_rate * 8.0)
    ticket_escalation_velocity = np.round(escalation_velocity, 1)

    # Tech Stack Alignment Score (1.0 = stack nativo Inetum, 0.3 = legado complejo)
    tech_stack_alignment = np.clip(rng.beta(7.5, 2.2, size=n_records) - (budget_burn_variance * 0.2), 0.25, 1.0)
    tech_stack_alignment_score = np.round(tech_stack_alignment, 3)

    # Payment Delay Days (señal financiera temprana de fricción operativa)
    payment_delay_days = np.round(rng.exponential(scale=9.0, size=n_records) + (sla_breach_rate * 25.0), 1)

    # Renewal Count (historial previo de renovaciones exitosas)
    renewal_count = np.clip(rng.poisson(lam=1.3, size=n_records), 0, 7)

    # 4. Motor de Supervivencia Latente Weibull
    # Coeficientes verdaderos beta conocidos para validación estadística posterior
    linear_predictor = (
        0.95 * sla_breach_rate
        + 0.72 * engineer_turnover_ratio
        + 0.54 * budget_burn_variance
        + 0.08 * ticket_escalation_velocity
        - 0.88 * tech_stack_alignment_score
        + 0.015 * payment_delay_days
        - 0.22 * renewal_count
        + customer_re_arr
        + service_re_arr
    )

    # Parámetro de forma Weibull por Tier (k > 1 indica riesgo acumulativo por desgaste temporal)
    k_shape_map = {"STRATEGIC": 1.9, "ENTERPRISE": 1.6, "MID_MARKET": 1.3, "STANDARD": 1.05}
    k_shapes = np.array([k_shape_map[t] for t in contract_tiers])

    # Escala base lambda_0 (aproximadamente 1200 días / ~3.3 años de vida media nominal)
    lambda_0 = 1200.0

    # Generación del tiempo latente de evento T a partir de Uniforme U
    u = rng.uniform(1e-6, 1.0 - 1e-6, size=n_records)
    latent_event_time = lambda_0 * ((-np.log(u)) / np.exp(linear_predictor)) ** (1.0 / k_shapes)
    latent_event_time = np.clip(latent_event_time, 1.0, 3650.0)

    # 5. Censura Administrativa a la Derecha (C)
    # Simula la ventana de corte de observación del sistema Inetum (1 a 4 años)
    censor_time = rng.uniform(365.0, 1460.0, size=n_records)

    # Tiempo observado y variable binaria delta de evento
    observed_duration = np.minimum(latent_event_time, censor_time)
    event_cancelled = (latent_event_time <= censor_time).astype(int)

    # 6. Construcción del DataFrame Canónico
    contract_ids = [f"CNT-{202500000 + i}" for i in range(1, n_records + 1)]

    df = pd.DataFrame({
        "contract_id": contract_ids,
        "customer_id": assigned_customers,
        "contract_tier": contract_tiers,
        "service_line": service_lines,
        "duration_days": np.round(observed_duration, 1),
        "event_cancelled": event_cancelled,
        "arr": np.round(arr_values, 2),
        "sla_breach_rate": np.round(sla_breach_rate, 4),
        "engineer_turnover_ratio": np.round(engineer_turnover_ratio, 4),
        "budget_burn_variance": np.round(budget_burn_variance, 4),
        "ticket_escalation_velocity": ticket_escalation_velocity,
        "tech_stack_alignment_score": tech_stack_alignment_score,
        "payment_delay_days": payment_delay_days,
        "renewal_count": renewal_count,
    })

    elapsed = time.time() - start_time

    # Métricas diagnósticas
    censoring_pct = (1.0 - df["event_cancelled"].mean()) * 100
    events_count = int(df["event_cancelled"].sum())
    median_duration = float(df["duration_days"].median())

    print(f"================================================================================")
    print(f" [Data Generator] Inetum Contract Portfolio: {n_records:,} records in {elapsed:.2f}s")
    print(f"  - Censoring Rate:      {censoring_pct:.1f}% ({n_records - events_count:,} contracts active)")
    print(f"  - Observed Churn Events:{events_count:,} ({df['event_cancelled'].mean()*100:.1f}%)")
    print(f"  - Median Duration:     {median_duration:.1f} days")
    print(f"  - Total Monitored ARR: ${df['arr'].sum():,.2f} USD")
    print(f"================================================================================")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_parquet(output_path, index=False)
        print(f" [Data Generator] Artifact persisted -> {output_path}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generador estocástico de contratos para Inetum.")
    parser.add_argument("--records", type=int, default=50000, help="Número de registros (default 50k)")
    parser.add_argument("--seed", type=int, default=20260919, help="Semilla aleatoria")
    parser.add_argument("--output", type=str, default="data/raw_contracts.parquet", help="Ruta de guardado Parquet")
    args = parser.parse_args()

    generate_contract_dataset(args.records, args.seed, args.output)
