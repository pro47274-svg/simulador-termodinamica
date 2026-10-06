"""
Generación del dataset sintético para entrenar el modelo de IA.

Uso:
    python generate_dataset.py
    python generate_dataset.py --samples 20000 --output data/dataset.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from physics import R, simulate_process


PROCESS_NAMES = ["isobarico", "isocorico", "isotermico", "adiabatico"]


def generate_dataset(samples: int = 10000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    for _ in range(samples):
        process = rng.choice(PROCESS_NAMES)

        # Valores físicamente razonables para 1 mol de gas ideal.
        V1 = rng.uniform(0.005, 0.100)       # m³
        T1 = rng.uniform(250.0, 600.0)       # K
        P1 = R * T1 / V1                     # Pa, de PV=nRT con n=1
        gamma = rng.uniform(1.30, 1.67)

        # Para mantener consistencia termodinámica, Cv se obtiene de:
        # gamma = Cp/Cv y Cp-Cv=R  ->  Cv=R/(gamma-1)
        Cv = R / (gamma - 1.0)

        if process == "isocorico":
            V2 = V1
        else:
            # Cambios de volumen moderados, evitando V2=V1 con demasiada frecuencia.
            ratio = rng.uniform(0.55, 1.80)
            V2 = V1 * ratio

        result = simulate_process(
            process=process,
            P1=P1,
            V1=V1,
            T1=T1,
            V2=V2,
            n=1.0,
            gamma=gamma,
            Cv=Cv,
        )

        rows.append({
            "P1": result.P1,
            "V1": result.V1,
            "T1": result.T1,
            "V2": result.V2,
            "gamma": result.gamma,
            "Cv": result.Cv,
            "process": result.process,
            "P2": result.P2,
            "T2": result.T2,
            "delta_U": result.delta_u,
            "W": result.W,
            "Q": result.Q,
        })

    df = pd.DataFrame(rows)

    # One-hot encoding del tipo de proceso, tal como pide el proyecto.
    process_dummies = pd.get_dummies(df["process"], prefix="process", dtype=int)
    df = pd.concat([df.drop(columns=["process"]), process_dummies], axis=1)

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/dataset.csv")
    args = parser.parse_args()

    if args.samples < 100:
        raise ValueError("Usa al menos 100 muestras.")

    df = generate_dataset(args.samples, args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    print(f"Dataset creado: {output}")
    print(f"Muestras: {len(df)}")
    print(f"Columnas: {', '.join(df.columns)}")
    print("\nPrimeras filas:")
    print(df.head())


if __name__ == "__main__":
    main()
