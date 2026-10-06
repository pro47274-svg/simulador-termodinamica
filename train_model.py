"""
Entrenamiento del MLP multi-salida.

Uso:
    python train_model.py

Genera:
    models/mlp_thermodynamics.joblib
    models/metrics.txt
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATASET = Path("data/dataset.csv")
MODEL_DIR = Path("models")
MODEL_FILE = MODEL_DIR / "mlp_thermodynamics.joblib"
METRICS_FILE = MODEL_DIR / "metrics.txt"

FEATURES = [
    "P1", "V1", "T1", "V2", "gamma", "Cv",
    "process_adiabatico",
    "process_isobarico",
    "process_isocorico",
    "process_isotermico",
]

TARGETS = ["P2", "T2", "delta_U", "W", "Q"]


def train():
    if not DATASET.exists():
        raise FileNotFoundError(
            "No existe data/dataset.csv. Ejecuta primero: python generate_dataset.py"
        )

    df = pd.read_csv(DATASET)

    missing = [c for c in FEATURES + TARGETS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en el dataset: {missing}")

    X = df[FEATURES].astype(float)
    y = df[TARGETS].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # Pipeline: normalización de entradas + MLP.
    model = Pipeline([
        ("x_scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            max_iter=800,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=30,
            random_state=42,
        )),
    ])

    # Escalamos también las salidas porque P2 y las magnitudes energéticas
    # pueden tener escalas muy diferentes.
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)

    model.fit(X_train, y_train_scaled)

    y_pred_scaled = model.predict(X_test)
    y_pred = y_scaler.inverse_transform(y_pred_scaled)

    metrics = []
    metrics.append("MÉTRICAS DEL MODELO MLP\n")
    metrics.append(f"Muestras totales: {len(df)}")
    metrics.append(f"Entrenamiento: {len(X_train)}")
    metrics.append(f"Prueba: {len(X_test)}\n")

    for i, target in enumerate(TARGETS):
        mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], y_pred[:, i]))
        r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
        metrics.append(
            f"{target:10s} | MAE={mae:.6g} | RMSE={rmse:.6g} | R²={r2:.6f}"
        )

    # Verificación aproximada de conservación de energía:
    # Q = ΔU + W
    q_idx = TARGETS.index("Q")
    du_idx = TARGETS.index("delta_U")
    w_idx = TARGETS.index("W")
    energy_residual = np.abs(
        y_pred[:, q_idx] - (y_pred[:, du_idx] + y_pred[:, w_idx])
    )

    metrics.append("\nCONSERVACIÓN DE ENERGÍA")
    metrics.append(
        f"Residual medio |Q-(ΔU+W)| = {energy_residual.mean():.6g} J"
    )
    metrics.append(
        f"Residual máximo |Q-(ΔU+W)| = {energy_residual.max():.6g} J"
    )

    MODEL_DIR.mkdir(exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "y_scaler": y_scaler,
            "features": FEATURES,
            "targets": TARGETS,
        },
        MODEL_FILE,
    )

    METRICS_FILE.write_text("\n".join(metrics), encoding="utf-8")

    print("\n".join(metrics))
    print(f"\nModelo guardado en: {MODEL_FILE}")


if __name__ == "__main__":
    train()
