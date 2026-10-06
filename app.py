"""
Dashboard interactivo con Streamlit.

Ejecutar:
    streamlit run app.py

El dashboard calcula la solución analítica y la compara con la predicción
del MLP entrenado.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from physics import R, p_v_curve, simulate_process


MODEL_FILE = Path("mlp_thermodynamics.joblib")

st.set_page_config(
    page_title="Simulador Termodinámico + IA",
    page_icon="🌡️",
    layout="wide",
)

st.title("🌡️ Simulador Termodinámico con Inteligencia Artificial")
st.caption(
    "Comparación entre la solución analítica de cuatro procesos ideales "
    "y la predicción de una red neuronal MLP."
)

if not MODEL_FILE.exists():
    st.error(
        "No se encontró el modelo entrenado. Ejecuta primero:\n\n"
        "1. `python generate_dataset.py`\n"
        "2. `python train_model.py`"
    )
    st.stop()

bundle = joblib.load(MODEL_FILE)
model = bundle["model"]
y_scaler = bundle["y_scaler"]
features = bundle["features"]
targets = bundle["targets"]

with st.sidebar:
    st.header("Parámetros de entrada")

    process_label = st.selectbox(
        "Tipo de proceso",
        ["Isobárico", "Isocórico", "Isotérmico", "Adiabático"],
    )
    process_map = {
        "Isobárico": "isobarico",
        "Isocórico": "isocorico",
        "Isotérmico": "isotermico",
        "Adiabático": "adiabatico",
    }
    process = process_map[process_label]

    P1_kPa = st.number_input(
        "Presión inicial P₁ (kPa)",
        min_value=1.0, max_value=2000.0, value=100.0, step=1.0
    )
    V1_L = st.number_input(
        "Volumen inicial V₁ (L)",
        min_value=1.0, max_value=100.0, value=24.0, step=0.5
    )
    T1 = st.number_input(
        "Temperatura inicial T₁ (K)",
        min_value=100.0, max_value=1500.0, value=300.0, step=5.0
    )

    if process == "isocorico":
        st.info("En un proceso isocórico V₂ = V₁.")
        V2_L = V1_L
    else:
        V2_L = st.number_input(
            "Volumen final V₂ (L)",
            min_value=1.0, max_value=100.0, value=36.0, step=0.5
        )

    n = st.number_input(
        "Cantidad de sustancia n (mol)",
        min_value=0.01, max_value=20.0, value=1.0, step=0.1
    )

    gamma = st.number_input(
        "Coeficiente γ",
        min_value=1.01, max_value=2.0, value=1.40, step=0.01
    )

    default_cv = R / (gamma - 1.0)
    Cv = st.number_input(
        "Capacidad calorífica Cv [J/(mol·K)]",
        min_value=1.0, max_value=1000.0,
        value=float(round(default_cv, 3)),
        step=0.1
    )

    if process == "adiabatico":
        physical_cv = R / (gamma - 1.0)
        if abs(Cv - physical_cv) / physical_cv > 0.05:
            st.warning(
                f"Para un gas ideal consistente con γ={gamma:.2f}, "
                f"Cv debería ser aproximadamente {physical_cv:.3f} J/(mol·K)."
            )

P1 = P1_kPa * 1000.0
V1 = V1_L / 1000.0
V2 = V2_L / 1000.0

try:
    exact = simulate_process(
        process, P1, V1, T1, V2, n=n, gamma=gamma, Cv=Cv
    )
except ValueError as exc:
    st.error(str(exc))
    st.stop()

# Vector de entrada exactamente igual al usado durante el entrenamiento.
x = pd.DataFrame([{
    "P1": exact.P1,
    "V1": exact.V1,
    "T1": exact.T1,
    "V2": exact.V2,
    "gamma": exact.gamma,
    "Cv": exact.Cv,
    "process_adiabatico": int(process == "adiabatico"),
    "process_isobarico": int(process == "isobarico"),
    "process_isocorico": int(process == "isocorico"),
    "process_isotermico": int(process == "isotermico"),
}])[features]

pred_scaled = model.predict(x)
pred = y_scaler.inverse_transform(pred_scaled)[0]
prediction = dict(zip(targets, pred))

st.subheader("Resultados")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("P₂ exacta", f"{exact.P2 / 1000:.3f} kPa")
col2.metric("T₂ exacta", f"{exact.T2:.3f} K")
col3.metric("ΔU exacta", f"{exact.delta_u:.3f} J")
col4.metric("W exacto", f"{exact.W:.3f} J")
col5.metric("Q exacto", f"{exact.Q:.3f} J")

comparison = pd.DataFrame({
    "Variable": ["P₂", "T₂", "ΔU", "W", "Q"],
    "Exacto": [
        exact.P2 / 1000,
        exact.T2,
        exact.delta_u,
        exact.W,
        exact.Q,
    ],
    "IA (MLP)": [
        prediction["P2"] / 1000,
        prediction["T2"],
        prediction["delta_U"],
        prediction["W"],
        prediction["Q"],
    ],
})

comparison["Error absoluto"] = np.abs(comparison["Exacto"] - comparison["IA (MLP)"])

st.dataframe(
    comparison.style.format({
        "Exacto": "{:.6g}",
        "IA (MLP)": "{:.6g}",
        "Error absoluto": "{:.6g}",
    }),
    use_container_width=True,
    hide_index=True,
)

st.subheader("Comparación visual")

tab1, tab2 = st.tabs(["Diagrama P–V", "Conservación de energía"])

with tab1:
    V, P = p_v_curve(exact)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(V * 1000, P / 1000, label="Proceso teórico")
    ax.scatter(
        [exact.V1 * 1000, exact.V2 * 1000],
        [exact.P1 / 1000, exact.P2 / 1000],
        label="Estados 1 y 2",
    )
    ax.set_xlabel("Volumen (L)")
    ax.set_ylabel("Presión (kPa)")
    ax.set_title(f"Proceso {process_label}")
    ax.grid(True, alpha=0.25)
    ax.legend()
    st.pyplot(fig)

with tab2:
    exact_balance = exact.delta_u + exact.W
    pred_balance = prediction["delta_U"] + prediction["W"]

    energy_df = pd.DataFrame({
        "Magnitud": ["Q", "ΔU + W"],
        "Exacto": [exact.Q, exact_balance],
        "IA": [prediction["Q"], pred_balance],
    })

    st.bar_chart(energy_df.set_index("Magnitud"))
    st.write(
        f"Residual teórico: **{exact.energy_error:.3e} J**. "
        f"Residual de IA: "
        f"**{prediction['Q'] - pred_balance:.3e} J**."
    )

st.subheader("Datos del modelo")
st.write(
    "Entradas: P₁, V₁, T₁, V₂, γ, Cv y tipo de proceso mediante one-hot encoding."
)
st.write(
    "Salidas: P₂, T₂, ΔU, W y Q. "
    "El modelo es un MLP con tres capas ocultas (128–64–32)."
)
