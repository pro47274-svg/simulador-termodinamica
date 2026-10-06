# Simulador Termodinámico + IA

Proyecto en Python para simular procesos termodinámicos ideales y entrenar
una red neuronal MLP que predice P₂, T₂, ΔU, W y Q.

## 1. Estructura

```text
termodinamica_ia/
├── physics.py
├── generate_dataset.py
├── train_model.py
├── app.py
├── run_project.py
├── requirements.txt
├── README.md
├── data/
│   └── dataset.csv              # se crea al ejecutar el proyecto
└── models/
    ├── mlp_thermodynamics.joblib
    └── metrics.txt
```

## 2. Instalar dependencias

En la terminal de VS Code:

```bash
python -m pip install -r requirements.txt
```

## 3. Ejecutar todo de una vez

```bash
python run_project.py
```

Esto genera 10,000 muestras y entrena el MLP.

También se puede hacer por separado:

```bash
python generate_dataset.py --samples 10000
python train_model.py
```

## 4. Abrir el dashboard

```bash
streamlit run app.py
```

Se abrirá una página local en el navegador.

## 5. Modelo físico

Se usa un gas ideal y la convención:

- W > 0: trabajo realizado por el gas.
- Q > 0: calor recibido por el gas.
- Primera Ley: ΔU = Q - W, por lo tanto Q = ΔU + W.

Procesos:

### Isobárico
P = constante.

### Isocórico
V = constante y W = 0.

### Isotérmico
T = constante y ΔU = 0.

### Adiabático
Q = 0 y PV^γ = constante.

No se resuelven numéricamente ecuaciones diferenciales para obtener estos
resultados; se usan las expresiones analíticas de los procesos ideales.

## 6. Sobre el proceso isocórico

Para determinar P₂ y T₂ en un proceso isocórico se necesita conocer una
segunda condición de estado. En esta implementación se considera el caso
trivial de V₂ = V₁ y T₂ = T₁, por lo que W = Q = ΔU = 0.

Si el proyecto necesita un isocórico con calentamiento/enfriamiento,
conviene agregar T₂ como entrada o ΔT como parámetro de entrada.

## 7. Conservación de energía

El entrenamiento verifica:

```text
Q ≈ ΔU + W
```

También se muestra el residual de energía en el dashboard.

## 8. Nota sobre γ y Cv

Para un gas ideal:

```text
γ = Cp/Cv
Cp - Cv = R
```

por lo que:

```text
Cv = R/(γ - 1)
```

El dataset genera γ aleatoriamente y calcula Cv de forma consistente.
