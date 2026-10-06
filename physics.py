"""
Motor de simulación física para gases ideales.

Convención de signos:
    W > 0  -> trabajo realizado POR el gas.
    Q > 0  -> calor recibido POR el gas.
Primera Ley:
    ΔU = Q - W  ->  Q = ΔU + W
"""

from dataclasses import dataclass
import math

R = 8.31446261815324  # J/(mol K)


@dataclass
class ProcessResult:
    process: str
    P1: float
    V1: float
    T1: float
    P2: float
    V2: float
    T2: float
    delta_u: float
    W: float
    Q: float
    gamma: float
    Cv: float

    @property
    def energy_error(self) -> float:
        """Residual de la Primera Ley: Q - (ΔU + W)."""
        return self.Q - (self.delta_u + self.W)


def validate_inputs(P1, V1, T1, V2, n=1.0, gamma=1.4, Cv=None):
    if P1 <= 0 or V1 <= 0 or T1 <= 0 or V2 <= 0:
        raise ValueError("P1, V1, T1 y V2 deben ser mayores que cero.")
    if n <= 0:
        raise ValueError("La cantidad de sustancia n debe ser mayor que cero.")
    if gamma <= 1:
        raise ValueError("gamma debe ser mayor que 1.")
    if Cv is None:
        Cv = R / (gamma - 1)
    if Cv <= 0:
        raise ValueError("Cv debe ser mayor que cero.")
    return float(Cv)


def simulate_process(
    process: str,
    P1: float,
    V1: float,
    T1: float,
    V2: float,
    n: float = 1.0,
    gamma: float = 1.4,
    Cv: float | None = None,
) -> ProcessResult:
    """
    Simula uno de los cuatro procesos ideales sin resolver EDO numéricamente.

    process: isobarico, isocorico, isotermico o adiabatico
    Unidades: Pa, m³, K, mol y J.
    """
    Cv = validate_inputs(P1, V1, T1, V2, n, gamma, Cv)
    process = process.lower().strip()

    if process == "isobarico":
        P2 = P1
        T2 = T1 * V2 / V1
        W = P1 * (V2 - V1)
        delta_u = n * Cv * (T2 - T1)
        Q = delta_u + W

    elif process == "isocorico":
        # En un proceso isocórico el volumen permanece constante.
        V2 = V1
        T2 = T1  # Se requiere una variable adicional para fijar un cambio de T.
        P2 = P1
        W = 0.0
        delta_u = 0.0
        Q = 0.0

    elif process == "isotermico":
        T2 = T1
        P2 = P1 * V1 / V2
        W = n * R * T1 * math.log(V2 / V1)
        delta_u = 0.0
        Q = W

    elif process == "adiabatico":
        # PV^gamma = cte
        P2 = P1 * (V1 / V2) ** gamma
        T2 = T1 * (V1 / V2) ** (gamma - 1.0)
        W = (P1 * V1 - P2 * V2) / (gamma - 1.0)
        delta_u = n * Cv * (T2 - T1)
        Q = 0.0

    else:
        raise ValueError(
            "Proceso no reconocido. Usa: isobarico, isocorico, "
            "isotermico o adiabatico."
        )

    return ProcessResult(
        process=process,
        P1=P1, V1=V1, T1=T1,
        P2=P2, V2=V2, T2=T2,
        delta_u=delta_u, W=W, Q=Q,
        gamma=gamma, Cv=Cv,
    )


def simulate_isobaric(P1, V1, T1, V2, n=1.0, Cv=None, gamma=1.4):
    return simulate_process("isobarico", P1, V1, T1, V2, n, gamma, Cv)


def simulate_isochoric(P1, V1, T1, n=1.0, Cv=None, gamma=1.4):
    return simulate_process("isocorico", P1, V1, T1, V1, n, gamma, Cv)


def simulate_isothermal(P1, V1, T1, V2, n=1.0, Cv=None, gamma=1.4):
    return simulate_process("isotermico", P1, V1, T1, V2, n, gamma, Cv)


def simulate_adiabatic(P1, V1, T1, V2, n=1.0, Cv=None, gamma=1.4):
    return simulate_process("adiabatico", P1, V1, T1, V2, n, gamma, Cv)


def p_v_curve(result: ProcessResult, points: int = 200):
    """Devuelve puntos P,V del proceso para graficarlo."""
    import numpy as np

    V = np.linspace(result.V1, result.V2, points)

    if result.process == "isobarico":
        P = np.full_like(V, result.P1)
    elif result.process == "isocorico":
        V = np.full(points, result.V1)
        P = np.full_like(V, result.P1)
    elif result.process == "isotermico":
        P = result.P1 * result.V1 / V
    elif result.process == "adiabatico":
        P = result.P1 * (result.V1 / V) ** result.gamma
    else:
        raise ValueError("Proceso desconocido.")

    return V, P
