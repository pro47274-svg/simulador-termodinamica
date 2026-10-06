"""
Ejecuta automáticamente la generación del dataset y el entrenamiento.

Uso:
    python run_project.py
"""

from generate_dataset import generate_dataset
from train_model import train


def main():
    print("=" * 60)
    print("1) GENERANDO DATASET")
    print("=" * 60)
    from pathlib import Path

    output = Path("data/dataset.csv")
    output.parent.mkdir(exist_ok=True)

    df = generate_dataset(samples=10000, seed=42)
    df.to_csv(output, index=False)
    print(f"Dataset guardado en {output} ({len(df)} muestras).")

    print("\n" + "=" * 60)
    print("2) ENTRENANDO MODELO MLP")
    print("=" * 60)
    train()

    print("\n" + "=" * 60)
    print("PROYECTO LISTO")
    print("=" * 60)
    print("Para abrir el dashboard:")
    print("    streamlit run app.py")


if __name__ == "__main__":
    main()
