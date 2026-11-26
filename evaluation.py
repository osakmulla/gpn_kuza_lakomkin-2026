
import pandas as pd
from pathlib import Path


def evaluate(path):
    df = pd.read_csv(path)

    required = {"ID", "class"}

    if set(df.columns) != required:
        raise ValueError(f"Неверные столбцы: {df.columns.tolist()}")

    if df.isna().any().any():
        raise ValueError("Есть пропущенные значения")

    if df["ID"].duplicated().any():
        raise ValueError("Есть повторяющиеся ID")

    if len(df) < 60000:
        raise ValueError("Недостаточно строк")

    if df["class"].nunique() != 9:
        raise ValueError("Количество классов не равно 9")

    print("Файл прошёл проверку.")
    print(f"Строк: {len(df)}")
    print(f"Классов: {df['class'].nunique()}")


if __name__ == "__main__":
    path = Path(__file__).parent / "submission" / "final_submission.csv"
    evaluate(path)
