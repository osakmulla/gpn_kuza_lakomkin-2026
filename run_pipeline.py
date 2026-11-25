
import subprocess
import sys
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent

scripts = [
    "src/preprocessing/preprocessing.py",
    "src/features/features.py",
    "experiments/make_candidates.py"
]

for script in scripts:
    print(f"\nЗапуск: {script}")
    subprocess.run(
        [sys.executable, str(ROOT / script)],
        cwd=ROOT,
        check=True
    )

candidate = ROOT / "submission/candidate_C_segments.csv"
df = pd.read_csv(candidate)

final = df[["ID", "label"]].rename(columns={"label": "class"})
final.to_csv(ROOT / "submission/final_submission.csv", index=False)

print("\nPipeline завершён!")
print("Финальный файл:", ROOT / "submission/final_submission.csv")
