import numpy as np
import pandas as pd

data = pd.read_csv("data/raw/hakhathon_data.csv")
data["Time"] = pd.to_datetime(data["Time"], format="%d.%m.%Y %H:%M:%S")

signals = ["Depth", "HL", "TravelingBlockPosition", "ManifoldPressure", "ROP",
           "InletFlow", "Load", "BitPosition", "WedgePosition", "Turn_rotor"]

print("строк:", len(data))
print()
print("1. ПРОПУСКИ")
print(data.isna().sum().to_string())
print("бесконечных значений:", int(np.isinf(data[signals]).sum().sum()))

print()
print("2. ДУБЛИКАТЫ")
print("полностью одинаковых строк:", int(data.duplicated().sum()))
print("повторяющихся ID:", int(data["ID"].duplicated().sum()))
print("повторяющихся Time:", int(data["Time"].duplicated().sum()))
print("строк, где все сигналы совпадают с предыдущей:",
      int((data[signals].diff().abs().sum(axis=1) == 0).sum()))

print()
print("3. ПОРЯДОК")
print("ID идут подряд без дыр:", bool((data["ID"].diff().dropna() == 1).all()))
print("время строго растёт:", bool((data["Time"].diff().dropna().dt.total_seconds() > 0).all()))

print()
print("4. ШАГ ПО ВРЕМЕНИ")
step = data["Time"].diff().dt.total_seconds()
print(step.value_counts().head(8).to_string())
big = data[step > 60]
print("разрывов больше 60 секунд:", len(big))
for i in big.index:
    print("  ID", data.loc[i, "ID"], "пауза", int(step[i]), "сек, после", data.loc[i - 1, "Time"])

print()
print("5. ПОДОЗРИТЕЛЬНЫЕ ЗНАЧЕНИЯ")
for name in signals:
    column = data[name]
    low = column.quantile(0.001)
    high = column.quantile(0.999)
    print(name, "| отрицательных:", int((column < 0).sum()),
          "| ниже 0.1% квантиля:", int((column < low).sum()),
          "| выше 99.9% квантиля:", int((column > high).sum()))
print("BitPosition больше Depth более чем на 1:", int((data["BitPosition"] > data["Depth"] + 1).sum()))
