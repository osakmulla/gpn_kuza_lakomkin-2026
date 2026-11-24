import pandas as pd

data = pd.read_csv("data/raw/hakhathon_data.csv")

print("строк:", data.shape[0])
print("столбцов:", data.shape[1])
print()
print("названия и типы столбцов:")
print(data.dtypes)
print()
print("первые 5 строк:")
print(data.head().to_string())
print()
print("последние 3 строки:")
print(data.tail(3).to_string())
print()
print("первое время:", data["Time"].iloc[0])
print("последнее время:", data["Time"].iloc[-1])
print("ID от", data["ID"].min(), "до", data["ID"].max())
