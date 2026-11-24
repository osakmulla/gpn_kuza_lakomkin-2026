import pandas as pd

data = pd.read_csv("data/raw/hakhathon_data.csv")

signals = ["Depth", "HL", "TravelingBlockPosition", "ManifoldPressure", "ROP",
           "InletFlow", "Load", "BitPosition", "WedgePosition", "Turn_rotor"]

print("типы сигналов:", list(data[signals].dtypes.unique()))
print()
print("признак | уникальных | доля главного значения | значений на 95% данных | нулей | отрицательных | вывод")

for name in signals:
    column = data[name]
    shares = column.round(2).value_counts(normalize=True)
    unique_count = len(shares)
    top_share = shares.iloc[0]
    need95 = (shares.cumsum() < 0.95).sum() + 1
    zeros = (column == 0).mean()
    negative = (column < 0).mean()

    if need95 <= 10:
        verdict = "категориальный"
    elif top_share > 0.5:
        verdict = "числовой, есть главное значение"
    else:
        verdict = "числовой"

    print(name, "|", unique_count, "|", round(top_share, 3), "|", need95, "|",
          round(zeros, 3), "|", round(negative, 3), "|", verdict)

print()
wedge = data["WedgePosition"]
print("WedgePosition: меньше 0.2:", round((wedge < 0.2).mean(), 3))
print("WedgePosition: от 0.2 до 0.8:", round(((wedge >= 0.2) & (wedge <= 0.8)).mean(), 3))
print("WedgePosition: больше 0.8:", round((wedge > 0.8).mean(), 3))

print()
bit = data["BitPosition"]
depth = data["Depth"]
print("BitPosition не больше Depth:", round((bit <= depth + 1).mean(), 3))
print("BitPosition больше 100:", round((bit > 100).mean(), 3))

print()
pump = data["InletFlow"] > 0
high_pressure = data["ManifoldPressure"] > 5
print("доли: строки - насос, столбцы - давление выше 5")
print(pd.crosstab(pump, high_pressure, normalize="all").round(3))

print()
rop_main = data["ROP"].mode()[0]
depth_max = depth.max()
rop_const = data["ROP"] == rop_main
depth_const = depth == depth_max
print("ROP равен главному значению:", round(rop_const.mean(), 3))
print("Depth равен максимуму:", round(depth_const.mean(), 3))
print("оба условия сразу:", round((rop_const & depth_const).mean(), 3))
