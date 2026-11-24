import pandas as pd

data = pd.read_csv("data/processed/clean.csv")

print("1. ДЛИННЫЕ ЭПИЗОДЫ ОТРИЦАТЕЛЬНЫХ ЗНАЧЕНИЙ")
for name in ["HL", "TravelingBlockPosition", "Turn_rotor"]:
    negative = data[name] < 0
    group = (negative != negative.shift()).cumsum()
    lengths = negative.groupby(group).sum()
    lengths = lengths[lengths > 0]
    print(name, "| эпизодов:", len(lengths), "| самый длинный:", int(lengths.max()),
          "| всего строк:", int(negative.sum()))

print()
print("2. BitPosition БОЛЬШЕ Depth БОЛЕЕ ЧЕМ НА 1")
above = data["BitPosition"] > data["Depth"] + 1
group = (above != above.shift()).cumsum()
lengths = above.groupby(group).sum()
lengths = lengths[lengths > 0]
print("эпизодов:", len(lengths), "| длины:", lengths.tolist())
print("по сериям:", data[above].groupby("series_id").size().to_dict())
