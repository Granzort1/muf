import pandas as pd

# 실내 측정 데이터 읽기
indoor_data = [
    ("가산A1타워주차장", pd.read_csv("C:/muf/가산A1타워주차장_20230331.csv", encoding='utf-8')),
    ("에이샵스크린골프", pd.read_csv("C:/muf/에이샵스크린골프_20230331.csv", encoding='utf-8')),
    ("영통역대합실", pd.read_csv("C:/muf/영통역대합실_20230331.csv", encoding='utf-8')),
    ("영통역지하역사", pd.read_csv("C:/muf/영통역지하역사_20230331.csv", encoding='utf-8')),
    ("이든어린이집", pd.read_csv("C:/muf/이든어린이집_20230331.csv", encoding='utf-8')),
    ("좋은이웃데이케어센터1", pd.read_csv("C:/muf/좋은이웃데이케어센터1_20230331.csv", encoding='utf-8')),
    ("좋은이웃데이케어센터2", pd.read_csv("C:/muf/좋은이웃데이케어센터2_20230331.csv", encoding='utf-8')),
    ("좋은이웃데이케어센터3", pd.read_csv("C:/muf/좋은이웃데이케어센터3_20230331.csv", encoding='utf-8')),
    ("좋은이웃데이케어센터4", pd.read_csv("C:/muf/좋은이웃데이케어센터4_20230331.csv", encoding='utf-8')),
    ("하이씨앤씨학원", pd.read_csv("C:/muf/하이씨앤씨학원_20230331.csv", encoding='utf-8'))
]

models = ["pm10", "pm25", "pm1", "humi", "temp", "hcho", "co", "no2", "rn", "voc", "co2", "tab"]

for location, data in indoor_data:
    print(f"{location}:")
    all_zero_models = []

    for model in models:
        if (data[model] <= 0).all():
            all_zero_models.append(model)

    if all_zero_models:
        print(f"  모두 0 이하인 물질: {', '.join(all_zero_models)}")
    else:
        print("  모두 0 이하인 물질이 없습니다.")

    print()