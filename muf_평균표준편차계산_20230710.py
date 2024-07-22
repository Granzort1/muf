import os
import numpy as np
import pandas as pd


'''# CSV 파일들을 읽어 리스트에 저장합니다.
files = ["가산A1타워주차장_20230331.csv",
         "에이샵스크린골프_20230331.csv",
         "영통역대합실_20230331.csv",
         "영통역지하역사_20230331.csv",
         "이든어린이집_20230331.csv",
         "좋은이웃데이케어센터1_20230331.csv",
         "좋은이웃데이케어센터2_20230331.csv",
         "좋은이웃데이케어센터3_20230331.csv",
         "좋은이웃데이케어센터4_20230331.csv",
         "하이씨앤씨학원_20230331.csv"]

path = "C:/muf/"
dfs = [pd.read_csv(path + f, encoding='utf-8') for f in files]

models_one = ["pm10", "pm25", "pm1", "humi", "temp", "hcho", "co", "no2", "rn", "voc", "co2", "tab"]

# 각 파일에 대한 평균과 표준편차를 계산합니다.
data = {}
for df, file in zip(dfs, files):
    file = os.path.splitext(file)[0]  # 확장자를 제거합니다.
    data[file] = {}
    for model_one in models_one:
        mean = df[model_one].mean()
        std = df[model_one].std()
        # 평균+-표준편차를 계산하여 저장합니다.
        data[file][model_one] = f"{round(mean, 2)}±{round(std, 2)}"

# 결과를 DataFrame으로 변환하고 엑셀 파일로 저장합니다.
result_df = pd.DataFrame(data).T

# 열에 대한 평균과 표준편차를 계산합니다.
for model_one in models_one:
    temp_values = []
    for key, value in data.items():
        if float(value[model_one].split("±")[0]) != 0:  # 오염물질 측정데이터셋이 모두 0이면 제외
            temp_values.append(float(value[model_one].split("±")[0]))
    if temp_values:  # 공백 리스트가 아닐 경우만 계산
        mean = np.mean(temp_values)
        std = np.std(temp_values)
        result_df.loc['total', model_one] = f"{round(mean, 2)}±{round(std, 2)}"

result_df.to_excel("1221.xlsx")'''

# CSV 파일들을 읽어 리스트에 저장합니다.
files = ["가산A1타워주차장_20230331.csv",
         "에이샵스크린골프_20230331.csv",
         "영통역대합실_20230331.csv",
         "영통역지하역사_20230331.csv",
         "이든어린이집_20230331.csv",
         "좋은이웃데이케어센터1_20230331.csv",
         "좋은이웃데이케어센터2_20230331.csv",
         "좋은이웃데이케어센터3_20230331.csv",
         "좋은이웃데이케어센터4_20230331.csv",
         "하이씨앤씨학원_20230331.csv"]

path = "C:/muf/"
dfs = [pd.read_csv(path + f, encoding='utf-8') for f in files]

models_one = ["pm10", "pm25", "pm1", "humi", "temp", "hcho", "co", "no2", "rn", "voc", "co2", "tab"]

# Standards for each pollutant
standards = {'pm10': 200, 'pm25': 200, 'pm1': 200, 'humi': 100, 'temp': 10, 'hcho': 100, 'co': 25,
             'no2': 300, 'rn': 4, 'voc': 1000, 'co2': 1000, 'tab': 100}

# 각 파일에 대한 평균과 표준편차를 계산합니다.
results = {}
for df, file in zip(dfs, files):
    location = os.path.splitext(file)[0]  # Remove the extension.
    results[location] = {}
    for model_one in models_one:
        exceed_count = (df[model_one] > standards[model_one]).sum()
        exceed_ratio = exceed_count / len(df)
        results[location][model_one] = exceed_ratio

# Convert the result to a DataFrame and save it as an excel file.
results_df = pd.DataFrame(results).T
results_df.to_excel('exceed_ratios.xlsx')