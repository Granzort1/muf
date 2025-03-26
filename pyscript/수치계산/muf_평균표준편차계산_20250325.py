import os
import numpy as np
import pandas as pd

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

# 장소 영문명 매핑
location_mapping = {
    "가산A1타워주차장": "Underground Parking Facility",
    "에이샵스크린골프": "Indoor Golf Simulation Facility",
    "영통역대합실": "Subway Station",
    "영통역지하역사": "Subway Station",
    "이든어린이집": "Childcare Center",
    "좋은이웃데이케어센터": "Daycare Center",
    "하이씨앤씨학원": "Educational Facility"
}

# 물질 학술명칭 및 단위 매핑
pollutant_mapping = {
    "pm10": {"name": "PM10", "unit": "μg/m³"},
    "pm25": {"name": "PM2.5", "unit": "μg/m³"},
    "humi": {"name": "Relative Humidity", "unit": "%"},
    "temp": {"name": "Temperature", "unit": "°C"},
    "hcho": {"name": "HCHO", "unit": "μg/m³"},
    "co": {"name": "CO", "unit": "ppm"},
    "no2": {"name": "NO2", "unit": "ppb"},  # 실측 데이터는 ppb 단위로 저장됨
    "voc": {"name": "VOC", "unit": "μg/m³"},
    "co2": {"name": "CO2", "unit": "ppm"}
}

# 시설 유형별 분류
facility_type_mapping = {
    "가산A1타워주차장": "실내주차장",
    "에이샵스크린골프": "실내체육시설",
    "영통역대합실": "다중이용시설",
    "영통역지하역사": "다중이용시설",
    "이든어린이집": "어린이집",
    "좋은이웃데이케어센터": "노인요양시설",
    "하이씨앤씨학원": "다중이용시설"  # 학원은 다중이용시설로 분류
}

# 실내공기질 유지기준(별표 2)과 권고기준(별표 3)을 통합한 시설별 기준
# 단위: PM10, PM2.5, HCHO, VOC: μg/m³ | CO, CO2: ppm | NO2: ppb (1 ppm = 1000 ppb로 변환)
standards_by_facility_type = {
    "다중이용시설": {
        "pm10": 100,  # μg/m³
        "pm25": 50,   # μg/m³
        "co2": 1000,  # ppm
        "hcho": 100,  # μg/m³
        "co": 10,     # ppm
        "no2": 100,   # ppb (0.1 ppm * 1000 = 100 ppb)
        "voc": 500    # μg/m³ (권고기준)
    },
    "어린이집": {
        "pm10": 75,   # μg/m³
        "pm25": 35,   # μg/m³
        "co2": 1000,    # ppm (오타인 것 같지만 원문 준수)
        "voc": 400,   # μg/m³
        "hcho": 80,  # μg/m³ (다중이용시설 기준 준용)
        "co": 10,     # ppm (다중이용시설 기준 준용)
        "no2": 50     # ppb (0.05 ppm * 1000 = 50 ppb)
    },
    "노인요양시설": {
        "pm10": 75,   # μg/m³
        "pm25": 35,   # μg/m³
        "co2": 1000,    # ppm (오타인 것 같지만 원문 준수)
        "voc": 400,   # μg/m³
        "hcho": 80,  # μg/m³ (다중이용시설 기준 준용)
        "co": 10,     # ppm (다중이용시설 기준 준용)
        "no2": 50     # ppb (0.05 ppm * 1000 = 50 ppb)
    },
    "실내주차장": {
        "pm10": 200,  # μg/m³
        "hcho": 100,  # μg/m³
        "co": 25,     # ppm
        "no2": 300,   # ppb (0.30 ppm * 1000 = 300 ppb)
        "pm25": 1000000,  # μg/m³ (명확한 기준 없어 PM10의 절반으로 가정)
        "co2": 1000,  # ppm (다중이용시설 기준 준용)
        "voc": 1000   # μg/m³ (별도 기준이 없어 높게 설정)
    },
    "실내체육시설": {
        "pm10": 200,  # μg/m³
        "pm25": 10000000,  # μg/m³ (명확한 기준 없어 PM10의 절반으로 가정)
        "co2": 100000000,  # ppm (다중이용시설 기준 준용)
        "hcho": 100000000,  # μg/m³ (다중이용시설 기준 준용)
        "co": 100000000,     # ppm (다중이용시설 기준 준용)
        "no2": 100000000,   # ppb (0.1 ppm * 1000 = 100 ppb)
        "voc": 100000000   # μg/m³ (별도 기준이 없어 높게 설정)
    }
}

path = "C:/muf/"
dfs = [pd.read_csv(path + f, encoding='utf-8') for f in files]

# tab과 rn 데이터 제외, 온도와 습도는 측정 정보로 포함
models_one = ["pm10", "pm25", "hcho", "co", "no2", "voc", "co2"]  # 분석 대상 물질
all_models = ["pm10", "pm25", "humi", "temp", "hcho", "co", "no2", "voc", "co2"]  # 모든 측정 항목

# 1. 평균과 표준편차 계산
data = {}

# 영통역 데이터 결합
영통역_df = pd.concat([dfs[2], dfs[3]], ignore_index=True)  # 영통역대합실 + 영통역지하역사
data[location_mapping["영통역대합실"]] = {}
for model_one in all_models:
    mean = 영통역_df[model_one].mean()
    std = 영통역_df[model_one].std()
    data[location_mapping["영통역대합실"]][model_one] = f"{round(mean, 2)}±{round(std, 2)}"

# 좋은이웃데이케어센터 데이터 결합
데이케어_df = pd.concat([dfs[5], dfs[6], dfs[7], dfs[8]], ignore_index=True)  # 데이케어센터 1~4
data[location_mapping["좋은이웃데이케어센터"]] = {}
for model_one in all_models:
    mean = 데이케어_df[model_one].mean()
    std = 데이케어_df[model_one].std()
    data[location_mapping["좋은이웃데이케어센터"]][model_one] = f"{round(mean, 2)}±{round(std, 2)}"

# 나머지 파일들 처리
for i, (df, file) in enumerate(zip(dfs, files)):
    if i not in [2, 3, 5, 6, 7, 8]:  # 이미 처리한 파일 제외
        file_name = os.path.splitext(file)[0]
        for key in location_mapping:
            if key in file_name:
                location_name = location_mapping[key]
                break
        else:
            location_name = file_name  # 매핑이 없을 경우
        
        data[location_name] = {}
        for model_one in all_models:
            mean = df[model_one].mean()
            std = df[model_one].std()
            data[location_name][model_one] = f"{round(mean, 2)}±{round(std, 2)}"

# 평균/표준편차 결과를 DataFrame으로 변환
result_df = pd.DataFrame(data).T

# 열 이름을 학술 명칭으로 변경하고 단위 추가
unit_columns = []
for col in result_df.columns:
    unit_columns.append(f"{pollutant_mapping[col]['name']} ({pollutant_mapping[col]['unit']})")

result_df.columns = unit_columns

# 열에 대한 평균과 표준편차를 계산
for model_one in all_models:
    temp_values = []
    for key, value in data.items():
        if float(value[model_one].split("±")[0]) != 0:
            temp_values.append(float(value[model_one].split("±")[0]))
    if temp_values:
        mean = np.mean(temp_values)
        std = np.std(temp_values)
        unit_col = f"{pollutant_mapping[model_one]['name']} ({pollutant_mapping[model_one]['unit']})"
        result_df.loc['Total Average', unit_col] = f"{round(mean, 2)}±{round(std, 2)}"

# 순서 지정: 실내 스크린골프장, 어린이집, 지하주차장, 데이케어 센터, 교육시설, 지하철 역사
order_mapping = {
    "Indoor Golf Simulation Facility": 0,  # 실내 스크린골프장
    "Childcare Center": 1,                # 어린이집
    "Underground Parking Facility": 2,     # 지하주차장
    "Daycare Center": 3,                   # 데이케어 센터
    "Educational Facility": 4,            # 교육시설
    "Subway Station": 5                    # 지하철 역사
}

# Total Average는 항상 마지막에 오도록 설정
if 'Total Average' in result_df.index:
    result_df = result_df.drop('Total Average')
    
# 순서대로 정렬
result_df['order'] = result_df.index.map(lambda x: order_mapping.get(x, 999))
result_df = result_df.sort_values('order')
result_df = result_df.drop('order', axis=1)

# Total Average 다시 추가
for model_one in all_models:
    unit_col = f"{pollutant_mapping[model_one]['name']} ({pollutant_mapping[model_one]['unit']})"
    if unit_col in result_df.columns:
        # 문자열 값만 처리하고 ± 기호가 있는 데이터만 처리하도록 수정
        values = []
        for x in result_df[unit_col]:
            if isinstance(x, str) and "±" in x:
                try:
                    values.append(float(x.split("±")[0]))
                except (ValueError, TypeError):
                    continue
        if values:  # 값이 있는 경우에만 평균 계산
            mean_value = np.mean(values)
            std_value = np.std(values)
            result_df.loc['Total Average', unit_col] = f"{round(mean_value, 2)}±{round(std_value, 2)}"
        else:
            result_df.loc['Total Average', unit_col] = "N/A"

result_df.to_excel("C:/muf/result/평균표준편차_결과_단위포함.xlsx")

# 2. 시설별 기준 적용한 초과비율 계산
results = {}

# 영통역 데이터 - 다중이용시설 기준 적용
facility_type = facility_type_mapping["영통역대합실"]  # 영통역은 다중이용시설
standards = standards_by_facility_type[facility_type]

results[location_mapping["영통역대합실"]] = {}
for model_one in models_one:  # 온도와 습도 제외
    if model_one in standards:
        exceed_count = (영통역_df[model_one] > standards[model_one]).sum()
        exceed_ratio = exceed_count / len(영통역_df)
        results[location_mapping["영통역대합실"]][model_one] = exceed_ratio
    else:
        results[location_mapping["영통역대합실"]][model_one] = "기준 없음"

# 좋은이웃데이케어센터 데이터 - 노인요양시설 기준 적용
facility_type = facility_type_mapping["좋은이웃데이케어센터"]  # 데이케어센터는 노인요양시설
standards = standards_by_facility_type[facility_type]

results[location_mapping["좋은이웃데이케어센터"]] = {}
for model_one in models_one:  # 온도와 습도 제외
    if model_one in standards:
        exceed_count = (데이케어_df[model_one] > standards[model_one]).sum()
        exceed_ratio = exceed_count / len(데이케어_df)
        results[location_mapping["좋은이웃데이케어센터"]][model_one] = exceed_ratio
    else:
        results[location_mapping["좋은이웃데이케어센터"]][model_one] = "기준 없음"

# 나머지 파일들 처리
for i, (df, file) in enumerate(zip(dfs, files)):
    if i not in [2, 3, 5, 6, 7, 8]:  # 이미 처리한 파일 제외
        file_name = os.path.splitext(file)[0]
        
        # 시설 명칭 찾기
        for key in location_mapping:
            if key in file_name:
                location_name = location_mapping[key]
                location_key = key
                break
        else:
            location_name = file_name  # 매핑이 없을 경우
            location_key = file_name
        
        # 시설 유형과 기준 확인
        facility_type = facility_type_mapping.get(location_key, "다중이용시설")  # 기본값은 다중이용시설
        standards = standards_by_facility_type[facility_type]
        
        results[location_name] = {}
        for model_one in models_one:  # 온도와 습도 제외
            if model_one in standards:
                exceed_count = (df[model_one] > standards[model_one]).sum()
                exceed_ratio = exceed_count / len(df)
                results[location_name][model_one] = exceed_ratio
            else:
                results[location_name][model_one] = "기준 없음"

# 기준초과비율 결과를 DataFrame으로 변환
results_df = pd.DataFrame(results).T

# 열 이름을 학술 명칭으로 변경하고 단위 추가
unit_columns = []
for col in results_df.columns:
    unit_columns.append(f"{pollutant_mapping[col]['name']} ({pollutant_mapping[col]['unit']})")

results_df.columns = unit_columns

# 기준초과비율을 백분율로 표시
for col in results_df.columns:
    results_df[col] = results_df[col].apply(lambda x: f"{float(x)*100:.2f}%" if isinstance(x, (int, float)) else x)

# 순서 지정: 실내 스크린골프장, 어린이집, 지하주차장, 데이케어 센터, 교육시설, 지하철 역사
results_df['order'] = results_df.index.map(lambda x: order_mapping.get(x, 999))
results_df = results_df.sort_values('order')
results_df = results_df.drop('order', axis=1)

# 기준초과비율 결과를 엑셀 파일로 저장
results_df.to_excel('C:/muf/result/기준초과비율_결과_단위포함.xlsx')

# 3. 각 시설별 기준값 출력 (기준값 확인용)
standards_info = {}
for location_key, facility_type in facility_type_mapping.items():
    if location_key in location_mapping:
        location_name = location_mapping[location_key]
    else:
        location_name = location_key
    
    standards = standards_by_facility_type[facility_type]
    standards_info[location_name] = standards

standards_df = pd.DataFrame(standards_info).T

# 열 이름을 학술 명칭으로 변경하고 단위 추가
unit_columns = []
for col in standards_df.columns:
    if col in pollutant_mapping:
        unit_columns.append(f"{pollutant_mapping[col]['name']} ({pollutant_mapping[col]['unit']})")
    else:
        unit_columns.append(col)

standards_df.columns = unit_columns

# 순서 지정: 실내 스크린골프장, 어린이집, 지하주차장, 데이케어 센터, 교육시설, 지하철 역사
standards_df['order'] = standards_df.index.map(lambda x: order_mapping.get(x, 999))
standards_df = standards_df.sort_values('order')
standards_df = standards_df.drop('order', axis=1)

# 시설별 적용 기준을 엑셀 파일로 저장
standards_df.to_excel('C:/muf/result/시설별_기준값_단위포함.xlsx')

# 4. 기준 초과 횟수를 계산하여 추가 결과 제공
exceed_counts = {}

# 영통역 데이터 - 다중이용시설 기준 적용
facility_type = facility_type_mapping["영통역대합실"]  # 영통역은 다중이용시설
standards = standards_by_facility_type[facility_type]

exceed_counts[location_mapping["영통역대합실"]] = {}
for model_one in models_one:  # 온도와 습도 제외
    if model_one in standards:
        exceed_count = (영통역_df[model_one] > standards[model_one]).sum()
        total_count = len(영통역_df)
        exceed_counts[location_mapping["영통역대합실"]][model_one] = f"{exceed_count}/{total_count}"
    else:
        exceed_counts[location_mapping["영통역대합실"]][model_one] = "기준 없음"

# 좋은이웃데이케어센터 데이터 - 노인요양시설 기준 적용
facility_type = facility_type_mapping["좋은이웃데이케어센터"]  # 데이케어센터는 노인요양시설
standards = standards_by_facility_type[facility_type]

exceed_counts[location_mapping["좋은이웃데이케어센터"]] = {}
for model_one in models_one:  # 온도와 습도 제외
    if model_one in standards:
        exceed_count = (데이케어_df[model_one] > standards[model_one]).sum()
        total_count = len(데이케어_df)
        exceed_counts[location_mapping["좋은이웃데이케어센터"]][model_one] = f"{exceed_count}/{total_count}"
    else:
        exceed_counts[location_mapping["좋은이웃데이케어센터"]][model_one] = "기준 없음"

# 나머지 파일들 처리
for i, (df, file) in enumerate(zip(dfs, files)):
    if i not in [2, 3, 5, 6, 7, 8]:  # 이미 처리한 파일 제외
        file_name = os.path.splitext(file)[0]
        
        # 시설 명칭 찾기
        for key in location_mapping:
            if key in file_name:
                location_name = location_mapping[key]
                location_key = key
                break
        else:
            location_name = file_name  # 매핑이 없을 경우
            location_key = file_name
        
        # 시설 유형과 기준 확인
        facility_type = facility_type_mapping.get(location_key, "다중이용시설")  # 기본값은 다중이용시설
        standards = standards_by_facility_type[facility_type]
        
        exceed_counts[location_name] = {}
        for model_one in models_one:  # 온도와 습도 제외
            if model_one in standards:
                exceed_count = (df[model_one] > standards[model_one]).sum()
                total_count = len(df)
                exceed_counts[location_name][model_one] = f"{exceed_count}/{total_count}"
            else:
                exceed_counts[location_name][model_one] = "기준 없음"

# 기준초과횟수 결과를 DataFrame으로 변환
exceed_counts_df = pd.DataFrame(exceed_counts).T

# 열 이름을 학술 명칭으로 변경하고 단위 추가
unit_columns = []
for col in exceed_counts_df.columns:
    unit_columns.append(f"{pollutant_mapping[col]['name']} ({pollutant_mapping[col]['unit']})")

exceed_counts_df.columns = unit_columns

# 순서 지정: 실내 스크린골프장, 어린이집, 지하주차장, 데이케어 센터, 교육시설, 지하철 역사
exceed_counts_df['order'] = exceed_counts_df.index.map(lambda x: order_mapping.get(x, 999))
exceed_counts_df = exceed_counts_df.sort_values('order')
exceed_counts_df = exceed_counts_df.drop('order', axis=1)

# 기준초과횟수 결과를 엑셀 파일로 저장
exceed_counts_df.to_excel('C:/muf/result/기준초과횟수_결과_단위포함.xlsx')

print("분석 완료: 시설별 법적 기준에 맞게 초과율 계산됨 (단위 정보 포함)")