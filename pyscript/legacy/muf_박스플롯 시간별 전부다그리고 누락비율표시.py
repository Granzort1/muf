import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.font_manager as fm
from scipy.stats import norm
import matplotlib.ticker as ticker
import math
from matplotlib.dates import DateFormatter
import matplotlib.patches as mpatches
from datetime import datetime, timedelta
import matplotlib.ticker as mticker
import matplotlib.lines as mlines


plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
# mathtext에서도 유니코드 마이너스 문제 해결
plt.rcParams['mathtext.default'] = 'regular'
# 로그스케일에서 마이너스 기호 사용 방지
plt.rcParams['axes.formatter.use_mathtext'] = True

# 장소 영문명 매핑 (필요시 사용)
location_mapping = {
    "가산A1타워주차장": "Underground Parking Facility",
    "에이샵스크린골프": "Indoor Golf Simulation Facility",
    "영통역대합실": "Subway Station",
    "영통역지하역사": "Subway Station",
    "이든어린이집": "Childcare Center",
    "좋은이웃데이케어센터": "Daycare Center",
    "하이씨앤씨학원": "Educational Facility"
}

# 물질 학술명칭 및 단위 매핑 (필요시 사용)
pollutant_mapping = {
    "pm10": {"name": "PM10", "unit": "μg/m³"},
    "pm25": {"name": "PM2.5", "unit": "μg/m³"},
    "humi": {"name": "Relative Humidity", "unit": "%"},
    "temp": {"name": "Temperature", "unit": "°C"},
    "hcho": {"name": "HCHO", "unit": "μg/m³"},
    "co": {"name": "CO", "unit": "ppm"},
    "no2": {"name": "NO2", "unit": "ppb"},  # 실측 데이터는 ppb 단위
    "voc": {"name": "VOC", "unit": "μg/m³"},
    "co2": {"name": "CO2", "unit": "ppm"},
    "pm1": {"name": "PM1.0", "unit": "μg/m³"},  # 참고 코드에는 없지만 임의로 추가
    "rn": {"name": "Rn", "unit": "pCi/L"},      # 참고 코드에는 없지만 임의로 추가
    "tab": {"name": "TAB", "unit": "CFU/m³"}    # 참고 코드에는 없지만 임의로 추가
}

# 시설 유형별 분류
facility_type_mapping = {
    "가산A1타워주차장": "실내주차장",
    "에이샵스크린골프": "실내체육시설",
    "영통역대합실": "다중이용시설",
    "영통역지하역사": "다중이용시설",
    "이든어린이집": "어린이집",
    "좋은이웃데이케어센터": "노인요양시설",
    "하이씨앤씨학원": "다중이용시설"
}

# 시설 유형별 실내공기질 유지기준 + 권고기준
# (단위: PM10, PM2.5, HCHO, VOC: μg/m³ | CO, CO2: ppm | NO2: ppb)
standards_by_facility_type = {
    "다중이용시설": {
        "pm10": 100,  # μg/m³
        "pm25": 50,   # μg/m³
        "co2": 1000,  # ppm
        "hcho": 100,  # μg/m³
        "co": 10,     # ppm
        "no2": 100,   # ppb
        "voc": 500    # μg/m³
    },
    "어린이집": {
        "pm10": 75,   # μg/m³
        "pm25": 35,   # μg/m³
        "co2": 1000,  # ppm
        "voc": 400,   # μg/m³
        "hcho": 80,   # μg/m³
        "co": 10,     # ppm
        "no2": 50     # ppb
    },
    "노인요양시설": {
        "pm10": 75,   # μg/m³
        "pm25": 35,   # μg/m³
        "co2": 1000,  # ppm
        "voc": 400,   # μg/m³
        "hcho": 80,   # μg/m³
        "co": 10,     # ppm
        "no2": 50     # ppb
    },
    "실내주차장": {
        "pm10": 200,    # μg/m³
        "pm25": 1000000,# μg/m³ (명확 기준 없어 임의)
        "hcho": 100,    # μg/m³
        "co": 25,       # ppm
        "no2": 300,     # ppb
        "co2": 1000,    # ppm
        "voc": 1000     # μg/m³
    },
    "실내체육시설": {
        "pm10": 200,         # μg/m³
        "pm25": 10000000,    # μg/m³ (명확 기준 없어 임의)
        "co2": 100000000,    # ppm (임의 크게 설정)
        "hcho": 100000000,   # μg/m³
        "co": 100000000,     # ppm
        "no2": 100000000,    # ppb
        "voc": 100000000     # μg/m³
    }
}




# CSV 읽기
ob1_min = pd.read_csv("C:/muf/input/가산A1타워주차장_20230331.csv", encoding='utf-8')
ob2_min = pd.read_csv("C:/muf/input/에이샵스크린골프_20230331.csv", encoding='utf-8')
ob3_min = pd.read_csv("C:/muf/input/영통역대합실_20230331.csv", encoding='utf-8')
ob4_min = pd.read_csv("C:/muf/input/영통역지하역사_20230331.csv", encoding='utf-8')
ob5_min = pd.read_csv("C:/muf/input/이든어린이집_20230331.csv", encoding='utf-8')
ob6_min = pd.read_csv("C:/muf/input/좋은이웃데이케어센터1_20230331.csv", encoding='utf-8')
ob7_min = pd.read_csv("C:/muf/input/좋은이웃데이케어센터2_20230331.csv", encoding='utf-8')
ob8_min = pd.read_csv("C:/muf/input/좋은이웃데이케어센터3_20230331.csv", encoding='utf-8')
ob9_min = pd.read_csv("C:/muf/input/좋은이웃데이케어센터4_20230331.csv", encoding='utf-8')
ob10_min = pd.read_csv("C:/muf/input/하이씨앤씨학원_20230331.csv", encoding='utf-8')

# ob별 시설명(키)를 사전에 매핑하여 사용(파일명에 포함된 키)
ob_to_facility_key = {
    'ob1_min': "가산A1타워주차장",
    'ob2_min': "에이샵스크린골프",
    'ob3_min': "영통역대합실",
    'ob4_min': "영통역지하역사",
    'ob5_min': "이든어린이집",
    'ob6_min': "좋은이웃데이케어센터",  # 좋은이웃...1~4 모두 '좋은이웃데이케어센터' 동일 유형
    'ob7_min': "좋은이웃데이케어센터",
    'ob8_min': "좋은이웃데이케어센터",
    'ob9_min': "좋은이웃데이케어센터",
    'ob10_min': "하이씨앤씨학원"
}

def calculate_missing_percentage(df):
    """
    datetime 컬럼을 바탕으로 누락된 데이터를 계산하는 함수.
    - dataframe의 datetime 최솟값~최댓값에 대해 시간별 60개(분 단위)가 다 있는지 확인
    """
    if 'datetime' not in df.columns:
        return 0.0

    total_hours = (df['datetime'].max() - df['datetime'].min()).total_seconds() // 3600

    df['hour'] = df['datetime'].dt.hour
    df['date'] = df['datetime'].dt.date

    # 시간별 갯수
    counts_per_hour = df.groupby(['date', 'hour']).size()

    # 각 시간에서 60개보다 적은 개수인 경우 그 누락 개수를 합산
    missing_counts = (60 - counts_per_hour[counts_per_hour < 60]).sum()

    # 전체 시간 * 60(분)에 대해 누락 비율 계산
    missing_percentage = (missing_counts / (total_hours * 60)) * 100
    return missing_percentage

def get_standard_line(facility_key, pollutant):
    """
    참고 코드의 facility_type_mapping, standards_by_facility_type를 사용해
    시설별 기준을 동적으로 가져와 반환하는 함수.
    기준이 없으면 None을 반환.
    """
    # 시설유형
    facility_type = facility_type_mapping.get(facility_key, None)
    if not facility_type:
        return None  # 매핑이 없는 경우

    # 해당 시설유형의 기준표
    standards_for_facility = standards_by_facility_type.get(facility_type, {})

    # 해당 pollutant에 대한 기준값
    return standards_for_facility.get(pollutant, None)

def graph(df, title_name, facility_key):
    # tmfc_d, tmfc_h로부터 datetime 생성
    df['datetime'] = pd.to_datetime(
        df['tmfc_d'].astype(str) + 
        df['tmfc_h'].astype(str).str.zfill(2), 
        format='%Y%m%d%H'
    )

    df['log_pm10'] = np.log(df['pm10']) if 'pm10' in df.columns else None

    # 분석 대상(혹은 그래프 대상) 항목
    models_one = ["pm10", "pm25", "pm1", "humi", "temp", 
                  "hcho", "co", "no2", "rn", "voc", "co2", "tab"]
    
    # 측정 항목마다 그래프 생성
    for model_one in models_one:
        if model_one not in df.columns:
            # CSV에 해당 항목이 없다면 스킵
            continue
        
        # 기준선 가져오기
        standard_line = get_standard_line(facility_key, model_one)

        # 표시용 라벨 (기본값으로 설정)
        model_name_4th = model_one.upper()
        model_name_label = "(unit)"
        
        # pollutant_mapping에서 실제 이름, 단위 가져오기
        if model_one in pollutant_mapping:
            model_name_4th = pollutant_mapping[model_one]['name']
            model_name_label = pollutant_mapping[model_one]['unit']

        # 그래프 그리기
        plt.figure(figsize=(14, 7))
        plt.plot(df['datetime'], df[model_one],
                 'o', markersize=1, label=f"{model_name_4th}",
                 color='black', markeredgewidth=0)
        
        # y축 로그 스케일
        plt.yscale("symlog")

        # 기준선(standard_line) 존재 시 빨간 점선 표시
        if standard_line is not None:
            plt.axhline(y=standard_line, color='red', linestyle='--',
                        label=f"기준 = {standard_line}")

        # 축/제목
        plt.xlabel('Date', size=12)
        plt.ylabel(model_name_label, size=12)
        plt.title(f'Time series of {model_name_4th} {title_name}', size=12)

        # 범례(마커 크게)
        bigger_dot = mlines.Line2D([], [], color='black', marker='o', linestyle='None',
                                   markersize=3.7)  # 마커 크기 키움
        # 기준선도 추가되었으면 legend 표시에 포함
        if standard_line is not None:
            plt.legend([bigger_dot, mlines.Line2D([], [], color='red', linestyle='--')],
                       [f"{model_name_4th}", f"기준={standard_line}"],
                       fontsize=8)
        else:
            plt.legend([bigger_dot], [f"{model_name_4th}"], fontsize=8)

        # 누락 데이터(빨간선) 표시
        all_dates = pd.date_range(start=df['datetime'].min(), 
                                  end=df['datetime'].max(), freq='h')
        # 한 시간(1H)에 60개 미만이면 누락된 것으로 간주(혹은 완전 빠진 시간대)
        incomplete_hours = df['datetime'].value_counts().loc[lambda x: x < 60].index
        missing_dates = pd.Index(all_dates).difference(df['datetime']).union(incomplete_hours)
        if not missing_dates.empty:
            print(f'Missing dates for {model_one} in {title_name}: {missing_dates}')

        missing_dates = missing_dates.to_series().sort_values()
        missing_periods = [
            (g.iloc[0], g.iloc[-1]) 
            for _, g in missing_dates.groupby((missing_dates.diff().dt.total_seconds() > 3600).cumsum())
        ]

        y_min_value = df[model_one].min()
        for start, end in missing_periods:
            if start == end:
                plt.scatter(start, y_min_value, color='red', s=1) 
            else:
                plt.hlines(y=y_min_value, xmin=start, xmax=end, color='red')

        # 월 경계선 (예: 2,5,8,11월 다음 달 초 vertical line)
        for year in df['datetime'].dt.year.unique():
            for month in [2, 5, 8, 11]:
                if ((df['datetime'] >= datetime(year, month, 1)).any() 
                    and (df['datetime'] < datetime(year, month+1, 1)).any()):
                    plt.axvline(x=datetime(year, month+1, 1),
                                color='blue', linestyle='--')

        # 그래프 안쪽에 누락퍼센트 텍스트로 표시
        missing_percentage = calculate_missing_percentage(df)
        plt.text(0.01, 0.01, f'Missing data: {missing_percentage:.2f}%',
                 fontsize=12, transform=plt.gca().transAxes)
        
        plt.tight_layout()
        # 파일명에 model_one, 시설명 등 구분
        plt.savefig(f"C:/muf/result/graph/{model_one}_time({title_name})(log).png", dpi=370)
        plt.close()


##############################################
# 3. 실제 각 ob별로 그래프 그리기
##############################################
# graph(ob1_min, "(가산A1타워주차장)", "가산A1타워주차장")
# graph(ob2_min, "(에이샵스크린골프)", "에이샵스크린골프")
# graph(ob3_min, "(영통역대합실)", "영통역대합실")
# graph(ob4_min, "(영통역지하역사)", "영통역지하역사")
# graph(ob5_min, "(이든어린이집)", "이든어린이집")
# graph(ob6_min, "(좋은이웃데이케어센터1)", "좋은이웃데이케어센터")
# graph(ob7_min, "(좋은이웃데이케어센터2)", "좋은이웃데이케어센터")
# graph(ob8_min, "(좋은이웃데이케어센터3)", "좋은이웃데이케어센터")
graph(ob9_min, "(좋은이웃데이케어센터4)", "좋은이웃데이케어센터")
# graph(ob10_min, "(하이씨앤씨학원)", "하이씨앤씨학원")

print("그래프 생성 완료!")
