import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from datetime import datetime
import os

# -----------------------------------------------------------------------------
# 한글 폰트 및 LaTeX 수식 설정 (예: Windows '맑은 고딕')
# -----------------------------------------------------------------------------
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.default'] = 'regular'  # 수식(Text) 표현을 위해

# -----------------------------------------------------------------------------
# [A] 시설유형 매핑 및 기준표
# -----------------------------------------------------------------------------
facility_type_mapping = {
    "가산A1타워주차장": "실내주차장",
    "에이샵스크린골프": "실내체육시설",
    "영통역대합실": "다중이용시설",
    "영통역지하역사": "다중이용시설",
    "이든어린이집": "어린이집",
    "좋은이웃데이케어센터": "노인요양시설",
    "좋은이웃데이케어센터1": "노인요양시설",
    "좋은이웃데이케어센터2": "노인요양시설",
    "좋은이웃데이케어센터3": "노인요양시설",
    "좋은이웃데이케어센터4": "노인요양시설",
    "하이씨앤씨학원": "다중이용시설"
}

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
        "pm10": 75,
        "pm25": 35,
        "co2": 1000,
        "voc": 400,
        "hcho": 80,
        "co": 10,
        "no2": 50
    },
    "실내주차장": {
        "pm10": 200,
        "hcho": 100,
        "co": 25,
        "no2": 300,
        "co2": 1000,
        "voc": 1000
    },
    "실내체육시설": {
        "pm10": 200,
        # 필요시 다른 물질 기준 추가
    }
}

# -----------------------------------------------------------------------------
# [B] CSV 불러오기 (예시)
# -----------------------------------------------------------------------------
ob1_min = pd.read_csv("C:/muf/input/가산A1타워주차장_20230331.csv", encoding='utf-8')
ob2_min = pd.read_csv("C:/muf/input/에이샵스크린골프_20230331.csv", encoding='utf-8')
ob3_min = pd.read_csv("C:/muf/input/영통역대합실_20230331.csv", encoding='utf-8')
ob41_min = pd.read_csv("C:/muf/input/영통역지하역사_20230331.csv", encoding='utf-8')
ob42_min = pd.read_csv("C:/muf/input/이든어린이집_20230331.csv", encoding='utf-8')
ob51_min = pd.read_csv("C:/muf/input/좋은이웃데이케어센터1_20230331.csv", encoding='utf-8')
ob52_min = pd.read_csv("C:/muf/input/좋은이웃데이케어센터2_20230331.csv", encoding='utf-8')
ob53_min = pd.read_csv("C:/muf/input/좋은이웃데이케어센터3_20230331.csv", encoding='utf-8')
ob54_min = pd.read_csv("C:/muf/input/좋은이웃데이케어센터4_20230331.csv", encoding='utf-8')
ob6_min = pd.read_csv("C:/muf/input/하이씨앤씨학원_20230331.csv", encoding='utf-8')

# 필요하다면 다른 CSV도 추가로 로드:


# -----------------------------------------------------------------------------
# [C] 보조함수
# -----------------------------------------------------------------------------
def get_facility_type(place_name: str) -> str:
    """
    장소명 -> 시설유형(예: '노인요양시설') 반환.
    '좋은이웃데이케어센터1' ~ '좋은이웃데이케어센터4'처럼 숫자 붙는 경우도 처리.
    """
    if "좋은이웃데이케어센터" in place_name:
        return facility_type_mapping.get("좋은이웃데이케어센터", None)
    return facility_type_mapping.get(place_name, None)

def get_standard_line(place_name: str, pollutant: str):
    """
    장소명 -> 시설유형 -> 기준표 -> 특정 물질(pollutant)의 기준농도
    
    반환값:
    - 기준이 있고 의미있는 값인 경우: 해당 기준값
    - 기준이 없거나 너무 큰 값(100000 초과)인 경우: None (기준선 미표시)
    """
    facility_type = get_facility_type(place_name)
    if not facility_type:
        return None
    
    facility_standards = standards_by_facility_type.get(facility_type, {})
    standard_value = facility_standards.get(pollutant, None)
    
    if standard_value is None or standard_value > 100000:
        return None
        
    return standard_value

def find_missing_periods(df, time_col='datetime'):
    """
    시계열 데이터에서 missing 구간(1시간에 60개 미만 or 아예 시간 자체가 없는 구간) 찾기
    반환값: [(start, end), (start, end), ...] 형태
    """
    if time_col not in df.columns:
        return []

    # 모든 시간별로 60개가 정상치라고 가정
    all_dates = pd.date_range(start=df[time_col].min(),
                              end=df[time_col].max(), freq='h')
    # "해당 시간에 60개 미만"인 경우
    incomplete_hours = df[time_col].value_counts().loc[lambda x: x < 60].index
    # 전체 시계열에서 누락된 시간 + 불충분(hour당 60개 미만) 시간 합집합
    missing_dates = pd.Index(all_dates).difference(df[time_col]).union(incomplete_hours)

    if missing_dates.empty:
        return []

    missing_dates = missing_dates.to_series().sort_values()
    # 1시간 초과로 이어지는 구간은 하나의 interval로 묶음
    missing_periods = []
    for _, group in missing_dates.groupby((missing_dates.diff().dt.total_seconds() > 3600).cumsum()):
        missing_periods.append((group.iloc[0], group.iloc[-1]))
    return missing_periods

# -----------------------------------------------------------------------------
# [D] 축 라벨(학술 표기) 매핑 딕셔너리 (LaTeX 수식 사용)
# -----------------------------------------------------------------------------
pollutant_label_dict = {
    "pm10": r"PM$_{10} (ug/m^3)$",
    "pm25": r"PM$_{2.5} (ug/m^3)$",
    "co2": r"CO$_2$ (ppm)",
    "voc": r"VOC $(ug/m^3)$",
    "temp": "Temperature(°C)",
    "humi": "Humidity(%)",
    "hcho": r"HCHO $(ug/m^3)$",
    "co": r"CO $(ppm)$",
    "no2": r"NO$_2$ $(ppb)$"
}

# -----------------------------------------------------------------------------
# [E] 통합 그래프 함수 (matplotlib만 활용)
# -----------------------------------------------------------------------------
def plot_air_quality_for_location(
    df: pd.DataFrame,
    place_name: str,
    pollutants: list,
    pollutant_scale_dict: dict,
    save_dir: str = "C:/muf/graph"
):
    """
    장소(place_name)에 대해, 각 물질(pollutants)별로
    1) 시간 시리즈 그래프(time series)
    2) 시간별 박스플롯(hourly boxplot)
    3) (연-월 표시) 월별 박스플롯
    를 그려서 각각 PNG 파일로 저장.

    - 0 이하값(로그 스케일 사용 시 문제 발생)을 NaN으로 대체
    - 축 라벨은 pollutant_label_dict를 사용 (학술 표기: LaTeX)
    """
    # 결과 저장 폴더 없으면 생성
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # datetime 변환
    df['datetime'] = pd.to_datetime(
        df['tmfc_d'].astype(str) + df['tmfc_h'].astype(str).str.zfill(2),
        format='%Y%m%d%H'
    )
    # tmfc_h를 24시간(0~23) 범위 정리
    df["tmfc_h"] = df["tmfc_h"].astype(int) % 24

    # 연-월(YYYY-MM) 정보를 담은 열 추가
    df['year_month'] = df['datetime'].dt.strftime('%Y-%m')

    for pollutant in pollutants:
        if pollutant not in df.columns:
            print(f"Warning: [{place_name}]에 {pollutant} 데이터가 없습니다.")
            continue

        # y축 스케일 설정
        scale_y = pollutant_scale_dict.get(pollutant, "linear")

        # (로그/시밀로그) 0 이하값을 NaN으로 치환
        if scale_y in ["log", "symlog"]:
            df.loc[df[pollutant] <= 0, pollutant] = np.nan

        # y축 라벨(LaTeX 학술 표기)
        y_label = pollutant_label_dict.get(pollutant, pollutant)

        # ============= A. 시간 시리즈 그래프 =============
        plt.figure(figsize=(10, 6))
        plt.title(f"[{place_name}] {y_label} Time Series", fontsize=14)

        # 시간 시리즈 산점도
        plt.plot(df['datetime'], df[pollutant], 'o', markersize=2)
        plt.ylabel(y_label, fontsize=10)
        plt.yscale(scale_y)

        # 기준선
        cut_conc = get_standard_line(place_name, pollutant)
        if cut_conc is not None:
            plt.axhline(y=cut_conc, linestyle='--')

        # 누락 구간 표시 (다만 색상 지정은 제거)
        missing_periods = find_missing_periods(df)
        for (start, end) in missing_periods:
            if start == end:
                # 단일 시점 누락 -> 작은 점 표시
                plt.scatter(start, 0, s=10)
            else:
                # 구간 누락 -> 수평선 표시
                plt.hlines(y=0, xmin=start, xmax=end)

        plt.tight_layout()
        plt.savefig(f"{save_dir}/time_series_{place_name}_{pollutant}.png", dpi=300)
        plt.close()

        # ============= B. 시간별 박스플롯 =============
        # 시간대(hour)별로 자료를 나누어 boxplot
        plt.figure(figsize=(9, 6))
        plt.title(f"[{place_name}] {y_label} Hourly Boxplot", fontsize=14)

        # 시간 목록(정렬)
        hours_sorted = sorted(df["tmfc_h"].unique())
        # 시간대별 그룹
        grouped_data_hour = [
            df.loc[df["tmfc_h"] == h, pollutant].dropna() for h in hours_sorted
        ]

        # 박스플롯
        # positions를 1부터 시작하도록, x축에 시간 표시
        plt.boxplot(grouped_data_hour, positions=range(1, len(hours_sorted)+1))
        plt.xlabel("Hour of Day")
        plt.ylabel(y_label)
        plt.yscale(scale_y)
        plt.xticks(range(1, len(hours_sorted)+1), [str(h) for h in hours_sorted])

        # 기준선
        if cut_conc is not None:
            plt.axhline(y=cut_conc, linestyle='--')

        # 시간대별 평균 찍기
        means_hourly = [g.mean() for g in grouped_data_hour]
        plt.scatter(range(1, len(hours_sorted)+1), means_hourly)

        plt.tight_layout()
        plt.savefig(f"{save_dir}/hourly_boxplot_{place_name}_{pollutant}.png", dpi=300)
        plt.close()

        # ============= C. (연-월 표시) 월별 박스플롯 =============
        # pollutant 값이 NaN이 아닌 레코드만 추출
        df_nonan = df.dropna(subset=[pollutant]).copy()

        # 실제 데이터가 있는 연-월만 추림
        valid_year_months = sorted(df_nonan["year_month"].unique())
        if len(valid_year_months) == 0:
            print(f"Note: [{place_name}] {pollutant} 데이터가 존재하지 않아 월별 박스플롯을 건너뜁니다.")
            continue

        # 연-월별 그룹
        plt.figure(figsize=(9, 6))
        plt.title(f"[{place_name}] {y_label} Monthly Boxplot", fontsize=14)

        # 각 연-월에 해당하는 값 리스트 생성
        grouped_data_month = [
            df_nonan.loc[df_nonan["year_month"] == ym, pollutant].dropna()
            for ym in valid_year_months
        ]

        # 박스플롯
        plt.boxplot(grouped_data_month, positions=range(1, len(valid_year_months)+1))
        plt.xlabel("Year-Month")
        plt.ylabel(y_label)
        plt.yscale(scale_y)
        plt.xticks(range(1, len(valid_year_months)+1), valid_year_months, rotation=45)

        # 기준선
        if cut_conc is not None:
            plt.axhline(y=cut_conc, linestyle='--')

        # 연-월별 평균(점 표시)
        means_monthly = [arr.mean() for arr in grouped_data_month]
        plt.scatter(range(1, len(valid_year_months)+1), means_monthly)

        plt.tight_layout()
        plt.savefig(f"{save_dir}/monthly_boxplot_{place_name}_{pollutant}.png", dpi=300)
        plt.close()

# -----------------------------------------------------------------------------
# [F] 실제 호출부 (예시)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # 그릴 물질 리스트
    target_pollutants = ["pm10", "pm25", "co2", "voc", "temp", "humi", "hcho", "co", "no2"]

    # 물질별 y축 스케일
    pollutant_scale_dict = {
        "pm10": "symlog",
        "pm25": "symlog",
        "co2": "linear",
        "voc": "symlog",
        "temp": "linear",
        "humi": "linear",
        "hcho": "symlog",
        "co": "symlog",
        "no2": "symlog"
    }

    # "좋은이웃데이케어센터1" CSV 파일에 대해 그래프 생성
    plot_air_quality_for_location(
        ob1_min,
        place_name="좋은이웃데이케어센터1",
        pollutants=target_pollutants,
        pollutant_scale_dict=pollutant_scale_dict,
        save_dir="C:/muf/result/graph/좋은이웃1"
    )

    # 필요시 다른 시설도 동일 패턴 호출 예:
    # plot_air_quality_for_location(ob2_min, "에이샵스크린골프", target_pollutants, pollutant_scale_dict, "C:/muf/result/graph/에이샵")
    # plot_air_quality_for_location(ob3_min, "영통역대합실", target_pollutants, pollutant_scale_dict, "C:/muf/result/graph/영통역")

    print("모든 그래프 생성 완료!")
