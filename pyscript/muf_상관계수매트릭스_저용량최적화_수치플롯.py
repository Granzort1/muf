import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.font_manager as fm
from scipy.stats import norm
import subprocess
import sys
import io
import os
import shutil

# OptiPNG 경로를 직접 지정 (필요시 실제 경로로 수정)
OPTIPNG_PATH = "C:/Tool/optipng-0.7.7-win32/optipng.exe"  # 예시 경로
# OptiPNG 활성화 여부 (False로 설정하면 최적화 건너뜀)
ENABLE_OPTIMIZATION = False

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

korean_font = fm.FontProperties(family='Batang')

# 물질명 학술 영문 매핑
scientific_names = {
    'pm10': 'PM₁₀',
    'pm25': 'PM₂.₅',
    'pm1': 'PM₁',  # 제외 예정
    'humi': 'Relative\nHumidity',
    'temp': 'Temperature',
    'hcho': 'HCHO',
    'co': 'CO',
    'no2': 'NO₂',
    'rn': 'Radon',  # 제외 예정
    'voc': 'VOCs',
    'co2': 'CO₂',
    'tab': 'TAB'  # 제외 예정
}

# 상관계수 매트릭스에서 제외할 물질 목록
excluded_columns = ['tab', 'pm1', 'rn']  # TAB, PM1, Radon 제외

# 시설명 영문 매핑
facility_names = {
    "가산A1타워주차장": "Gasan A1 Tower Parking Lot",
    "에이샵스크린골프": "A-Shop Screen Golf",
    "영통역대합실": "Yeongtong Station",  # 영통역대합실과 영통역지하역사는 함께 평균을 낼 것임
    "영통역지하역사": "Yeongtong Station",
    "이든어린이집": "Eden Daycare",
    "좋은이웃데이케어센터1": "Good Neighbor Daycare Center",  # 4개 데이케어센터는 함께 평균을 낼 것임
    "좋은이웃데이케어센터2": "Good Neighbor Daycare Center",
    "좋은이웃데이케어센터3": "Good Neighbor Daycare Center",
    "좋은이웃데이케어센터4": "Good Neighbor Daycare Center",
    "하이씨앤씨학원": "Hi C&C Academy"
}

# 그래프 타이틀 수동 설정
custom_titles = {
    "Gasan A1 Tower Parking Lot": "Correlation Analysis: Underground Parking Facility",
    "A-Shop Screen Golf": "Correlation Analysis: Indoor Golf Simulation Facility",
    "Yeongtong Station": "Correlation Analysis: Subway Station",
    "Eden Daycare": "Correlation Analysis: Childcare Center",
    "Good Neighbor Daycare Center": "Correlation Analysis: Daycare Center",
    "Hi C&C Academy": "Correlation Analysis: Educational Facility"
}

# 파일명 목록
filenames = ["가산A1타워주차장", "에이샵스크린골프", "영통역대합실", "영통역지하역사", "이든어린이집",
             "좋은이웃데이케어센터1", "좋은이웃데이케어센터2", "좋은이웃데이케어센터3",
             "좋은이웃데이케어센터4", "하이씨앤씨학원"]

# CSV 파일 불러오기
dataframes = {}
for filename in filenames:
    dataframes[filename] = pd.read_csv(f"C:/muf/input2/{filename}_20230331.csv", encoding='utf-8')

def encode_filename(filename):
    return filename.encode('utf-8').decode('utf-8', errors='ignore')

def optimize_png(filename):
    # OptiPNG 비활성화 상태라면 바로 리턴
    if not ENABLE_OPTIMIZATION:
        print(f"이미지 최적화 비활성화 상태: {filename}")
        return
        
    # 최적화 시도
    try:
        # OptiPNG 경로 확인
        optipng_cmd = OPTIPNG_PATH if os.path.exists(OPTIPNG_PATH) else 'optipng'
        
        # OptiPNG가 설치되어 있는지 확인
        if shutil.which(optipng_cmd) is not None:
            unicode_filename = os.path.abspath(filename)
            subprocess.run([optipng_cmd, '-o7', unicode_filename], check=False)
            print(f"이미지 최적화 완료: {filename}")
        else:
            print(f"OptiPNG가 설치되어 있지 않아 최적화를 건너뜁니다: {filename}")
    except Exception as e:
        print(f"이미지 최적화 중 오류 발생 (무시하고 계속 진행): {e}")
        # 오류가 발생해도 프로그램은 계속 실행

def create_correlation_matrix(df):
    # 상관계수 매트릭스 계산 (제외할 열 제거 후)
    df_filtered = df.drop(columns=excluded_columns, errors='ignore')  # 해당 열이 없으면 무시
    return df_filtered.corr()

def plot_correlation_matrix(corr_matrix, facility_name, custom_title=None):
    plt.figure(figsize=(14, 12))
    
    # 히트맵 생성 (빨간색에서 파란색으로 변하는 색상 맵 사용)
    cmap = sns.diverging_palette(230, 20, as_cmap=True)  # 파란색(-1)에서 빨간색(1)으로
    
    # 히트맵 그리기 (vmin, vmax 설정으로 -1부터 1까지 표시)
    ax = sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap=cmap, center=0,
                square=True, linewidths=0.5, 
                vmin=-1, vmax=1,  # 명시적으로 -1부터 1까지 범위 설정
                cbar_kws={"shrink": 1.0,  # 레전드 높이 최대화
                          "fraction": 0.05,  # 레전드 폭 조정
                          "aspect": 30,  # 레전드 종횡비
                          "pad": 0.01,  # 매트릭스와 레전드 사이 간격
                          "ticks": [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]},  # 컬러바 눈금 추가
                annot_kws={"size": 12})  # 상관계수 숫자 크기 증가
    
    # 컬러바 레이블 설정
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12, size=6, pad=5, length=5)  # 컬러바 레이블 크기와 위치 조정
    
    # x, y 축 레이블 학술명으로 변경 (회전 제거 & 글자 크기 증가)
    plt.xticks(np.arange(len(corr_matrix.columns)) + 0.5, 
               [scientific_names.get(col, col) for col in corr_matrix.columns], 
               rotation=0, fontsize=14)
    plt.yticks(np.arange(len(corr_matrix.index)) + 0.5, 
               [scientific_names.get(idx, idx) for idx in corr_matrix.index], 
               rotation=0, fontsize=14)
    
    # 제목 설정 (수동 타이틀 또는 기본 타이틀 사용)
    title_text = custom_title if custom_title else f"Correlation Matrix: {facility_name}"
    plt.title(title_text, fontsize=18)
    
    # 그래프 여백 최소화
    plt.tight_layout(pad=0.5)  # 패딩 값을 줄여 여백 최소화
    
    # 그래프 저장 (파일명은 facility_name 기준으로 유지)
    plt.savefig(f"C:/muf/result/{facility_name.replace(' ', '_')}_correlation_matrix.png", dpi=350, bbox_inches='tight')
    
    # 이미지 최적화 (주석 처리하여 비활성화)
    optimize_png(f"C:/muf/result/{facility_name.replace(' ', '_')}_correlation_matrix.png")
    
    plt.close()

# 그룹별 평균 상관관계 행렬 계산 및 플롯 생성
def process_data():
    # 데이케어센터 4개 파일의 평균 상관관계 계산
    daycare_files = ["좋은이웃데이케어센터1", "좋은이웃데이케어센터2", "좋은이웃데이케어센터3", "좋은이웃데이케어센터4"]
    daycare_corrs = [create_correlation_matrix(dataframes[file]) for file in daycare_files]
    daycare_avg_corr = sum(daycare_corrs) / len(daycare_corrs)
    facility_name = "Daycare Center"
    plot_correlation_matrix(daycare_avg_corr, facility_name, custom_titles.get(facility_name))
    
    # 영통역 관련 2개 파일의 평균 상관관계 계산
    station_files = ["영통역대합실", "영통역지하역사"]
    station_corrs = [create_correlation_matrix(dataframes[file]) for file in station_files]
    station_avg_corr = sum(station_corrs) / len(station_corrs)
    facility_name = "Subway Station"
    plot_correlation_matrix(station_avg_corr, facility_name, custom_titles.get(facility_name))
    
    # 나머지 개별 파일들 처리
    individual_files = ["가산A1타워주차장", "에이샵스크린골프", "이든어린이집", "하이씨앤씨학원"]
    for file in individual_files:
        corr_matrix = create_correlation_matrix(dataframes[file])
        facility_name = facility_names[file]
        plot_correlation_matrix(corr_matrix, facility_name, custom_titles.get(facility_name))

# 실행
process_data()

