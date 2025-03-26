import pandas as pd
import numpy as np
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



korean_font = fm.FontProperties(family='Batang')
# Read CSV
ob1_min = pd.read_csv("C:/muf/가산A1타워주차장_20230331.csv", encoding='utf-8')
ob2_min = pd.read_csv("C:/muf/에이샵스크린골프_20230331.csv", encoding='utf-8')
ob3_min = pd.read_csv("C:/muf/영통역대합실_20230331.csv", encoding='utf-8')
ob4_min = pd.read_csv("C:/muf/영통역지하역사_20230331.csv", encoding='utf-8')
ob5_min = pd.read_csv("C:/muf/이든어린이집_20230331.csv", encoding='utf-8')
ob6_min = pd.read_csv("C:/muf/좋은이웃데이케어센터1_20230331.csv", encoding='utf-8')
ob7_min = pd.read_csv("C:/muf/좋은이웃데이케어센터2_20230331.csv", encoding='utf-8')
ob8_min = pd.read_csv("C:/muf/좋은이웃데이케어센터3_20230331.csv", encoding='utf-8')
ob9_min = pd.read_csv("C:/muf/좋은이웃데이케어센터4_20230331.csv", encoding='utf-8')
ob10_min = pd.read_csv("C:/muf/하이씨앤씨학원_20230331.csv", encoding='utf-8')


def compute_missing_data_ratio(data):
    # Calculate total number of data points
    start_time = data['datetime'].min()
    end_time = data['datetime'].max()
    total_data_points = int((end_time - start_time).total_seconds() / 60) + 1

    # Calculate existing data points
    existing_data_points = data['datetime'].value_counts().sum()

    # Calculate missing data points and ratio
    missing_data_points = total_data_points - existing_data_points
    missing_data_ratio = missing_data_points / total_data_points

    return missing_data_ratio, missing_data_points

def count_data_per_day(data):
    # Convert 'datetime' to date
    data['date'] = data['datetime'].dt.date

    # Count the number of data per day
    daily_counts = data.groupby('date').count()

    return daily_counts



def graph(ob1_min, title_name):
    # Convert tmfc_d and tmfc_h to datetime format
    ob1_min['datetime'] = pd.to_datetime(ob1_min['tmfc_d'].astype(str) + ob1_min['tmfc_h'].astype(str).str.zfill(2), format='%Y%m%d%H')
    ob1_min['log_pm10'] = np.log(ob1_min['pm10'])

    models_one = ["pm10", "pm25", "pm1", "humi", "temp", "hcho", "co", "no2", "rn", "voc", "co2", "tab"]
    missing_data_ratios = {}
    daily_counts_ob1_min = count_data_per_day(ob1_min)
    print(daily_counts_ob1_min, f"{title_name}")
    for model_one in models_one:

        if model_one == 'pm10':
            model_name_4th = "PM10"
            model_name_label = "ug/m³"
            cut_conc = 200
            col = 'red'
            scale_y = "symlog"
            scale_name = "log"
        elif model_one == 'pm25':
            model_name_4th = "PM2.5"
            model_name_label = "ug/m³"
            cut_conc = 200
            col = 'red'
            scale_y = "symlog"
            scale_name = "log"
        elif model_one == 'pm1':
            model_name_4th = "PM1.0"
            model_name_label = "ug/m³"
            cut_conc = 200
            col = 'red'
            scale_y = "symlog"
            scale_name = "log"
        elif model_one == 'humi':
            model_name_4th = "Relative Humidity"
            model_name_label = "%"
            cut_conc = 100
            col = 'None'
            scale_y = "linear"
            scale_name = "linear"
        elif model_one == 'temp':
            model_name_4th = "Temperature"
            model_name_label = "°C"
            cut_conc = 10
            col = 'None'
            scale_y = "linear"
            scale_name = "linear"
        elif model_one == 'hcho':
            model_name_4th = "HCHO"
            model_name_label = "ug/m³"
            cut_conc = 100
            col = 'red'
            scale_y = "symlog"
            scale_name = "log"
        elif model_one == 'co':
            model_name_4th = "CO"
            model_name_label = "ppm"
            cut_conc = 25
            col = 'red'
            scale_y = "symlog"
            scale_name = "log"
        elif model_one == 'no2':
            model_name_4th = "NO2"
            model_name_label = "ppb"
            cut_conc = 300
            col = 'red'
            scale_y = "symlog"
            scale_name = "log"
        elif model_one == 'rn':
            model_name_4th = "Rn"
            model_name_label = "Bq/m³"
            cut_conc = 4
            col = 'red'
            scale_y = "symlog"
            scale_name = "log"
        elif model_one == 'voc':
            model_name_4th = "VOC"
            model_name_label = "ug/m³"
            cut_conc = 1000
            col = 'red'
            scale_y = "symlog"
            scale_name = "log"
        elif model_one == 'co2':
            model_name_4th = "CO2"
            model_name_label = "ppm"
            cut_conc = 1000
            col = 'red'
            scale_y = "symlog"
            scale_name = "log"
        elif model_one == 'tab':
            model_name_4th = "TAB"
            model_name_label = "CFU/m³"
            cut_conc = 100
            col = 'None'
            scale_y = "symlog"
            scale_name = "log"
        else:
            pass






graph(ob6_min, "(좋은이웃데이케어센터1)")
graph(ob7_min, "(좋은이웃데이케어센터2)")
graph(ob8_min, "(좋은이웃데이케어센터3)")
graph(ob9_min, "(좋은이웃데이케어센터4)")
graph(ob10_min, "(하이엔씨학원)")

# 결과를 저장하기 위한 빈 DataFrame 생성
df_results = pd.DataFrame()

# 각각의 파일에 대한 그래프 생성 및 데이터 수집
for data, title in zip([ob6_min, ob7_min, ob8_min, ob9_min, ob10_min],
                       ["좋은이웃데이케어센터1", "좋은이웃데이케어센터2", "좋은이웃데이케어센터3", "좋은이웃데이케어센터4", "하이엔씨학원"]):
    daily_counts = count_data_per_day(data)
    daily_counts["Location"] = title
    df_results = df_results.append(daily_counts)

# DataFrame을 Excel 파일로 저장
df_results.to_excel("C:\\muf\\daily_counts_results.xlsx")









