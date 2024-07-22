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




def graph(ob1_min, title_name):
    # Convert tmfc_d and tmfc_h to datetime format
    ob1_min['datetime'] = pd.to_datetime(ob1_min['tmfc_d'].astype(str) + ob1_min['tmfc_h'].astype(str).str.zfill(2), format='%Y%m%d%H')
    ob1_min['log_pm10'] = np.log(ob1_min['pm10'])

    models_one = ["pm10", "pm25", "pm1", "humi", "temp", "hcho", "co", "no2", "rn", "voc", "co2", "tab"]
    missing_data_ratios = {}
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

        missing_data_ratios[model_one] = compute_missing_data_ratio(ob1_min)

graph(ob1_min, "(가산A1타워주차장)")
graph(ob2_min, "(에이샵스크린골프)")
graph(ob3_min, "(영통역대합실)")
graph(ob4_min, "(영통역지하역사)")
graph(ob5_min, "(이든어린이집)")
graph(ob6_min, "(좋은이웃데이케어센터1)")
graph(ob7_min, "(좋은이웃데이케어센터2)")
graph(ob8_min, "(좋은이웃데이케어센터3)")
graph(ob9_min, "(좋은이웃데이케어센터4)")
graph(ob10_min, "(하이씨앤씨학원)")

# Put your datasets into a dictionary for easier access
datasets = {
    "가산A1타워주차장": ob1_min,
    "에이샵스크린골프": ob2_min,
    "영통역대합실": ob3_min,
    "영통역지하역사": ob4_min,
    "이든어린이집": ob5_min,
    "좋은이웃데이케어센터1": ob6_min,
    "좋은이웃데이케어센터2": ob7_min,
    "좋은이웃데이케어센터3": ob8_min,
    "좋은이웃데이케어센터4": ob9_min,
    "하이씨앤씨학원": ob10_min,
}
header_list = ["pm10", "pm25", "pm1", "humi", "temp", "hcho", "co", "no2", "rn", "voc", "co2", "tab"]

results = {}
overall_results = {}

for header in header_list:
    column_data = df[header]
        means = []
        std_devs = []

        for place in df.index.unique():
            place_data = df.loc[place, column]
            # Exclude the place if all measurements for the substance are 0
            print(place_data)
            if not (place_data == 0).all():
                mean = place_data.mean()
                std_dev = place_data.std()
                means.append(mean)
                std_devs.append(std_dev)

        results[column] = list(zip(means, std_devs))

        # Calculate overall mean and std_dev only if there are any values
        if means and std_devs:
            overall_mean = np.mean(means)
            overall_std_dev = np.std(std_devs)
            overall_results[column] = (overall_mean, overall_std_dev)

# Create DataFrame for each place
df_results = pd.DataFrame(results, index=datasets.keys())

# Convert values to "mean ± std_dev" format
df_results = df_results.applymap(lambda x: f"{x[0]:.2f} ± {x[1]:.2f}" if isinstance(x, tuple) else x)

# Create DataFrame for overall results
df_overall = pd.DataFrame(overall_results, index=['Overall'])
df_overall = df_overall.applymap(lambda x: f"{x[0]:.2f} ± {x[1]:.2f}" if isinstance(x, tuple) else x)

# Append overall results to the main DataFrame
df_results = pd.concat([df_results, df_overall])

# Save to Excel
df_results.to_excel('output_file.xlsx')