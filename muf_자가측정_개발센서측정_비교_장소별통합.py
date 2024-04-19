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
from matplotlib.ticker import FuncFormatter



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


self_inparking = pd.read_csv("C:\\muf\\standard_conc\\indoor_parking.csv", encoding='utf-8')
self_insports = pd.read_csv("C:\\muf\\standard_conc\\indoorsports.csv", encoding='utf-8')
self_under_station = pd.read_csv("C:\\muf\\standard_conc\\underground_station.csv", encoding='utf-8')
self_daycare = pd.read_csv("C:\\muf\\standard_conc\\daycarecenter.csv", encoding='utf-8')
self_academy = pd.read_csv("C:\\muf\\standard_conc\\academy.csv", encoding='utf-8')

# List all the dataframes and their corresponding titles

dataframes = [self_under_station, ob3_min, ob4_min, self_academy, ob10_min, self_daycare, ob5_min, ob6_min, ob7_min, ob8_min, ob9_min, self_inparking, ob1_min, self_insports, ob2_min]


titles = ["기존측정\n(지하역사)", "영통역\n대합실", "영통역\n지하역사", "기존측정\n(학원)", "하이씨앤씨\n학원", "기존측정\n(어린이집)", "이든어린이집", "좋은이웃\n데이케어센터1", "좋은이웃\n데이케어센터2", "좋은이웃\n데이케어센터3", "좋은이웃\n데이케어센터4",
          "기존측정\n(실내주차장)", "가산A1\n타워주차장", "기존측정\n(실내체육시설)", "에이샵\n스크린골프"]



# List all the pollutants
pollutants = ["pm1", "humi", "temp", "co2"]
pollutants2 = ["pm10", "co2", "hcho", "tab", "co", "pm25", "no2", "rn", "voc"]

def formatter(y, pos):
    return '{:,.0f}'.format(y)  # format the number as a decimal string

# Create a color palette for the boxplots. The first 5 will be one color (the reference sites), the rest another color.
colors = ['red', 'red', 'red', 'red', 'red', 'red', 'red', 'red', 'red', 'red', 'red', 'red', 'red', 'red', 'red']
# Iterate through each pollutant
for pollutant in pollutants2:

    if pollutant == 'pm10':
        model_name_4th = "PM10"
        model_name_label = "ug/m³"
        cut_conc1 = 100
        cut_conc2 = 75
        cut_conc3 = 200
        cut_conc4 = 200
        col1 = 'red'
        col2 = 'red'
        col3 = 'red'
        col4 = 'red'
        scale_y = "symlog"
        scale_name = "log"
        mul = 1
    elif pollutant == 'pm25':
        model_name_4th = "PM2.5"
        model_name_label = "ug/m³"
        cut_conc1 = 50
        cut_conc2 = 35
        cut_conc3 = 0
        cut_conc4 = 0
        col1 = 'red'
        col2 = 'red'
        col3 = 'None'
        col4 = 'None'
        scale_y = "symlog"
        scale_name = "log"
        mul = 1
    elif pollutant == 'pm1':
        model_name_4th = "PM1.0"
        model_name_label = "ug/m³"
        cut_conc1 = 0
        cut_conc2 = 0
        cut_conc3 = 0
        cut_conc4 = 0
        col1 = 'None'
        col2 = 'None'
        col3 = 'None'
        col4 = 'None'
        scale_y = "symlog"
        scale_name = "log"
        mul = 1
    elif pollutant == 'humi':
        model_name_4th = "Relative Humidity"
        model_name_label = "%"
        cut_conc1 = 90
        cut_conc2 = 90
        cut_conc3 = 90
        cut_conc4 = 90
        col1 = 'None'
        col2 = 'None'
        col3 = 'None'
        col4 = 'None'
        scale_y = "linear"
        scale_name = "linear"
        mul = 1
    elif pollutant == 'temp':
        model_name_4th = "Temperature"
        model_name_label = "°C"
        cut_conc1 = 10
        cut_conc2 = 10
        cut_conc3 = 10
        cut_conc4 = 10
        col1 = 'None'
        col2 = 'None'
        col3 = 'None'
        col4 = 'None'
        scale_y = "linear"
        scale_name = "linear"
        mul = 1
    elif pollutant == 'hcho':
        model_name_4th = "HCHO"
        model_name_label = "ug/m³"
        cut_conc1 = 100
        cut_conc2 = 80
        cut_conc3 = 100
        cut_conc4 = 0
        col1 = 'red'
        col2 = 'red'
        col3 = 'red'
        col4 = 'None'
        scale_y = "symlog"
        scale_name = "log"
        mul = 1
    elif pollutant == 'co':
        model_name_4th = "CO"
        model_name_label = "ppm"
        cut_conc1 = 10
        cut_conc2 = 10
        cut_conc3 = 25
        cut_conc4 = 0
        col1 = 'red'
        col2 = 'red'
        col3 = 'red'
        col4 = 'None'
        scale_y = "symlog"
        scale_name = "log"
        mul = 1
    elif pollutant == 'no2':
        model_name_4th = "NO2"
        model_name_label = "ppm"
        cut_conc1 = 0.1
        cut_conc2 = 0.05
        cut_conc3 = 0.30
        cut_conc4 = 10
        col1 = 'red'
        col2 = 'red'
        col3 = 'red'
        col4 = 'None'
        scale_y = "symlog"
        scale_name = "log"
        mul = 0.001
    elif pollutant == 'rn':
        model_name_4th = "Rn"
        model_name_label = "Bq/m³"
        cut_conc1 = 148
        cut_conc2 = 148
        cut_conc3 = 148
        cut_conc4 = 0
        col1 = 'red'
        col2 = 'red'
        col3 = 'red'
        col4 = 'None'
        scale_y = "symlog"
        scale_name = "log"
        mul = 1
    elif pollutant == 'voc':
        model_name_4th = "VOC"
        model_name_label = "ug/m³"
        cut_conc1 = 500
        cut_conc2 = 400
        cut_conc3 = 1000
        cut_conc4 = 0
        col1 = 'red'
        col2 = 'red'
        col3 = 'red'
        col4 = 'None'
        scale_y = "symlog"
        scale_name = "log"
        mul = 1
    elif pollutant == 'co2':
        model_name_4th = "CO2"
        model_name_label = "ppm"
        cut_conc1 = 1000
        cut_conc2 = 1000
        cut_conc3 = 1000
        cut_conc4 = 0
        col1 = 'red'
        col2 = 'red'
        col3 = 'red'
        col4 = 'None'
        scale_y = "symlog"
        scale_name = "log"
        mul = 1
    elif pollutant == 'tab':
        model_name_4th = "TAB"
        model_name_label = "CFU/m³"
        cut_conc1 = 0
        cut_conc2 = 800
        cut_conc3 = 0
        cut_conc4 = 0
        col1 = 'None'
        col2 = 'red'
        col3 = 'None'
        col4 = 'None'
        scale_y = "symlog"
        scale_name = "log"
        mul = 1
    else:
        pass

    # Prepare a list to hold the data for this pollutant from all sites
    data_all_sites = []
    all_na = True
    min_value = float('inf')
    # Iterate through each site
    for df in dataframes:
        # Drop missing values from this pollutant's column
        data = (df[pollutant].dropna())*mul

        # If there is any data for this site and pollutant, set all_na to False
        if not data.empty:
            all_na = False

        # Update the minimum value if necessary
        min_value = min(min_value, data.min())

        # Append the data to the list
        data_all_sites.append(data)

    # If all values for this pollutant at all sites are missing, print a message and continue to the next pollutant
    if all_na:
        print(f"No data for pollutant {pollutant}. Skipping.")
        continue

        # Create a new figure
    fig, ax = plt.subplots(figsize=(16, 10))

    # Create the boxplot
    sns.boxplot(data=data_all_sites, palette=colors, ax=ax)
    ax.set_yscale("symlog")
    plt.plot([-0.5, 4.5], [cut_conc1, cut_conc1], color=col1, linestyle='--')
    plt.plot([4.5, 10.5], [cut_conc2, cut_conc2], color=col2, linestyle='--')
    plt.plot([10.5, 12.5], [cut_conc3, cut_conc3], color=col3, linestyle='--')
    plt.plot([12.5, 14.5], [cut_conc4, cut_conc4], color=col4, linestyle='--')
    plt.yscale("symlog")
    # Set the x-tick labels to be the site names
    ax.set_xticks(range(len(titles)))
    ax.set_xticklabels(titles, fontproperties=korean_font, fontweight='bold')

    # Add a title
    plt.ylabel(model_name_label, size=12)
    ax.set_title(f'Boxplot of {model_name_4th} concentrations at different sites')

    # Set the minimum y-value to be the smallest value among all data minus 0.2
    ax.set_ylim([min_value - 0.2, ax.get_ylim()[1]])
    # Create a FuncFormatter object from the function
    # formatter = FuncFormatter(formatter)
    # cut_conc_values = [cut_conc1, cut_conc2, cut_conc3, cut_conc4]
    # # Use this formatter for the y-axis tick labels
    # ax.yaxis.set_major_formatter(formatter)
    # for cut_conc in cut_conc_values:
    #     plt.yticks(list(plt.yticks()[0]) + [cut_conc])
    #     ax = plt.gca()  # get current axes
    #     yticks = ax.get_yticks().tolist()
    #     yticks[-1] = f'{cut_conc:.2f}'
    #     ax.set_yticklabels(yticks)
    # # Create a FuncFormatter object from the function
    # formatter = FuncFormatter(formatter)
    #
    # # Use this formatter for the y-axis tick labels
    # ax.yaxis.set_major_formatter(formatter)

    # 정의된 색상 리스트
    colorsk = ['red', 'black', 'black', 'red', 'black', 'red', 'black', 'black', 'black', 'black', 'black', 'red', 'black', 'red', 'black']

    # 레이블들을 가져오기
    labels = ax.get_xticklabels()

    # 각 레이블의 색상을 변경하기
    for i, label in enumerate(labels):
        label.set_color(colorsk[i])

    plt.tight_layout()
    # Show the plot
    plt.savefig(f'C:\\muf\\graph\\{pollutant}_boxplot.png', dpi=600)
    plt.close()