import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
from matplotlib.ticker import FuncFormatter

korean_font = fm.FontProperties(family='Batang')

# Read CSV files
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
dataframes = [self_under_station, ob3_min, ob4_min, self_academy, ob10_min, self_daycare, ob5_min, ob6_min, ob7_min,
              ob8_min, ob9_min, self_inparking, ob1_min, self_insports, ob2_min]
titles = ["기존측정\n(지하역사)", "영통역\n대합실", "영통역\n지하역사", "기존측정\n(학원)", "하이씨앤씨\n학원", "기존측정\n(어린이집)", "이든어린이집", "좋은이웃\n데이케어센터1",
          "좋은이웃\n데이케어센터2", "좋은이웃\n데이케어센터3", "좋은이웃\n데이케어센터4",
          "기존측정\n(실내주차장)", "가산A1\n타워주차장", "기존측정\n(실내체육시설)", "에이샵\n스크린골프"]

# List all the pollutants
pollutants = ["pm10", "co2", "hcho", "tab", "co", "pm25", "no2", "rn", "voc", 'humi', 'temp']

# Create a color palette for the boxplots
colors = ['blue' if '기존측정' in title else 'red' for title in titles]

def formatter(y, pos):
    return '{:,.0f}'.format(y)


# Iterate through each pollutant
for pollutant in pollutants:
    # Set parameters based on the pollutant
    if pollutant == 'pm10':
        model_name_4th, model_name_label = "PM10", "ug/m³"
        cut_conc1, cut_conc2, cut_conc3, cut_conc4 = 100, 75, 200, 200
        col1, col2, col3, col4 = 'red', 'red', 'red', 'red'
        scale_y, scale_name, mul = "log", "log", 1
    elif pollutant == 'pm25':
        model_name_4th, model_name_label = "PM2.5", "ug/m³"
        cut_conc1, cut_conc2, cut_conc3, cut_conc4 = 50, 35, 0, 0
        col1, col2, col3, col4 = 'red', 'red', 'None', 'None'
        scale_y, scale_name, mul = "log", "log", 1
    elif pollutant == 'hcho':
        model_name_4th, model_name_label = "HCHO", "ug/m³"
        cut_conc1, cut_conc2, cut_conc3, cut_conc4 = 100, 80, 100, 0
        col1, col2, col3, col4 = 'red', 'red', 'red', 'None'
        scale_y, scale_name, mul = "log", "log", 1
    elif pollutant == 'co':
        model_name_4th, model_name_label = "CO", "ppm"
        cut_conc1, cut_conc2, cut_conc3, cut_conc4 = 10, 10, 25, 0
        col1, col2, col3, col4 = 'red', 'red', 'red', 'None'
        scale_y, scale_name, mul = "log", "log", 1
    elif pollutant == 'no2':
        model_name_4th, model_name_label = "NO2", "ppm"
        cut_conc1, cut_conc2, cut_conc3, cut_conc4 = 0.1, 0.05, 0.30, 10
        col1, col2, col3, col4 = 'red', 'red', 'red', 'None'
        scale_y, scale_name, mul = "log", "log", 0.001
    elif pollutant == 'rn':
        model_name_4th, model_name_label = "Rn", "Bq/m³"
        cut_conc1, cut_conc2, cut_conc3, cut_conc4 = 148, 148, 148, 0
        col1, col2, col3, col4 = 'red', 'red', 'red', 'None'
        scale_y, scale_name, mul = "log", "log", 1
    elif pollutant == 'voc':
        model_name_4th, model_name_label = "VOC", "ug/m³"
        cut_conc1, cut_conc2, cut_conc3, cut_conc4 = 500, 400, 1000, 0
        col1, col2, col3, col4 = 'red', 'red', 'red', 'None'
        scale_y, scale_name, mul = "log", "log", 1
    elif pollutant == 'co2':
        model_name_4th, model_name_label = "CO2", "ppm"
        cut_conc1, cut_conc2, cut_conc3, cut_conc4 = 1000, 1000, 1000, 0
        col1, col2, col3, col4 = 'red', 'red', 'red', 'None'
        scale_y, scale_name, mul = "log", "log", 1
    elif pollutant == 'tab':
        model_name_4th, model_name_label = "TAB", "CFU/m³"
        cut_conc1, cut_conc2, cut_conc3, cut_conc4 = 0, 800, 0, 0
        col1, col2, col3, col4 = 'None', 'red', 'None', 'None'
        scale_y, scale_name, mul = "log", "log", 1
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
        plu = ""
    elif pollutant == 'temp':
        model_name_4th = "Temperature"
        model_name_label = "°C"
        cut_conc1 = 10
        cut_conc2 = 10
        cut_conc3 = 10
        cut_conc4 = 0
        col1 = 'None'
        col2 = 'None'
        col3 = 'None'
        col4 = 'None'
        scale_y = "linear"
        scale_name = "linear"
        mul = 1
        plu = ""

    else:
        continue

    # Prepare a dictionary to hold the data for this pollutant from all sites
    data_all_sites = {}
    all_na = True
    min_value = float('inf')

    for data in data_all_sites.values():
        positive_data = data[data > 0]
        if not positive_data.empty:
            min_value = min(min_value, positive_data.min())

    # 모든 값이 0 이하인 경우 기본값 설정
    if min_value == float('inf'):
        min_value = 0.1

    # Iterate through each site
    for df, title in zip(dataframes, titles):
        if pollutant in df.columns and not df[pollutant].dropna().empty:
            data = (df[pollutant].dropna()) * mul
            if not data.empty:
                all_na = False
                data_all_sites[title] = data
                min_value = min(min_value, data[data > 0].min())
        else:
            data_all_sites[title] = pd.Series([])  # 빈 시리즈 추가

    if all_na:
        print(f"No data for pollutant {pollutant}. Skipping.")
        continue

    # Create a new figure
    fig, ax = plt.subplots(figsize=(16, 10))

    # Create the boxplot
    boxplot = sns.boxplot(data=pd.DataFrame(data_all_sites), palette=colors, ax=ax)

    # 수염 색상을 검은색으로 설정
    for i, artist in enumerate(boxplot.artists):
        # 박스 색상 설정
        artist.set_facecolor(colors[i])
        # 수염과 중앙선 색상을 검은색으로 설정
        for j in range(i*6, i*6+6):
            boxplot.lines[j].set_color('black')


    ax.set_yscale("log")  # Changed from "log" to "log"
    ax.set_ylim(bottom=min_value)  # Set lower limit to exclude negative values

    plt.plot([-0.5, 4.5], [cut_conc1, cut_conc1], color=col1, linestyle='--')
    plt.plot([4.5, 10.5], [cut_conc2, cut_conc2], color=col2, linestyle='--')
    plt.plot([10.5, 12.5], [cut_conc3, cut_conc3], color=col3, linestyle='--')
    plt.plot([12.5, 14.5], [cut_conc4, cut_conc4], color=col4, linestyle='--')
    plt.yscale("log")

    # Set the x-tick labels to be the site names
    ax.set_xticks(range(len(titles)))
    ax.set_xticklabels(titles, fontproperties=korean_font, fontweight='bold', rotation=45, ha='right')

    # Add a title and labels
    plt.ylabel(model_name_label, size=12)
    ax.set_title(f'Boxplot of {model_name_4th} concentrations at different sites')

    # Set the minimum y-value
    if min_value != float('inf'):
        ax.set_ylim([min_value*0.2, ax.get_ylim()[1]])

    # x틱 색상 기존측정소=레드, 나머지 블랙
    # colorsk = ['red', 'black', 'black', 'red', 'black', 'red', 'black', 'black', 'black', 'black', 'black', 'red',
    #            'black', 'red', 'black']

    # x틱 색상 모두블랙
    colorsk = ['black', 'black', 'black', 'black', 'black', 'black', 'black', 'black', 'black', 'black', 'black', 'black',
               'black', 'black', 'black']


    for label, color in zip(ax.get_xticklabels(), colorsk):
        label.set_color(color)

    plt.tight_layout()
    # Save the plot
    plt.savefig(f'C:\\muf\\graph\\{pollutant}_boxplot.png', dpi=600)
    plt.close()

print("All plots have been generated and saved.")