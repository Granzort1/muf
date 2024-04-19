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
import re



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
            cut_conc = 148
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




# First, let's put your data in a list and map each to its corresponding location
locations = ['가산A1타워주차장', '에이샵스크린골프', '영통역대합실', '영통역지하역사', '이든어린이집', '좋은이웃데이케어센터1', '좋은이웃데이케어센터2', '좋은이웃데이케어센터3', '좋은이웃데이케어센터4', '하이씨앤씨학원']
datasets = [ob1_min, ob2_min, ob3_min, ob4_min, ob5_min, ob6_min, ob7_min, ob8_min, ob9_min, ob10_min]

location_data_mapping = dict(zip(locations, datasets))

# Models
models = ["pm10", "pm25", "pm1", "humi", "temp", "hcho", "co", "no2", "rn", "voc", "co2", "tab"]

# Initialize a dataframe to store all the results
missing_data_ratios_df = pd.DataFrame(columns=locations, index=models)
# Initialize three dataframes to store all the results
missing_data_ratios_df = pd.DataFrame(columns=locations, index=models)
total_data_points_df = pd.DataFrame(columns=locations, index=models)
missing_data_points_df = pd.DataFrame(columns=locations, index=models)

# Loop through each dataset
for location, data in location_data_mapping.items():
    for model in models:
        missing_data_ratio, missing_data_points = compute_missing_data_ratio(data)

        # Calculate total data points
        total_data_points = data.shape[0] + missing_data_points

        # Store calculated values in the corresponding dataframes
        missing_data_ratios_df.loc[model, location] = missing_data_ratio
        total_data_points_df.loc[model, location] = total_data_points
        missing_data_points_df.loc[model, location] = missing_data_points

# Export dataframes to Excel files
# missing_data_ratios_df.to_excel("C:/muf/missing_data_ratios.xlsx")
# total_data_points_df.to_excel("C:/muf/total_data_points.xlsx")
# missing_data_points_df.to_excel("C:/muf/missing_data_points.xlsx")



# Initialize a dictionary to store dataframes for each location
data_count_per_hour_with_all_times = {}

# Loop through each dataset
for location, data in location_data_mapping.items():
    # Create separate columns for date and hour
    data['Date'] = data['datetime'].dt.date
    data['Hour'] = data['datetime'].dt.hour
    data['Datetime'] = pd.to_datetime(data['Date'].astype(str) + ' ' + data['Hour'].astype(str) + ':00')

    # Group by date and hour and count the number of data points
    data_count = data.groupby(['Datetime']).size().reset_index(name='Count')

    # Create a full timeseries index from the start to end of your data, with 1 hour frequency
    full_index = pd.date_range(start=data['Datetime'].min(), end=data['Datetime'].max(), freq='H')

    # Reindex your data_count to the full_index, and fill missing data points with 'NULL'
    data_count_reindexed = data_count.set_index('Datetime').reindex(full_index).fillna('NULL').reset_index()

    # Store dataframe in dictionary
    data_count_per_hour_with_all_times[location] = data_count_reindexed

# Export each dataframe to an Excel file
for location, data_count in data_count_per_hour_with_all_times.items():
    # Replace any non-alphanumeric characters in the location name with an underscore for the filename
    filename = re.sub(r'\W+', '_', location)

    # Export dataframe to an Excel file
    data_count.to_excel(f"C:/muf/data_count_per_hour_with_all_times_{filename}.xlsx", index=False)