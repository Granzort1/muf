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

    models_one = ["pm10", "pm25", "pm1", "humi", "temp", "hcho", "co", "no2", "rn", "voc", "co2", "tab"]
    missing_data_ratios = {}
    for model_one in models_one:
        if model_one in ['humi', 'temp']:
            model_name_4th = model_one.capitalize()
            model_name_label = "%" if model_one == 'humi' else "°C"
            cut_conc = 100 if model_one == 'humi' else 10
            col = 'None'
            scale_y = "linear"
            scale_name = "Linear"
        else:
            model_name_4th = model_one.upper()
            model_name_label = "μg/m³" if model_one in ['pm10', 'pm25', 'pm1', 'hcho', 'voc'] else "ppm" if model_one in ['co', 'co2'] else "ppb" if model_one == 'no2' else "Bq/m³" if model_one == 'rn' else "CFU/m³" if model_one == 'tab' else ""
            cut_conc = 100 if model_one in ['hcho', 'co', 'tab'] else 200 if model_one in ['pm10', 'pm25', 'pm1'] else 1000 if model_one in ['voc', 'co2'] else 300 if model_one == 'no2' else 148 if model_one == 'rn' else None
            col = 'red'
            scale_y = "linear" if model_one in ['humi', 'temp'] else "log"
            scale_name = "Logarithmic" if model_one not in ['humi', 'temp'] else "Linear"

        missing_data_ratios[model_one] = compute_missing_data_ratio(ob1_min)

        data_its = ob1_min[f"{model_one}"]

        # Time Series Plot
        plt.figure(figsize=(14, 7))
        if scale_y == "log":
            data_its_positive = data_its[data_its > 0]  # Exclude 0 and negative values
            datetime_positive = ob1_min['datetime'][data_its > 0]  # Filter datetime correspondingly

            plt.plot(datetime_positive, data_its_positive, 'o', markersize=1, label=f"{model_name_4th}", color='black',
                     markeredgewidth=0)
            plt.axhline(y=cut_conc, color=col, linestyle='--')

            plt.yscale('log')
            plt.ylabel(f"{model_name_label} (log₁₀ scale)")
            plt.title(f"Time Series Plot of {model_name_4th} (log₁₀ scale)")
        else:
            plt.plot(ob1_min['datetime'], data_its, 'o', markersize=1, label=f"{model_name_4th}", color='black', markeredgewidth=0)
            plt.axhline(y=cut_conc, color=col, linestyle='--')
            plt.ylabel(f"{model_name_label}")
            plt.title(f"Time Series Plot of {model_name_4th}")

        missing_data_ratio, _ = missing_data_ratios[model_one]
        plt.text(0.1, 0.9, 'Missing data ratio: {:.2%}'.format(missing_data_ratio), transform=plt.gca().transAxes)
        plt.xlabel('Date')

        bigger_dot = mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=3.7)
        plt.legend(handles=[bigger_dot], labels=[f"{model_name_4th}"], fontsize=8)

        all_dates = pd.date_range(start=ob1_min['datetime'].min(), end=ob1_min['datetime'].max(), freq='H')
        incomplete_hours = ob1_min['datetime'].value_counts().loc[lambda x: x < 60].index
        missing_dates = pd.Index(all_dates).difference(ob1_min['datetime']).union(incomplete_hours)

        if not missing_dates.empty:
            print(f'Missing dates for {model_one}: {missing_dates}')

        missing_dates = missing_dates.to_series().sort_values()
        missing_periods = [(g.iloc[0], g.iloc[-1]) for _, g in missing_dates.groupby((missing_dates.diff().dt.total_seconds() > 3600).cumsum())]

        for start, end in missing_periods:
            if start == end:
                plt.scatter(start, data_its_positive.min(), color='#2E8B57', s=1)
            else:
                plt.hlines(y=data_its_positive.min(), xmin=start, xmax=end, color='#2E8B57')

        for year in ob1_min['datetime'].dt.year.unique():
            for month in [2, 5, 8, 11]:
                if (ob1_min['datetime'] >= datetime(year, month, 1)).any() and (ob1_min['datetime'] < datetime(year, month + 1, 1)).any():
                    plt.axvline(x=datetime(year, month + 1, 1), color='blue', linestyle='--')

        plt.tight_layout()
        plt.savefig(f"C:\\muf\\graph\\{model_one}_timeseries({title_name})({scale_name}).png", dpi=370)
        plt.close()

        # QQ-plot
        if scale_y == "log":
            data_its_positive = data_its[data_its > 0]  # Exclude 0 and negative values
            data_log = np.log10(data_its_positive)
            fig = sm.qqplot(data_log, fit=True, line='45', markerfacecolor='black', markeredgecolor='None', alpha=0.5)
            ax = fig.get_axes()[0]
            plt.ylabel('Sample Quantiles (log₁₀ scale)')
            plt.title(f"Q-Q plot of {model_name_4th} (log₁₀ scale)")
            plt.tight_layout()
            plt.savefig(f"C:\\muf\\graph\\{model_one}_QQplot({title_name})(log10scale).png", dpi=370)
        else:
            fig = sm.qqplot(data_its, fit=True, line='45', markerfacecolor='black', markeredgecolor='None', alpha=0.5)
            ax = fig.get_axes()[0]
            plt.ylabel('Sample Quantiles')
            plt.title(f"Q-Q plot of {model_name_4th}")
            plt.tight_layout()
            plt.savefig(f"C:\\muf\\graph\\{model_one}_QQplot({title_name}).png", dpi=370)
        plt.close()

        # Histogram
        plt.figure(figsize=(14, 7))
        if scale_y == "log":
            data_its_positive = data_its[data_its > 0]  # Exclude 0 and negative values
            plt.hist(np.log10(data_its_positive), bins=50, edgecolor='black', color='black')
            plt.xlabel(f"log₁₀({model_name_4th}) {model_name_label}")
            plt.ylabel('Count')
            plt.title(f"Histogram of log₁₀({model_name_4th})")
        else:
            plt.hist(data_its, bins=50, edgecolor='black', color='black')
            plt.xlabel(f"{model_name_label}")
            plt.ylabel('Count')
            plt.title(f"Histogram of {model_name_4th}")

        plt.tight_layout()
        plt.savefig(f"C:\\muf\\graph\\{model_one}_histogram({title_name})({scale_name}).png", dpi=370)
        plt.close()

        # Boxplot
        plt.figure(figsize=(14, 7))
        if scale_y == "log":
            data_its_positive = data_its[data_its > 0]  # Exclude 0 and negative values
            sns.boxplot(y=data_its_positive, color='red', width=0.3,
                        whiskerprops={'color': 'black', 'linewidth': 1},
                        flierprops={'marker': 'o', 'markerfacecolor': 'gray', 'markersize': 3, 'linestyle': 'none'})
            plt.yscale('log')
            plt.ylabel(f"{model_name_label} (log₁₀ scale)")
            plt.title(f"Boxplot of {model_name_4th} (log₁₀ scale)")
        else:
            sns.boxplot(y=data_its, color='red', width=0.3, whiskerprops={'color': 'black', 'linewidth': 1},
                        flierprops={'marker': 'o', 'markerfacecolor': 'gray', 'markersize': 3, 'linestyle': 'none'})
            plt.ylabel(f"{model_name_label}")
            plt.title(f"Boxplot of {model_name_4th}")

        plt.tight_layout()
        plt.savefig(f"C:\\muf\\graph\\{model_one}_boxplot({title_name})({scale_name}).png", dpi=370)
        plt.close()

        # Hourly Boxplot
        hours = np.arange(24)

        plt.figure(figsize=(14, 7))
        means = data_its.groupby(ob1_min["tmfc_h"]).mean()

        if scale_y == "log":
            data_its_positive = data_its[data_its > 0]  # Exclude 0 and negative values
            tmfc_h_positive = ob1_min["tmfc_h"][data_its > 0]  # Filter tmfc_h correspondingly
            box_plot = sns.boxplot(x=tmfc_h_positive, y=data_its_positive, color='white', order=hours)
            plt.scatter(x=means.index, y=means.values, color='red', zorder=10)
            plt.axhline(y=cut_conc, color=col, linestyle='--')
            plt.yscale('log')
            plt.ylabel(f"{model_name_label} (log₁₀ scale)")
            plt.title(f"Hourly Boxplot of {model_name_4th} (log₁₀ scale)")
        else:
            box_plot = sns.boxplot(x=ob1_min["tmfc_h"], y=data_its, color='white', order=hours)
            plt.scatter(x=means.index, y=means.values, color='red', zorder=10)
            plt.axhline(y=cut_conc, color=col, linestyle='--')
            plt.ylabel(f"{model_name_label}")
            plt.title(f"Hourly Boxplot of {model_name_4th}")

        for i, artist in enumerate(box_plot.artists):
            artist.set_edgecolor('black')

        for j in range(len(box_plot.lines)):
            box_plot.lines[j].set_color('black')



        plt.xlabel('Hour of Day')

        plt.tight_layout()
        plt.savefig(f"C:\\muf\\graph\\{model_one}_hourly_boxplot({title_name})({scale_name}).png", dpi=370)
        plt.close()

# graph(ob1_min, "(가산A1타워주차장)")
# graph(ob2_min, "(에이샵스크린골프)")
# graph(ob3_min, "(영통역대합실)")
# graph(ob4_min, "(영통역지하역사)")
# graph(ob5_min, "(이든어린이집)")
# graph(ob6_min, "(좋은이웃데이케어센터1)")
# graph(ob7_min, "(좋은이웃데이케어센터2)")
# graph(ob8_min, "(좋은이웃데이케어센터3)")
graph(ob9_min, "(좋은이웃데이케어센터4)")
# graph(ob10_min, "(하이씨앤씨학원)")

# First, let's put your data in a list and map each to its corresponding location
locations = ['가산A1타워주차장', '에이샵스크린골프', '영통역대합실', '영통역지하역사', '이든어린이집', '좋은이웃데이케어센터1', '좋은이웃데이케어센터2', '좋은이웃데이케어센터3', '좋은이웃데이케어센터4', '하이씨앤씨학원']

datasets = [ob1_min, ob2_min, ob3_min, ob4_min, ob5_min, ob6_min, ob7_min, ob8_min, ob9_min, ob10_min]

location_data_mapping = dict(zip(locations, datasets))

models = ["pm10", "pm25", "pm1", "humi", "temp", "hcho", "co", "no2", "rn", "voc", "co2", "tab"]

missing_data_ratios_df = pd.DataFrame(columns=locations, index=models)

for location, data in location_data_mapping.items():
    for model in models:
        missing_data_ratio, _ = compute_missing_data_ratio(data)
        missing_data_ratios_df.loc[model, location] = missing_data_ratio

missing_data_ratios_df.to_excel("C:/muf/missing_data_ratios.xlsx")

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

results = {}
overall_results = {}

for name, df in datasets.items():
    for column in df.columns:
        means = []
        std_devs = []

        for place in df.index.unique():
            place_data = df.loc[place, column]
            # Exclude the place if all measurements for the substance are 0
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

df_results = pd.DataFrame(results, index=datasets.keys())

df_results = df_results.applymap(lambda x: f"{x[0]:.2f} ± {x[1]:.2f}" if isinstance(x, tuple) else x)

df_overall = pd.DataFrame(overall_results, index=['Overall'])
df_overall = df_overall.applymap(lambda x: f"{x[0]:.2f} ± {x[1]:.2f}" if isinstance(x, tuple) else x)

df_results = pd.concat([df_results, df_overall])

df_results.to_excel('output_file.xlsx')









