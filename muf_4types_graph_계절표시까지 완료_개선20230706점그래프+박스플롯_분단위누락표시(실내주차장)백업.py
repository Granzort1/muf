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
        #
        # plt.figure(figsize=(14, 7))
        # plt.plot(ob1_min['datetime'], ob1_min[f"{model_one}"], 'o', markersize=1, label=f"{model_name_4th}", color='black',
        #          markeredgewidth=0)
        # plt.yscale(f"{scale_y}")
        # plt.axhline(y=cut_conc, color=col, linestyle='--')
        # missing_data_ratio, _ = missing_data_ratios[model_one]
        # plt.text(0.1, 0.9, 'Missing data ratio: {:.2%}'.format(missing_data_ratio), transform=plt.gca().transAxes)
        # plt.xlabel('Date', size=12)
        # plt.ylabel(model_name_label, size=12)
        # plt.title(f'Time series plot of {model_name_4th}', size=12)
        #
        # # Create a Proxy Artist for the legend
        # bigger_dot = mlines.Line2D([], [], color='black', marker='o', linestyle='None',
        #                            markersize=3.7)  # Change the markersize to your desired size
        #
        # plt.legend(handles=[bigger_dot], labels=[f"{model_name_4th}"], fontsize=8)
        #
        # # Create a range of all dates/hours within your data
        # all_dates = pd.date_range(start=ob1_min['datetime'].min(), end=ob1_min['datetime'].max(), freq='H')
        #
        # # Check if there are any hours with less than 60 data points
        # incomplete_hours = ob1_min['datetime'].value_counts().loc[lambda x: x < 60].index
        # missing_dates = pd.Index(all_dates).difference(ob1_min['datetime']).union(incomplete_hours)
        #
        # if not missing_dates.empty:
        #     print(f'Missing dates for {model_one}: {missing_dates}')
        #
        # # Group missing_dates by contiguous periods
        # missing_dates = missing_dates.to_series().sort_values()
        # missing_periods = [(g.iloc[0], g.iloc[-1]) for _, g in
        #                    missing_dates.groupby((missing_dates.diff().dt.total_seconds() > 3600).cumsum())]
        #
        # # Draw a vertical red line for each missing date
        # for start, end in missing_periods:
        #     if start == end:
        #         plt.scatter(start, 0, color='red',
        #                     s=1)  # draw a dot for single missing hour
        #     else:
        #         plt.hlines(y=0, xmin=start, xmax=end,
        #                    color='red')  # draw a horizontal line for missing period longer than 1 hour
        #
        # for year in ob1_min['datetime'].dt.year.unique():
        #     for month in [2, 5, 8, 11]:  # end of Feb, May, Aug, Nov
        #         if (ob1_min['datetime'] >= datetime(year, month, 1)).any() and (
        #                 ob1_min['datetime'] < datetime(year, month + 1, 1)).any():
        #             plt.axvline(x=datetime(year, month + 1, 1), color='blue',
        #                         linestyle='--')  # draw a vertical line at the beginning of the next month
        #
        # plt.tight_layout()
        # plt.savefig(f"C:\\muf\\graph\\{model_one}_time({title_name})({scale_name}).png", dpi=370)
        # plt.close()

        #
        #
        # #e QQ-plot####################
#        data_its = ob1_min[f"{model_one}"]
        # fig = sm.qqplot(data_its, fit=True, line='45', markerfacecolor='black', markeredgecolor='None', alpha=0.5)
        # plt.yscale("symlog")
        # plt.xscale("symlog")
        # ax = fig.get_axes()[0]
        # plt.xlabel('Theoretical Quantiles', size=12)
        # plt.ylabel('Sample Quantiles', size=12)
        # plt.title(f"Q-Q plot of {model_name_4th}", size=12)
        # y_range2 = data_its.max()
        # ydata_max = float(f"{(y_range2):.1E}")
        # xmin, xmax = ax.get_xlim()
        #
        # # Create a list of tick labels based on the data range and the desired number of ticks, rounded to the nearest 2 decimal places
        # y_ticks = np.array([0, ydata_max])
        # x_position = np.array([xmin, xmax])
        # # Set the y-axis tick locations to match the x-axis tick locations
        # plt.yticks(x_position, y_ticks)
        # plt.tight_layout()
        # plt.savefig(f"C:\\muf\\graph\\{model_one}_QQplot({title_name})(logscale).png", dpi=370)
        # plt.close()
        #
        #
        # #일반단위 QQ플롯
        # data_its = ob1_min[f"{model_one}"]
        # fig = sm.qqplot(data_its, fit=True, line='45', markerfacecolor='black', markeredgecolor='None', alpha=0.5)
        # ax = fig.get_axes()[0]
        # plt.xlabel('Theoretical Quantiles', size=12)
        # plt.ylabel('Sample Quantiles', size=12)
        # plt.title(f"Q-Q plot of {model_name_4th}", size=12)
        # y_range2 = data_its.max()
        # ydata_max = float(f"{(y_range2):.1E}")
        # xmin, xmax = ax.get_xlim()
        #
        # # Create a list of tick labels based on the data range and the desired number of ticks, rounded to the nearest 2 decimal places
        # y_ticks = np.array([0, ydata_max])
        # x_position = np.array([xmin, xmax])
        # # Set the y-axis tick locations to match the x-axis tick locations
        # plt.yticks(x_position, y_ticks)
        # plt.tight_layout()
        # plt.savefig(f"C:\\muf\\graph\\{model_one}_QQplot({title_name}).png", dpi=370)
        # plt.close()
        #
        #
        #
        #
        #
        #
        # #히스토그램###################################################
        #
        # plt.figure(figsize=(14, 7))
        # #plt.rcParams["text.usetex"] = True
        # num_bins = 50
        #
        # # Create a new column to store the transformed values
        # ob1_min['transformed_var'] = ob1_min[f'{model_one}'].apply(
        #     lambda x: np.log10(x) if x > 0 else (-np.log10(-x) if x < 0 else 0))
        #
        # # Calculate the minimum and maximum of the range
        # min_val = np.floor(ob1_min['transformed_var'].min())
        # max_val = np.ceil(ob1_min['transformed_var'].max())
        #
        # # Create the bins
        # bins = np.linspace(min_val, max_val, num=num_bins)
        #
        # # Plot the histogram with these bin edges
        # plt.hist(ob1_min['transformed_var'], bins=bins, edgecolor='black', color='black')
        #
        # # The formatter will transform 10^x back to the original positive/negative value
        # def format_func(value, tick_number):
        #     return f"${'-' if value < 0 else ''}10^{{{'%.1f' % abs(value)}}}$"
        #
        # # Set the x-axis formatter
        # plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(format_func))
        #
        # plt.xlabel(f"{model_name_label}", size=12)
        # plt.ylabel('Count', size=12)
        # plt.title(f"Histogram of {model_name_4th}", size=12)
        # plt.legend()
        # plt.tight_layout()
        # plt.savefig(f"C:\\muf\\graph\\{model_one}_histogram({title_name})(logscale).png", dpi=370)
        # plt.close()
        #
        # #
        #
        # Boxplot


        # Define the function for generating y-ticks
        # def get_yticks(data):
        #     max_data, min_data = max(data), min(data)
        #
        #     if min_data < 0:
        #         yticks_negative = [-10 ** x for x in range(int(np.floor(np.log10(abs(min_data)))), -1, -1) if
        #                            -10 ** x >= min_data]
        #
        #     elif min_data == 0:
        #         yticks_negative = [0]
        #     else:
        #         yticks_negative = []
        #
        #     if max_data == 0:
        #         yticks_positive = [0]
        #
        #     else:
        #         yticks_positive = [10 ** x for x in range(0, int(np.floor(np.log10(max_data))) + 1) if 10 ** x <= max_data]
        #
        #     yticks = sorted(yticks_negative + yticks_positive)
        #     return yticks
        #
        #
        #
        # if max(data_its) < 1:
        #     data_max = 1
        # else:
        #     data_max = max(data_its)
        #
        # if min(data_its) <= 0:
        #     data_min = min(data_its)
        # else:
        #
        # # Now use this function in the plotting code
        # plt.figure(figsize=(14, 7))
        # sns.boxplot(y=data_its, color='red', width=0.3, whiskerprops={'color': 'black', 'linewidth': 1}, flierprops={'marker': 'o', 'markerfacecolor': 'gray', 'markersize': 3,
        #                 'linestyle': 'none'})
        # plt.yscale("symlog")
        # yticks = get_yticks(data_its)
        # plt.yticks(yticks)
        #
        #
        # plt.ylim(bottom=(min(data_its)-0.2))
        # plt.ylabel(f"{model_name_label}", size=12)
        # plt.title(f'Boxplot of {model_name_4th}', size=12)
        # plt.tight_layout()
        #
        # ax = plt.gca()  # get current axis
        #
        # # Change color of min and max y-tick labels to black
        # for label in ax.yaxis.get_ticklabels():
        #     if label._y in {min(yticks), max(yticks)}:  # Get y-data of label
        #         label.set_color('black')
        #
        # plt.savefig(f'C:\\muf\\graph\\{model_one}_boxplot({title_name})(logscale).png', dpi=370)
        # plt.close()

        #
#        hours = np.arange(24)

        #박스플롯 시간별 지수단위

        # plt.figure(figsize=(14, 7))
        #
        # # 평균값 계산
        #
        # # 박스 플롯 그리기. 박스 테두리 색상은 검정으로 설정
        # box_plot = sns.boxplot(x=ob1_min["tmfc_h"], y=data_its, color='white', order=hours)
        #
        # # 각 박스의 테두리 색상 변경
        # for i, artist in enumerate(box_plot.artists):
        #     artist.set_edgecolor('black')
        #
        # # whiskers(수염) 색상 변경
        # for j in range(len(box_plot.lines)):
        #     box_plot.lines[j].set_color('black')
        #
        # # 평균값 계산
        # means = data_its.groupby(ob1_min["tmfc_h"]).mean()
        #
        # # 평균값을 점으로 표시
        # plt.scatter(x=means.index, y=means.values, color='red', zorder=10)
        #
        # plt.yscale("symlog")
        # plt.axhline(y=cut_conc, color=col, linestyle='--')
        # plt.title(f'Hourly Boxplot of {model_name_4th}', size=12)
        # plt.xlabel('Hour of Day', size=12)
        # plt.ylabel(f"{model_name_label}", size=12)
        # plt.tight_layout()
        # plt.savefig(f'C:\\muf\\graph\\{model_one}_hourly_boxplot({title_name})(logscale).png', dpi=370)
        # plt.close()

        # 박스플롯 시간별 일반스케일

        # plt.figure(figsize=(14, 7))
        #
        # # 박스 플롯 그리기. 박스 테두리 색상은 검정으로 설정
        # box_plot = sns.boxplot(x=ob1_min["tmfc_h"], y=data_its, color='white', order=hours)
        #
        # # 각 박스의 테두리 색상 변경
        # for i, artist in enumerate(box_plot.artists):
        #     artist.set_edgecolor('black')
        #
        # # whiskers(수염) 색상 변경
        # for j in range(len(box_plot.lines)):
        #     box_plot.lines[j].set_color('black')
        #
        # # 평균값 계산
        # means = data_its.groupby(ob1_min["tmfc_h"]).mean()
        #
        # # 평균값을 점으로 표시
        # plt.scatter(x=means.index, y=means.values, color='red', zorder=10)
        # plt.axhline(y=cut_conc, color=col, linestyle='--')
        # plt.title(f'Hourly Boxplot of {model_name_4th}', size=12)
        # plt.xlabel('Hour of Day', size=12)
        # plt.ylabel(f"{model_name_label}", size=12)
        # plt.tight_layout()
        # plt.savefig(f'C:\\muf\\graph\\{model_one}_hourly_boxplot({title_name}).png', dpi=370)
        # plt.close()








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




# # First, let's put your data in a list and map each to its corresponding location
# locations = ['가산A1타워주차장', '에이샵스크린골프', '영통역대합실', '영통역지하역사', '이든어린이집', '좋은이웃데이케어센터1', '좋은이웃데이케어센터2', '좋은이웃데이케어센터3', '좋은이웃데이케어센터4', '하이씨앤씨학원']
# datasets = [ob1_min, ob2_min, ob3_min, ob4_min, ob5_min, ob6_min, ob7_min, ob8_min, ob9_min, ob10_min]
#
# location_data_mapping = dict(zip(locations, datasets))
#
# # Models
# models = ["pm10", "pm25", "pm1", "humi", "temp", "hcho", "co", "no2", "rn", "voc", "co2", "tab"]
#
# # Initialize a dataframe to store all the results
# missing_data_ratios_df = pd.DataFrame(columns=locations, index=models)
#
# # Loop through each dataset
# for location, data in location_data_mapping.items():
#     for model in models:
#         missing_data_ratio, _ = compute_missing_data_ratio(data)
#         missing_data_ratios_df.loc[model, location] = missing_data_ratio
#
# # Export dataframe to an Excel file
# missing_data_ratios_df.to_excel("C:/muf/missing_data_ratios.xlsx")


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