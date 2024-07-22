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







def graph(ob1_min, title_name):
    # Convert tmfc_d and tmfc_h to datetime format
    ob1_min['datetime'] = pd.to_datetime(ob1_min['tmfc_d'].astype(str) + ob1_min['tmfc_h'].astype(str).str.zfill(2), format='%Y%m%d%H')
    ob1_min['log_pm10'] = np.log(ob1_min['pm10'])

    models_one = ["pm10", "pm25", "pm1", "humi", "temp", "hcho", "co", "no2", "rn", "voc", "co2", "tab"]

    for model_one in models_one:

        if model_one == 'pm10':
            model_name_4th = "PM10"
            model_name_label = "ug/m³"
            cut_conc = 200
            col = 'red'
        elif model_one == 'pm25':
            model_name_4th = "PM2.5"
            model_name_label = "ug/m³"
            cut_conc = 200
            col = 'red'
        elif model_one == 'pm1':
            model_name_4th = "PM1.0"
            model_name_label = "ug/m³"
            cut_conc = 200
            col = 'red'
        elif model_one == 'humi':
            model_name_4th = "Relative Humidity"
            model_name_label = "%"
            cut_conc = 100
            col = 'None'
        elif model_one == 'temp':
            model_name_4th = "Temperature"
            model_name_label = "°C"
            cut_conc = 1
            col = 'None'
        elif model_one == 'hcho':
            model_name_4th = "HCHO"
            model_name_label = "ug/m³"
            cut_conc = 100
            col = 'red'
        elif model_one == 'co':
            model_name_4th = "CO"
            model_name_label = "ppm"
            cut_conc = 25
            col = 'red'
        elif model_one == 'no2':
            model_name_4th = "NO2"
            model_name_label = "ppb"
            cut_conc = 300
            col = 'red'
        elif model_one == 'rn':
            model_name_4th = "Rn"
            model_name_label = "pCi/L"
            cut_conc = 4
            col = 'red'
        elif model_one == 'voc':
            model_name_4th = "VOC"
            model_name_label = "ug/m³"
            cut_conc = 1000
            col = 'red'
        elif model_one == 'co2':
            model_name_4th = "CO2"
            model_name_label = "ppm"
            cut_conc = 1000
            col = 'red'
        elif model_one == 'tab':
            model_name_4th = "TAB"
            model_name_label = "CFU/m³"
            cut_conc = 0
            col = 'None'
        else:
            pass

        #Time series plot

        plt.figure(figsize=(14, 7))
        plt.plot(ob1_min['datetime'], ob1_min[f"{model_one}"],'o', markersize=1, label=f"{model_one}", color='black', markeredgewidth=0)
        plt.yscale("symlog")
        plt.axhline(y=cut_conc, color=col, linestyle='--')
        plt.xlabel('Date', size=12)
        plt.ylabel(model_name_label, size=12)
        plt.title(f'Time series plot of {model_name_4th}', size=12)
        plt.legend()

        # Create a range of all dates/hours within your data
        all_dates = pd.date_range(start=ob1_min['datetime'].min(), end=ob1_min['datetime'].max(), freq='H')
        missing_dates = all_dates.difference(ob1_min['datetime'])

        if not missing_dates.empty:
            print(f'Missing dates for {model_one}: {missing_dates}')

        # Group missing_dates by contiguous periods
        missing_dates = missing_dates.to_series().sort_values()
        missing_periods = [(g.iloc[0], g.iloc[-1]) for _, g in missing_dates.groupby((missing_dates.diff().dt.total_seconds() > 3600).cumsum())]

        # Draw a vertical red line for each missing date
        for start, end in missing_periods:
            if start == end:
                plt.scatter(start, ob1_min[f"{model_one}"].min(), color='red', s=1)  # draw a vertical line for single missing hour
            else:
                plt.hlines(y=ob1_min[f"{model_one}"].min(), xmin=start, xmax=end, color='red')  # draw a horizontal line for missing period longer than 1 day

        for year in ob1_min['datetime'].dt.year.unique():
            for month in [2, 5, 8, 11]:  # end of Feb, May, Aug, Nov
                if (ob1_min['datetime'] >= datetime(year, month, 1)).any() and (ob1_min['datetime'] < datetime(year, month+1, 1)).any():
                    plt.axvline(x=datetime(year, month+1, 1), color='blue', linestyle='--')  # draw a vertical line at the beginning of the next month

        plt.tight_layout()
        plt.savefig(f"C:\\muf\\graph\\{model_one}_time({title_name})(log).png", dpi=370)
        plt.close()

        #
        #
        # #e QQ-plot####################
        # data_its = ob1_min[f"{model_one}"]
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
        # # Boxplot
        # plt.figure(figsize=(14, 7))
        # sns.boxplot(y=data_its, color='red')
        # plt.yscale("symlog")
        # plt.ylabel(f"{model_name_label}", size=12)
        # plt.title(f'Boxplot of {model_name_4th}', size=12)
        # plt.tight_layout()
        # plt.savefig(f'C:\\muf\\graph\\{model_one}_boxplot({title_name})(logscale).png', dpi=370)
        # plt.close()
        #
        #
        #
        # #박스플롯 시간별 지수단위
        #
        # plt.figure(figsize=(14, 7))
        #
        # # 박스 플롯 그리기. 박스 테두리 색상은 검정으로 설정
        # box_plot = sns.boxplot(x=ob1_min["tmfc_h"], y=data_its, color='white')
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
        # plt.yscale("symlog")
        # plt.title(f'Hourly Boxplot of {model_name_4th}', size=12)
        # plt.xlabel('Hour of Day', size=12)
        # plt.ylabel(f"{model_name_label}", size=12)
        # plt.tight_layout()
        # plt.savefig(f'C:\\muf\\graph\\{model_one}_hourly_boxplot({title_name})(logscale).png', dpi=370)
        # plt.close()
        #
        # # 박스플롯 시간별 일반스케일
        #
        # plt.figure(figsize=(14, 7))
        #
        # # 박스 플롯 그리기. 박스 테두리 색상은 검정으로 설정
        # box_plot = sns.boxplot(x=ob1_min["tmfc_h"], y=data_its, color='white')
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
        # plt.title(f'Hourly Boxplot of {model_name_4th}', size=12)
        # plt.xlabel('Hour of Day', size=12)
        # plt.ylabel(f"{model_name_label}", size=12)
        # plt.tight_layout()
        # plt.savefig(f'C:\\muf\\graph\\{model_one}_hourly_boxplot({title_name}).png', dpi=370)
        # plt.close()
        #





graph(ob1_min, "(가산A1타워주차장)")
# graph(ob2_min, "(에이샵스크린골프)")
# graph(ob3_min, "(영통역대합실)")
# graph(ob4_min, "(영통역지하역사)")
# graph(ob5_min, "(이든어린이집)")
# graph(ob6_min, "(좋은이웃데이케어센터1)")
# graph(ob7_min, "(좋은이웃데이케어센터2)")
# graph(ob8_min, "(좋은이웃데이케어센터3)")
# graph(ob9_min, "(좋은이웃데이케어센터4)")
# graph(ob10_min, "(하이씨앤씨학원)")









