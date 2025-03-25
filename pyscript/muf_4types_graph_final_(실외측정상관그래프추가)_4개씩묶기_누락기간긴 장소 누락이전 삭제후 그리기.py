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
import os

korean_font = fm.FontProperties(family='Batang')

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def find_corresponding_outdoor_data(indoor_location):
    location_mapping = {
        "가산A1타워주차장": "가산A1타워주차장",
        "에이샵스크린골프": "에이샵스크린골프",
        "영통역대합실": "영통역",
        "영통역지하역사": "영통역",
        "이든어린이집": "이든어린이집",
        "좋은이웃데이케어센터1": "좋은이웃데이케어센터",
        "좋은이웃데이케어센터2": "좋은이웃데이케어센터",
        "좋은이웃데이케어센터3": "좋은이웃데이케어센터",
        "좋은이웃데이케어센터4": "좋은이웃데이케어센터",
        "하이씨앤씨학원": "하이씨앤씨학원",
    }

    column_mapping = {
        "pm10": "PM10",
        "pm25": "PM25",
        "co": "일산화탄소",
        "no2": "이산화질소",
    }

    return location_mapping[indoor_location], column_mapping

def compute_missing_data_ratio(data):
    start_time = data['datetime'].min()
    end_time = data['datetime'].max()
    total_data_points = int((end_time - start_time).total_seconds() / 60) + 1
    existing_data_points = data['datetime'].value_counts().sum()
    missing_data_points = total_data_points - existing_data_points
    missing_data_ratio = missing_data_points / total_data_points
    return missing_data_ratio, missing_data_points

def graph(ob_min, title_name, outdoor_data, column_mapping):
    create_directory(f"C:\\muf\\graph\\{title_name}")

    ob_min['datetime'] = pd.to_datetime(ob_min['tmfc_d'].astype(str) + ob_min['tmfc_h'].astype(str).str.zfill(2),
                                        format='%Y%m%d%H')

    # 특정 장소에 대한 데이터 필터링
    if title_name == "가산A1타워주차장":
        ob_min = ob_min[ob_min['datetime'] >= '2022-11-23']
    elif title_name == "좋은이웃데이케어센터4":
        ob_min = ob_min[ob_min['datetime'] >= '2022-10-20']
    elif title_name == "이든어린이집":
        pass  # 이든어린이집은 모든 데이터를 사용하되, VOC와 TAB에 대해서만 별도 처리

    models_one = ["pm10", "pm25", "pm1", "humi", "temp", "hcho", "co", "no2", "rn", "voc", "co2", "tab"]
    missing_data_ratios = {}

    for model_one in models_one:
        # 이든어린이집의 VOC와 TAB에 대한 특별 처리
        if title_name == "이든어린이집":
            if model_one == "voc":
                ob_min_filtered = ob_min[ob_min['datetime'] >= '2022-11-30']
            elif model_one == "tab":
                ob_min_filtered = ob_min[(ob_min['datetime'] >= '2022-12-29') & (ob_min['datetime'] <= '2023-03-22')]
            else:
                ob_min_filtered = ob_min
        else:
            ob_min_filtered = ob_min

        if model_one in ['humi', 'temp']:
            model_name_4th = model_one.capitalize()
            model_name_label = "%" if model_one == 'humi' else "°C"
            cut_conc = 100 if model_one == 'humi' else 10
            col = 'None'
            scale_y = "linear"
            scale_name = "Linear"
        else:
            model_name_4th = model_one.upper()
            model_name_label = "μg/m³" if model_one in ['pm10', 'pm25', 'pm1', 'hcho',
                                                        'voc'] else "ppm" if model_one in ['co',
                                                                                           'co2'] else "ppb" if model_one == 'no2' else "Bq/m³" if model_one == 'rn' else "CFU/m³" if model_one == 'tab' else ""
            cut_conc = 100 if model_one in ['hcho', 'co', 'tab'] else 200 if model_one in ['pm10', 'pm25',
                                                                                           'pm1'] else 1000 if model_one in [
                'voc', 'co2'] else 300 if model_one == 'no2' else 148 if model_one == 'rn' else None
            col = 'red'
            scale_y = "linear" if model_one in ['humi', 'temp'] else "log"
            scale_name = "Logarithmic" if model_one not in ['humi', 'temp'] else "Linear"

        missing_data_ratios[model_one] = compute_missing_data_ratio(ob_min_filtered)

        data_its = ob_min_filtered[f"{model_one}"]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

        if data_its[data_its > 0].empty:
            scale_y = "linear"

        # Time Series Plot
        if scale_y == "log":
            data_its_positive = data_its[data_its > 0]  # Exclude 0 and negative values
            datetime_positive = ob_min_filtered['datetime'][data_its > 0]  # Filter datetime correspondingly

            ax1.plot(datetime_positive, data_its_positive, 'o', markersize=1, label=f"{model_name_4th}", color='black',
                     markeredgewidth=0)
            ax1.axhline(y=cut_conc, color=col, linestyle='--')

            ax1.set_yscale('log')
            ax1.set_ylabel(f"{model_name_label} (log₁₀ scale)")
            ax1.set_title(f"Time Series Plot of {model_name_4th} (log₁₀ scale)")
        else:
            ax1.plot(ob_min_filtered['datetime'], data_its, 'o', markersize=1, label=f"{model_name_4th}", color='black',
                     markeredgewidth=0)
            ax1.axhline(y=cut_conc, color=col, linestyle='--')
            ax1.set_ylabel(f"{model_name_label}")
            ax1.set_title(f"Time Series Plot of {model_name_4th}")

        # X축 틱 레이블 수 제한
        start_date = ob_min_filtered['datetime'].min()
        end_date = ob_min_filtered['datetime'].max()

        # 이든어린이집 TAB 데이터에 대한 특별 처리
        if title_name == "이든어린이집" and model_one == "tab":
            end_date = pd.Timestamp('2023-03-22')

        date_range = pd.date_range(start=start_date, end=end_date, periods=6)
        ax1.set_xticks(date_range)
        ax1.set_xticklabels([d.strftime('%Y-%m-%d') for d in date_range])

        missing_data_ratio, _ = missing_data_ratios[model_one]
        ax1.text(0.1, 0.9, 'Missing data ratio: {:.2%}'.format(missing_data_ratio), transform=ax1.transAxes)
        ax1.set_xlabel('Date')

        bigger_dot = mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=3.7)
        ax1.legend(handles=[bigger_dot], labels=[f"{model_name_4th}"], fontsize=8)

        all_dates = pd.date_range(start=ob_min_filtered['datetime'].min(), end=ob_min_filtered['datetime'].max(), freq='H')
        incomplete_hours = ob_min_filtered['datetime'].value_counts().loc[lambda x: x < 60].index
        missing_dates = pd.Index(all_dates).difference(ob_min_filtered['datetime']).union(incomplete_hours)

        if not missing_dates.empty:
            print(f'Missing dates for {model_one}: {missing_dates}')

        missing_dates = missing_dates.to_series().sort_values()
        missing_periods = [(g.iloc[0], g.iloc[-1]) for _, g in
                           missing_dates.groupby((missing_dates.diff().dt.total_seconds() > 3600).cumsum())]

        for start, end in missing_periods:
            if start == end:
                ax1.scatter(start, data_its_positive.min(), color='#2E8B57', s=1)
            else:
                ax1.hlines(y=data_its_positive.min(), xmin=start, xmax=end, color='#2E8B57')

        for year in ob_min_filtered['datetime'].dt.year.unique():
            for month in [2, 5, 8, 11]:
                if (ob_min_filtered['datetime'] >= datetime(year, month, 1)).any() and (
                        ob_min_filtered['datetime'] < datetime(year, month + 1, 1)).any():
                    ax1.axvline(x=datetime(year, month + 1, 1), color='blue', linestyle='--')

        # Histogram
        if scale_y == "log":
            data_its_positive = data_its[data_its > 0]  # Exclude 0 and negative values
            ax2.hist(np.log10(data_its_positive), bins=50, edgecolor='black', color='black')
            ax2.set_xlabel(f"log₁₀({model_name_4th}) {model_name_label}")
            ax2.set_ylabel('Count')
            ax2.set_title(f"Histogram of log₁₀({model_name_4th})")
        else:
            ax2.hist(data_its, bins=50, edgecolor='black', color='black')
            ax2.set_xlabel(f"{model_name_label}")
            ax2.set_ylabel('Count')
            ax2.set_title(f"Histogram of {model_name_4th}")

        # Boxplot
        if scale_y == "log":
            data_its_positive = data_its[data_its > 0]  # Exclude 0 and negative values
            sns.boxplot(y=data_its_positive, color='red', width=0.3, ax=ax3,
                        whiskerprops={'color': 'black', 'linewidth': 1},
                        flierprops={'marker': 'o', 'markerfacecolor': 'gray', 'markersize': 3, 'linestyle': 'none'})
            ax3.set_yscale('log')
            ax3.set_ylabel(f"{model_name_label} (log₁₀ scale)")
            ax3.set_title(f"Boxplot of {model_name_4th} (log₁₀ scale)")
        else:
            sns.boxplot(y=data_its, color='red', width=0.3, ax=ax3, whiskerprops={'color': 'black', 'linewidth': 1},
                        flierprops={'marker': 'o', 'markerfacecolor': 'gray', 'markersize': 3, 'linestyle': 'none'})
            ax3.set_ylabel(f"{model_name_label}")
            ax3.set_title(f"Boxplot of {model_name_4th}")

        # QQ-plot
        if scale_y == "log":
            data_its_positive = data_its[data_its > 0]  # Exclude 0 and negative values
            data_log = np.log10(data_its_positive)
            sm.qqplot(data_log, fit=True, line='45', markerfacecolor='black', markeredgecolor='None', alpha=0.5, ax=ax4)
            ax4.set_ylabel('Sample Quantiles (log₁₀ scale)')
            ax4.set_title(f"Q-Q plot of {model_name_4th} (log₁₀ scale)")
        else:
            sm.qqplot(data_its, fit=True, line='45', markerfacecolor='black', markeredgecolor='None', alpha=0.5, ax=ax4)
            ax4.set_ylabel('Sample Quantiles')
            ax4.set_title(f"Q-Q plot of {model_name_4th}")

        plt.tight_layout()
        plt.savefig(f"C:\\muf\\graph\\{title_name}\\{model_one}_combined_{title_name}.png", dpi=370)
        plt.close()

        if model_one in ['pm10', 'pm25', 'co', 'no2']:
            # 실내 데이터를 시간 단위로 그룹화하고 평균 계산
            indoor_data_hourly = ob_min_filtered.groupby(['tmfc_d', 'tmfc_h'])[model_one].mean().reset_index()
            indoor_data_hourly['datetime'] = pd.to_datetime(
                indoor_data_hourly['tmfc_d'].astype(str) + indoor_data_hourly['tmfc_h'].astype(str).str.zfill(2),
                format='%Y%m%d%H')

            # 실외 데이터와 매칭
            outdoor_column_name = column_mapping[model_one]
            merged_data = pd.merge(indoor_data_hourly[['datetime', model_one]],
                                   outdoor_data[['datetime', outdoor_column_name]], on='datetime', how='inner')

            x = merged_data[outdoor_column_name]
            y = merged_data[model_one]

            # NaN 값이 있는지 확인하고 제거
            mask = ~(np.isnan(x) | np.isnan(y))
            x = x[mask]
            y = y[mask]

            r_squared = stats.pearsonr(x, y)[0] ** 2
            p_value = stats.pearsonr(x, y)[1]

            plt.figure(figsize=(9, 6))
            plt.scatter(x, y, color='black', alpha=0.5)
            # 선형 회귀선 추가
            slope, intercept, _, _, _ = stats.linregress(x, y)
            x_range = np.linspace(x.min(), x.max(), 100)
            y_predicted = slope * x_range + intercept
            plt.plot(x_range, y_predicted, color='blue', label='Linear Regression')

            # 축 라벨에 단위 추가
            if model_one in ['pm10', 'pm25']:
                xlabel = f'Outdoor {model_one.upper()} Concentration (μg/m³)'
                ylabel = f'Indoor {model_one.upper()} Concentration (μg/m³)'
            else:
                xlabel = f'Outdoor {model_one.upper()} Concentration (ppm)'
                ylabel = f'Indoor {model_one.upper()} Concentration (ppm)'

            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(f'Comparison of Indoor vs Outdoor {model_one.upper()} Concentration')

            if p_value < 0.001:
                p_value_text = 'p-value < 0.001'
            else:
                p_value_text = f'p-value = {p_value:.3f}'

            plt.text(0.6, 0.9, f'$r^2$ = {r_squared:.2f}\n{p_value_text}', transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            plt.legend()
            plt.tight_layout()
            plt.savefig(f"C:\\muf\\graph\\{title_name}\\{model_one}_in_vs_out_{title_name}.png", dpi=370)
            plt.close()


# 실내 측정 데이터 읽기
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

# 실외 측정 데이터 읽기
outdoor_data1 = pd.read_excel("C:/muf/input/가산A1타워주차장.xlsx")
outdoor_data2 = pd.read_excel("C:/muf/input/에이샵스크린골프.xlsx")
outdoor_data3 = pd.read_excel("C:/muf/input/영통역.xlsx")
outdoor_data4 = pd.read_excel("C:/muf/input/이든어린이집.xlsx")
outdoor_data5 = pd.read_excel("C:/muf/input/좋은이웃데이케어센터.xlsx")
outdoor_data6 = pd.read_excel("C:/muf/input/하이씨앤씨학원.xlsx")


# 데이터 전처리 및 datetime 열 추가
def preprocess_data(data):
    data['datetime'] = pd.to_datetime(data['날짜'], format='%Y-%m-%d-%H')
    return data


# 실외 측정 데이터를 딕셔너리로 저장
outdoor_data = {
    "가산A1타워주차장": preprocess_data(outdoor_data1),
    "에이샵스크린골프": preprocess_data(outdoor_data2),
    "영통역": preprocess_data(outdoor_data3),
    "이든어린이집": preprocess_data(outdoor_data4),
    "좋은이웃데이케어센터": preprocess_data(outdoor_data5),
    "하이씨앤씨학원": preprocess_data(outdoor_data6)
}

# 실내 측정 데이터와 실외 측정 데이터 매핑
indoor_data = [ob1_min, ob2_min, ob3_min, ob4_min, ob5_min, ob6_min, ob7_min, ob8_min, ob9_min, ob10_min]
indoor_locations = ["가산A1타워주차장", "에이샵스크린골프", "영통역대합실", "영통역지하역사", "이든어린이집", "좋은이웃데이케어센터1", "좋은이웃데이케어센터2", "좋은이웃데이케어센터3",
                    "좋은이웃데이케어센터4", "하이씨앤씨학원"]

for indoor_data, indoor_location in zip(indoor_data, indoor_locations):
    outdoor_location, column_mapping = find_corresponding_outdoor_data(indoor_location)
    outdoor_data_location = outdoor_data[outdoor_location]
    graph(indoor_data, indoor_location, outdoor_data_location, column_mapping)

# First, let's put your data in a list and map each to its corresponding location
locations = ['가산A1타워주차장', '에이샵스크린골프', '영통역대합실', '영통역지하역사', '이든어린이집', '좋은이웃데이케어센터1', '좋은이웃데이케어센터2', '좋은이웃데이케어센터3',
             '좋은이웃데이케어센터4', '하이씨앤씨학원']

datasets = [ob1_min, ob2_min, ob3_min, ob4_min, ob5_min, ob6_min, ob7_min, ob8_min, ob9_min, ob10_min]

location_data_mapping = dict(zip(locations, datasets))

models = ["pm10", "pm25", "pm1", "humi", "temp", "hcho", "co", "no2", "rn", "voc", "co2", "tab"]

missing_data_ratios_df = pd.DataFrame(columns=locations, index=models)

for location, data in location_data_mapping.items():
    # 데이터 필터링
    if location == "가산A1타워주차장":
        data['datetime'] = pd.to_datetime(data['tmfc_d'].astype(str) + data['tmfc_h'].astype(str).str.zfill(2),
                                          format='%Y%m%d%H')
        data = data[data['datetime'] >= '2022-11-23']
    elif location == "좋은이웃데이케어센터4":
        data['datetime'] = pd.to_datetime(data['tmfc_d'].astype(str) + data['tmfc_h'].astype(str).str.zfill(2),
                                          format='%Y%m%d%H')
        data = data[data['datetime'] >= '2022-10-20']

    for model in models:
        if location == "이든어린이집":
            if model == "voc":
                data_filtered = data[data['datetime'] >= '2022-11-30']
            elif model == "tab":
                data_filtered = data[data['datetime'] >= '2022-12-29']
            else:
                data_filtered = data
        else:
            data_filtered = data

        missing_data_ratio, _ = compute_missing_data_ratio(data_filtered)
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
    # 데이터 필터링
    if name == "가산A1타워주차장":
        df['datetime'] = pd.to_datetime(df['tmfc_d'].astype(str) + df['tmfc_h'].astype(str).str.zfill(2),
                                        format='%Y%m%d%H')
        df = df[df['datetime'] >= '2022-11-23']
    elif name == "좋은이웃데이케어센터4":
        df['datetime'] = pd.to_datetime(df['tmfc_d'].astype(str) + df['tmfc_h'].astype(str).str.zfill(2),
                                        format='%Y%m%d%H')
        df = df[df['datetime'] >= '2022-10-20']

    for column in df.columns:
        means = []
        std_devs = []

        if name == "이든어린이집":
            if column == "voc":
                df_filtered = df[df['datetime'] >= '2022-11-30']
            elif column == "tab":
                df_filtered = df[df['datetime'] >= '2022-12-29']
            else:
                df_filtered = df
        else:
            df_filtered = df

        for place in df_filtered.index.unique():
            place_data = df_filtered.loc[place, column]
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

