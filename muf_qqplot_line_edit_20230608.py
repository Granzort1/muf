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
    ob1_min['log_pm10'] = np.log(ob1_min['pm10'])

    # List of models to run
    models = [
        "pm10 ~ temp",
        "log_pm10 ~ temp",
        "pm10 ~ pm25",
        "pm10 ~ pm1",
        "pm25 ~ pm1",
        "voc ~ hcho",
        "hcho ~ co",
        "voc ~ co",
        "voc ~ humi",
    ]


    models_one = ["pm10", "temp", "log_pm10", "pm25", "pm1", "voc", "hcho", "co", "humi"]

# # Plot each model
# for model in models:
#     formula = model
#     result = smf.ols(formula, data=ob1_min).fit()
#     print(result.summary())
#
#     # calculate the 1st and 99th percentiles of the x and y data
#     x_data = ob1_min[model.split(" ~ ")[1]]
#     y_data = ob1_min[model.split(" ~ ")[0]]
#     x_low, x_high = np.percentile(x_data, [0.01, 99.9])
#     y_low, y_high = np.percentile(y_data, [0.01, 99.9])
#
#     plt.figure(figsize=(16, 8))
#     sns.regplot(x=ob1_min[model.split(" ~ ")[1]], y=ob1_min[model.split(" ~ ")[0]],
#                 line_kws={"color":"b","alpha":0.7,"lw":3},
#                 scatter_kws={"color": "black", "s": 20, "alpha": 0.5}, ci=None)  # 점의 색상을 검은색으로 변경하고, 크기를 절반으로 줄입니다.
#
# #
#         if model.split(' ~ ')[0] == 'pm10':
#             model_name0 = "PM10"
#         elif model.split(' ~ ')[0] == 'pm25':
#             model_name0 = "PM2.5"
#         elif model.split(' ~ ')[0] == 'pm1':
#             model_name0 = "PM1.0"
#         elif model.split(' ~ ')[0] == 'humi':
#             model_name0 = "Relative Humidity"
#         elif model.split(' ~ ')[0] == 'temp':
#             model_name0 = "Temperature"
#         elif model.split(' ~ ')[0] == 'hcho':
#             model_name0 = "HCHO"
#         elif model.split(' ~ ')[0] == 'co':
#             model_name0 = "CO"
#         elif model.split(' ~ ')[0] == 'no2':
#             model_name0 = "NO2"
#         elif model.split(' ~ ')[0] == 'rn':
#             model_name0 = "Rn"
#         elif model.split(' ~ ')[0] == 'voc':
#             model_name0 = "VOC"
#         elif model.split(' ~ ')[0] == 'co2':
#             model_name0 = "CO2"
#         elif model.split(' ~ ')[0] == 'tab':
#             model_name0 = "TAB"
#         else:
#             pass
#
#         if model.split(' ~ ')[1] == 'pm10':
#             model_name1 = "PM10"
#         elif model.split(' ~ ')[1] == 'pm25':
#             model_name1 = "PM2.5"
#         elif model.split(' ~ ')[1] == 'pm1':
#             model_name1 = "PM1.0"
#         elif model.split(' ~ ')[1] == 'humi':
#             model_name1 = "Relative Humidity"
#         elif model.split(' ~ ')[1] == 'temp':
#             model_name1 = "Temperature"
#         elif model.split(' ~ ')[1] == 'hcho':
#             model_name1 = "HCHO"
#         elif model.split(' ~ ')[1] == 'co':
#             model_name1 = "CO"
#         elif model.split(' ~ ')[1] == 'no2':
#             model_name1 = "NO2"
#         elif model.split(' ~ ')[1] == 'rn':
#             model_name1 = "Rn"
#         elif model.split(' ~ ')[1] == 'voc':
#             model_name1 = "VOC"
#         elif model.split(' ~ ')[1] == 'co2':
#             model_name1 = "CO2"
#         elif model.split(' ~ ')[1] == 'tab':
#             model_name1 = "TAB"
#         else:
#             pass

    #     if float(result.f_pvalue) < 0.001:
    #         p_res = "<0.001"
    #     else:
    #         p_res = f"={float(result.f_pvalue)}"
    #
    #
    #     plt.title(f"{model_name0} vs {model_name1} {title_name}", fontproperties=korean_font, fontsize=20, loc='left')
    #     plt.xlabel(f"{model_name1} concentration (ug/m$^{3}$)", fontsize=16)
    #     plt.ylabel(f"{model_name0} concentration (ug/m$^{3}$)", fontsize=16)
    #     plt.xlim(x_low, x_high)
    #     plt.ylim(y_low, y_high)
    #     plt.annotate(f'$r^{2}$ = {result.rsquared:.3f}(p-value{p_res})', xy=(0.05, 0.90), xycoords='axes fraction')
    #     plt.tight_layout()
    #     plt.savefig(f"C:/muf/x_temperature/{model.replace(' ~ ', '_').replace('np.log(', '').replace(')', '')}_S1{title_name}.png", dpi=375)
    #     plt.close()

    #추가 그래프#####################

    for model_one in models_one:


        if model_one == 'pm10':
            model_name_4th = "PM10"
        elif model_one == 'pm25':
            model_name_4th = "PM2.5"
        elif model_one == 'pm1':
            model_name_4th = "PM1.0"
        elif model_one == 'humi':
            model_name_4th = "Relative Humidity"
        elif model_one == 'temp':
            model_name_4th = "Temperature"
        elif model_one == 'hcho':
            model_name_4th = "HCHO"
        elif model_one == 'co':
            model_name_4th = "CO"
        elif model_one == 'no2':
            model_name_4th = "NO2"
        elif model_one == 'rn':
            model_name_4th = "Rn"
        elif model_one == 'voc':
            model_name_4th = "VOC"
        elif model_one == 'co2':
            model_name_4th = "CO2"
        elif model_one == 'tab':
            model_name_4th = "TAB"
        else:
            model_name_4th = "log(PM10)"
# # Time series plot
#         plt.figure(figsize=(14, 7))
#         plt.plot(ob1_min['ID'], ob1_min[f"{model_one}"], label=f"{model_one}", color='black')
#         plt.xlabel('Time', size=12)
#         plt.ylabel(f'ug/m$^{3}$', size=12)
#         plt.title(f'Time series plot of {model_name_4th}', size=12)
#         plt.legend()
#         plt.tight_layout()
#         plt.savefig(f"{model_one}_time({title_name}).png", dpi=370)
#         plt.close()
#
#         # Histogram
#         plt.figure(figsize=(14, 7))
#         plt.hist(ob1_min[f"{model_one}"], bins=50, label=f"{model_one}", color='black')
#         plt.xlabel(f"{model_name_4th}", size=12)
#         plt.ylabel('Count', size=12)
#         plt.title(f"Histogram of {model_name_4th}", size=12)
#         plt.legend()
#         plt.tight_layout()
#         plt.savefig(f"{model_one}_histogram({title_name}).png", dpi=370)
#         plt.close()
#
#         # Boxplot
#         plt.figure(figsize=(14, 7))
#         sns.boxplot(y=ob1_min[f"{model_one}"], color='red')
#         plt.ylabel(f"{model_name_4th}", size=12)
#         plt.title(f'Boxplot of {model_name_4th}', size=12)
#         plt.tight_layout()
#         plt.savefig(f'{model_one}_boxplot({title_name}).png', dpi=370)
#         plt.close()
#
        #Q-Q Normality plot




        # Calculate the percentiles
        lower_val = np.percentile(ob1_min[f"{model_one}"], 0.1)
        upper_val = np.percentile(ob1_min[f"{model_one}"], 99.9)


        data_its = ob1_min[f"{model_one}"]
        # Exclude the extreme percentiles
        ob1_min_filtered = data_its[(ob1_min[f"{model_one}"] > lower_val) & (ob1_min[f"{model_one}"] < upper_val)]

        # Now, the filtered data is used for the QQ-plot
        sm.qqplot(ob1_min_filtered, fit=True, line='45')
        plt.xlabel('Theoretical Quantiles', size=12)
        plt.ylabel('Sample Quantiles', size=12)
        plt.title(f"Q-Q plot of {model_name_4th}", size=12)
        plt.tight_layout()
        plt.savefig(f"C:\\muf\\graph\\{model_one}_QQplot({title_name}).png", dpi=370)
        plt.close()


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









