import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.font_manager as fm
from scipy.stats import norm


korean_font = fm.FontProperties(family='Batang')
# Read CSV
ob1_min = pd.read_csv("C:/muf/input2/가산A1타워주차장_20230331.csv", encoding='utf-8')
ob2_min = pd.read_csv("C:/muf/input2/에이샵스크린골프_20230331.csv", encoding='utf-8')
ob3_min = pd.read_csv("C:/muf/input2/영통역대합실_20230331.csv", encoding='utf-8')
ob4_min = pd.read_csv("C:/muf/input2/영통역지하역사_20230331.csv", encoding='utf-8')
ob5_min = pd.read_csv("C:/muf/input2/이든어린이집_20230331.csv", encoding='utf-8')
ob6_min = pd.read_csv("C:/muf/input2/좋은이웃데이케어센터1_20230331.csv", encoding='utf-8')
ob7_min = pd.read_csv("C:/muf/input2/좋은이웃데이케어센터2_20230331.csv", encoding='utf-8')
ob8_min = pd.read_csv("C:/muf/input2/좋은이웃데이케어센터3_20230331.csv", encoding='utf-8')
ob9_min = pd.read_csv("C:/muf/input2/좋은이웃데이케어센터4_20230331.csv", encoding='utf-8')
ob10_min = pd.read_csv("C:/muf/input2/하이씨앤씨학원_20230331.csv", encoding='utf-8')







def graph(ob1_min, title_name):
    ob1_min['log_pm10'] = np.log(ob1_min['pm10'])



    models_one = ["pm10", "temp", "log_pm10", "pm25", "pm1", "voc", "hcho", "co", "humi"]


    for model_one in models_one:
    #Q-Q Normality plot

        lower_val = np.percentile(ob1_min[f"{model_one}"], 10)
        upper_val = np.percentile(ob1_min[f"{model_one}"], 90)

        data_set = ob1_min[f"{model_one}"]
        ob1_min_filtered = data_set[(ob1_min[f"{model_one}"] >= lower_val) & (ob1_min[f"{model_one}"] <= upper_val)]
        print(ob1_min_filtered)
        # sm.qqplot(ob1_min_filtered, fit=True, line='45')
        # #stats.probplot(ob1_min_filtered, dist="norm", plot=plt)
        plt.hist(ob1_min_filtered)
        # plt.xlabel('Theoretical Quantiles')
        # plt.ylabel('Sample Quantiles')
        # plt.title(f"Q-Q plot of {model_name_4th}")
        # plt.tight_layout()
        plt.show()
        #lt.savefig(f"C:\\muf\\graph\\{model_one}_QQplot({title_name}).png", dpi=370)






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









