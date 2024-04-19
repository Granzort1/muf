import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.font_manager as fm
from scipy.stats import norm
from openpyxl import Workbook



korean_font = fm.FontProperties(family='Batang')
# Read CSV

# Read CSV
dataframes = [pd.read_csv(f"C:/muf/input2/{filename}_20230331.csv", encoding='utf-8')
              for filename in ["가산A1타워주차장", "에이샵스크린골프", "영통역대합실", "영통역지하역사", "이든어린이집",
                              "좋은이웃데이케어센터1", "좋은이웃데이케어센터2", "좋은이웃데이케어센터3",
                              "좋은이웃데이케어센터4", "하이씨앤씨학원"]]

filenames2 = ["가산A1타워주차장", "에이샵스크린골프", "영통역대합실", "영통역지하역사", "이든어린이집",
             "좋은이웃데이케어센터1", "좋은이웃데이케어센터2", "좋은이웃데이케어센터3",
             "좋은이웃데이케어센터4", "하이씨앤씨학원"]

# 각 장소별로 상관계수와 그 변수를 저장할 DataFrame
corr_df = pd.DataFrame(columns=["Location", "Variable Pair", "Correlation"])

# 각 데이터 프레임에 대해
for df, filename in zip(dataframes, filenames2):
    # 상관계수 계산
    corr_matrix = df.corr()

    # 절대값이 0.6 이상인 상관계수만 선택 (대각선은 제외)
    significant_corr = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
    significant_corr = significant_corr.stack().reset_index()

    # 필요한 조건에 맞는 상관계수를 찾기
    significant_corr = significant_corr[(abs(significant_corr[0]) >= 0.6) & (significant_corr[0] != 1.0)]

    for index, row in significant_corr.iterrows():
        corr_df = corr_df.append({
            "Location": filename,
            "Variable Pair": f"{row['level_0']} - {row['level_1']}",
            "Correlation": row[0]
        }, ignore_index=True)

# 엑셀에 저장
corr_df.to_excel('correlations.xlsx', index=False)