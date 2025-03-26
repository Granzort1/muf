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
dataframes = [pd.read_csv(f"C:/muf/input2/{filename}_20230331.csv", encoding='utf-8')
              for filename in ["가산A1타워주차장", "에이샵스크린골프", "영통역대합실", "영통역지하역사", "이든어린이집",
                              "좋은이웃데이케어센터1", "좋은이웃데이케어센터2", "좋은이웃데이케어센터3",
                              "좋은이웃데이케어센터4", "하이씨앤씨학원"]]

def replay(df):
    # 산점도 그리기 위한 그리드 설정
    grid = sns.PairGrid(df)

    # 상관계수 매트릭스 아래쪽은 각 두 변수간 산점도 그래프로 채움 (점색 블랙)
    grid.map_lower(sns.scatterplot, color='black', s=5, alpha=0.5, edgecolor='None')

    def scatterplot_with_downsampled_regression(x, y):
        downsampled_df = pd.concat([x, y], axis=1).sample(frac=0.1)  # Adjust the fraction as desired
        sns.regplot(x=downsampled_df[x.name], y=downsampled_df[y.name], scatter=False, color='red', ci='None')



    grid.map_lower(scatterplot_with_downsampled_regression)

    # 테두리 활성화
    for i, j in zip(*np.triu_indices_from(grid.axes, 1)):
        for edge in ['bottom', 'top', 'right', 'left']:
            grid.axes[i, j].spines[edge].set_visible(True)

    # 상관계수 매트릭스 위쪽은 두 변수간 상관계수로 채움
    def corr_heatmap(x, y, **kwargs):
        r = np.corrcoef(x, y)[0][1]
        ax = plt.gca()
        ax.annotate("{:.2f}".format(r),
                    xy=(.5, .5), xycoords=ax.transAxes,
                    horizontalalignment='center', verticalalignment='center', fontsize=20)

    grid.map_upper(corr_heatmap)

    # 행렬의 대각성분은 변수명과 함께 해당 변수데이터셋의 히스토그램으로 채움
    def label_diag(x, **kwargs):
        ax = plt.gca()
        ax.annotate(x.name,
                    xy=(.5, .5), xycoords=ax.transAxes,
                    horizontalalignment='center', verticalalignment='center', color='blue', fontsize=20)
    grid.map_diag(label_diag)

    plt.show()

for df in dataframes:
    replay(df)
