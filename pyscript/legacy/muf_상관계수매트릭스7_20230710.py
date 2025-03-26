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

def replay(df, filename):
    # 산점도 그리기 위한 그리드 설정
    #plt.figure(figsize=(16, 9), dpi=750)
    grid = sns.PairGrid(df, aspect=16/9, height=1.125)

    # 선형회귀 함수를 정의
    def scatterplot_with_downsampled_regression(x9, y9, color=None):
        downsampled_df = pd.concat([x9, y9], axis=1).sample(frac=0.1)  # Adjust the fraction as desired
        sns.regplot(x=downsampled_df[x9.name], y=downsampled_df[y9.name], scatter=False, color='red', ci=None)


    # 산점도 그리기
    def scatterplot(x8, y8, s=5, alpha=0.5, color=None, edgecolor='None'):
        sns.scatterplot(x=x8, y=y8, color='black', s=s, alpha=alpha, edgecolor=edgecolor)
        plt.gca().set_xlim(x8.quantile(0.003), x8.quantile(0.997))
        plt.gca().set_ylim(y8.quantile(0.003), y8.quantile(0.997))

    # 상관계수 매트릭스 아래쪽은 각 두 변수간 산점도 그래프로 채움
    grid.map_lower(scatterplot)
    grid.map_lower(scatterplot_with_downsampled_regression)

    for i in range(grid.axes.shape[0]):
        for j in range(grid.axes.shape[1]):
            for edge in ['bottom', 'top', 'right', 'left']:
                grid.axes[i, j].spines[edge].set_visible(True)

    # 상관계수 매트릭스 위쪽은 두 변수간 상관계수로 채움
    def corr_heatmap(x5, y5, **kwargs):
        r = np.corrcoef(x5, y5)[0][1]
        ax = plt.gca()
        ax.annotate("{:.2f}".format(r),
                    xy=(.5, .5), xycoords=ax.transAxes,
                    horizontalalignment='center', verticalalignment='center', fontsize=20)
        if abs(r) >= 0.6:  # If the absolute value of correlation is >= 0.6
            for edge in ['bottom', 'top', 'right', 'left']:
                ax.spines[edge].set_color('red')  # Change the border color to red
                ax.spines[edge].set_linewidth(3)  # Make the line thicker
        else:
            for edge in ['bottom', 'top', 'right', 'left']:
                ax.spines[edge].set_color(
                    'black')  # Change the border color to black (or any other color of your choice)
                ax.spines[edge].set_linewidth(1)  # Make the line normal thickness

    grid.map_upper(corr_heatmap)

    # 행렬의 대각성분은 변수명과 함께 해당 변수데이터셋의 히스토그램으로 채움
    def label_diag(x7, **kwargs):
        ax = plt.gca()
        ax.annotate(x7.name,
                    xy=(.5, .5), xycoords=ax.transAxes,
                    horizontalalignment='center', verticalalignment='center', color='blue', fontsize=25)
        #sns.histplot(x7, ax=ax, kde=True)

    grid.map_diag(label_diag)

    plt.savefig(f"C:/muf/result/{filename}_cor_matrix.png", dpi=750)
    plt.close()

filenames2 = ["가산A1타워주차장", "에이샵스크린골프", "영통역대합실", "영통역지하역사", "이든어린이집",
             "좋은이웃데이케어센터1", "좋은이웃데이케어센터2", "좋은이웃데이케어센터3",
             "좋은이웃데이케어센터4", "하이씨앤씨학원"]

for df, filename3 in zip(dataframes, filenames2):
    replay(df, filename3)

