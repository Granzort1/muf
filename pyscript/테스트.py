import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# 로그정규분포에서 1000개의 랜덤 표본 추출
mu, sigma = 3, 2  # 로그정규분포의 파라미터
data = np.random.lognormal(mean=mu, sigma=sigma, size=1000)
data_normal = np.random.normal(loc=10, scale=3, size=1000)
# data에 대한 QQ 플롯
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
stats.probplot(data, plot=plt)
plt.title('QQ plot for data (normal)')

# data의 로그값에 대한 QQ 플롯
log_data = np.log(data)
plt.subplot(1, 2, 2)
stats.probplot(log_data, plot=plt)
plt.title('QQ plot for log(data)')

plt.tight_layout()
plt.show()
plt.close()



# data에 대한 QQ 플롯
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
stats.probplot(data_normal, plot=plt)
plt.title('QQ plot for data (normal)')
plt.subplot(1, 2, 2)
log_nor = np.log10(data_normal)
stats.probplot(log_nor, plot=plt)
plt.title('QQ plot for log(data_nor)')
plt.tight_layout()
plt.show()