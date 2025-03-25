import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# CSV 파일 읽기
file_path = r"C:\muf\에이샵스크린골프_20230331.csv"
df = pd.read_csv(file_path, encoding='utf-8')

# 날짜와 시간 열 결합
df['datetime'] = pd.to_datetime(df['tmfc_d'].astype(str) + df['tmfc_h'].astype(str).str.zfill(2), format='%Y%m%d%H')

# PM10 누락 비율 계산
total_rows = len(df)
missing_rows = df['pm10'].isnull().sum()
missing_ratio = missing_rows / total_rows * 100

print(f"PM10 누락 비율: {missing_ratio:.2f}%")

# 시간별 누락 패턴 분석
hourly_missing = df.groupby('tmfc_h')['pm10'].isnull().mean() * 100
plt.figure(figsize=(12, 6))
hourly_missing.plot(kind='bar')
plt.title('시간별 PM10 누락 비율')
plt.xlabel('시간')
plt.ylabel('누락 비율 (%)')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# 날짜별 누락 패턴 분석
daily_missing = df.groupby('tmfc_d')['pm10'].isnull().mean() * 100
plt.figure(figsize=(12, 6))
daily_missing.plot(kind='line')
plt.title('날짜별 PM10 누락 비율')
plt.xlabel('날짜')
plt.ylabel('누락 비율 (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 히트맵으로 날짜-시간별 누락 패턴 시각화
df['date'] = df['datetime'].dt.date
df['hour'] = df['datetime'].dt.hour
heatmap_data = df.pivot_table(values='pm10', index='date', columns='hour', aggfunc=lambda x: x.isnull().mean())
plt.figure(figsize=(15, 10))
sns.heatmap(heatmap_data, cmap='YlOrRd', vmin=0, vmax=1)
plt.title('날짜-시간별 PM10 누락 패턴')
plt.xlabel('시간')
plt.ylabel('날짜')
plt.tight_layout()
plt.show()