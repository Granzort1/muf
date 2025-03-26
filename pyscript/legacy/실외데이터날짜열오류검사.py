import pandas as pd
import re

def correct_invalid_time(data, date_column='날짜'):
    corrected_dates = []
    for date in data[date_column]:
        if re.search(r'\b24\b', date):
            date_parts = date.split('-')
            date_parts[-1] = '00'
            corrected_date = '-'.join(date_parts)
            corrected_date = pd.to_datetime(corrected_date, format='%Y-%m-%d-%H') + pd.Timedelta(days=1)
            corrected_dates.append(corrected_date)
        else:
            corrected_dates.append(pd.to_datetime(date, format='%Y-%m-%d-%H'))
    data[date_column] = corrected_dates
    return data

def correct_and_save_datasets(datasets, output_directory):
    for name, data in datasets.items():
        corrected_data = correct_invalid_time(data)
        output_path = f"{output_directory}/{name}.xlsx"
        corrected_data.to_excel(output_path, index=False)
        print(f"Corrected data saved to {output_path}")

# 실외 데이터 읽기
outdoor_data1 = pd.read_excel("C:/muf/input/가산A1타워주차장.xlsx")
outdoor_data2 = pd.read_excel("C:/muf/input/에이샵스크린골프.xlsx")
outdoor_data3 = pd.read_excel("C:/muf/input/영통역.xlsx")
outdoor_data4 = pd.read_excel("C:/muf/input/이든어린이집.xlsx")
outdoor_data5 = pd.read_excel("C:/muf/input/좋은이웃데이케어센터.xlsx")
outdoor_data6 = pd.read_excel("C:/muf/input/하이씨앤씨학원.xlsx")

outdoor_datasets = {
    "가산A1타워주차장": outdoor_data1,
    "에이샵스크린골프": outdoor_data2,
    "영통역": outdoor_data3,
    "이든어린이집": outdoor_data4,
    "좋은이웃데이케어센터": outdoor_data5,
    "하이씨앤씨학원": outdoor_data6,
}

# 수정된 데이터를 저장할 디렉토리 설정
output_directory = "C:/muf/output"

correct_and_save_datasets(outdoor_datasets, output_directory)
