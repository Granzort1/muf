import os
import pandas as pd

# 입력 파일이 위치한 폴더 경로
input_folder = r"C:\muf\input\rawdata"

# 출력 파일이 저장될 폴더 경로
output_folder = r"C:\muf\input"

# 출력 파일 이름 리스트
output_file_names = ["가산A1타워주차장", "에이샵스크린골프", "영통역", "이든어린이집", "좋은이웃데이케어센터", "하이씨앤씨학원"]

# 입력 파일 리스트 생성
input_files = [f"data_past_time ({i}).xls" for i in range(42)]

# 7개씩 파일을 합치는 반복문
for i in range(0, len(input_files), 7):
    # 7개의 파일 그룹
    file_group = input_files[i:i + 7]

    # 파일 그룹에서 데이터를 합칠 데이터프레임 리스트
    df_list = []

    for j, file in enumerate(file_group):
        file_path = os.path.join(input_folder, file)
        df = pd.read_excel(file_path, header=0, skiprows=1)

        # 열 이름 변경
        df.columns = ['날짜', 'PM10', 'PM2.5', '오존', '이산화질소', '일산화탄소', '아황산가스']

        # "날짜" 열 데이터 처리
        if j < 3:
            df["날짜"] = "2022-" + df["날짜"].astype(str)
        else:
            df["날짜"] = "2023-" + df["날짜"].astype(str)

        # "날짜" 열을 제외한 나머지 열 데이터를 숫자로 변환
        numeric_columns = df.columns[df.columns != "날짜"]
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")

        df_list.append(df)

    # 데이터프레임 합치기
    merged_df = pd.concat(df_list, ignore_index=True)

    # 출력 파일 경로 생성
    output_file = output_file_names[i // 7] + ".xlsx"
    output_path = os.path.join(output_folder, output_file)

    # 출력 파일 저장
    merged_df.to_excel(output_path, index=False)