import pandas as pd

# 데이터셋을 DataFrame으로 읽어오기 (파일 경로는 실제 데이터셋 위치로 변경 필요)
df = pd.read_csv("McDonald'sReviews.csv", encoding='cp949')

# 필요한 값 추출
df_filtered = df[(df['rating'] != '3 stars') & df['rating'].notna()]
df_1_2_stars = df_filtered[df_filtered['rating'].isin(['1 star', '2 stars'])]
df_4_5_stars = df_filtered[df_filtered['rating'].isin(['4 stars', '5 stars'])]

# 샘플링된 데이터프레임 합치기
df_sampled = pd.concat([df_1_2_stars, df_4_5_stars])

# 'review' 컬럼만 남기기
df_sampled = df_sampled[['review']]

# 'label' 컬럼 생성
df_sampled['label'] = df_sampled['review'].apply(lambda x: 0 if x in df_1_2_stars['review'].values else 1)

# 결과 데이터프레임 출력 (또는 저장)
print(df_sampled)

# 파일로 저장하기 (필요할 경우)
df_sampled.to_csv("McDonald'sReviews_processed.csv", index=False)
