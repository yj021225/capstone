import pandas as pd

# 데이터셋을 DataFrame으로 읽어오기 (파일 경로는 실제 데이터셋 위치로 변경 필요)
df = pd.read_csv("McDonald'sReviews.csv", encoding='cp949')

# 필요한 값 추출
df_filtered = df[(df['rating'] != '3 stars') & df['rating'].notna()]
df_1_2_stars = df_filtered[df_filtered['rating'].isin(['1 star', '2 stars'])]
df_4_5_stars = df_filtered[df_filtered['rating'].isin(['4 stars', '5 stars'])]

# 각 그룹에서 1500개씩 샘플링
df_1_2_stars_sampled = df_1_2_stars.sample(n=1500, random_state=1)
df_4_5_stars_sampled = df_4_5_stars.sample(n=1500, random_state=1)

# 샘플링된 데이터프레임 합치기
df_sampled = pd.concat([df_1_2_stars_sampled, df_4_5_stars_sampled])

# 'review' 컬럼만 남기기
df_sampled = df_sampled[['review']]

# 'label' 컬럼 생성
df_sampled['label'] = df_sampled['review'].apply(lambda x: 0 if x in df_1_2_stars['review'].values else 1)

# 결과 데이터프레임 출력 (또는 저장)
print(df_sampled)

# 파일로 저장하기 (필요할 경우)
df_sampled.to_csv("McDonald'sReviews_labeled.csv", index=False)

import pandas as pd

# 데이터셋을 DataFrame으로 읽어오기 (파일 경로는 실제 데이터셋 위치로 변경 필요)
df = pd.read_csv("McDonald'sReviews.csv", encoding='cp949')

# 필요한 값 추출
df_filtered = df[(df['rating'] != '3 stars') & df['rating'].notna()]
df_1_2_stars = df_filtered[df_filtered['rating'].isin(['1 star', '2 stars'])]
df_4_5_stars = df_filtered[df_filtered['rating'].isin(['4 stars', '5 stars'])]

# 긍정적인 리뷰와 부정적인 리뷰의 개수 계산
num_positive_reviews = len(df_4_5_stars)
num_negative_reviews = len(df_1_2_stars)

# 긍정적인 리뷰는 1500개, 부정적인 리뷰는 750개만 추출하여 샘플링
df_4_5_stars_sampled = df_4_5_stars.sample(n=min(num_positive_reviews, 2000), random_state=1)
df_1_2_stars_sampled = df_1_2_stars.sample(n=min(num_negative_reviews, 1000), random_state=1)

# 샘플링된 데이터프레임 합치기
df_sampled = pd.concat([df_1_2_stars_sampled, df_4_5_stars_sampled])

# 'review' 컬럼만 남기기
df_sampled = df_sampled[['review']]

# 'label' 컬럼 생성
df_sampled['label'] = df_sampled['review'].apply(lambda x: 0 if x in df_1_2_stars['review'].values else 1)

# 결과 데이터프레임 출력 (또는 저장)
print(df_sampled)

# 파일로 저장하기 (필요할 경우)
df_sampled.to_csv("McDonald'sReviews_labeled_balanced.csv", index=False)

df2 = pd.read_csv("McDonald'sReviews_labeled.csv", encoding='utf-8')

label_counts = df2['label'].value_counts()

print(label_counts)

df3 = pd.read_csv("McDonald'sReviews_labeled_balanced.csv", encoding='utf-8')

label_counts3 = df3['label'].value_counts()

print(label_counts3)
