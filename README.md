<p align="center">
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/5/50/McDonald%27s_SVG_logo.svg/2095px-McDonald%27s_SVG_logo.svg.png" 
 width="450" height="450"/> <br>
</p>

# MobileBERT를 활용한 McDonald's 리뷰 분석 프로젝트 <br>
<!-- 
badge icon 참고 사이트
https://github.com/danmadeira/simple-icon-badges
-->
<img src="https://img.shields.io/badge/python-%233776AB.svg?&style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/pytorch-%23EE4C2C.svg?&style=for-the-badge&logo=pytorch&logoColor=white" />
<img src="https://img.shields.io/badge/pycharm-%23000000.svg?&style=for-the-badge&logo=pycharm&logoColor=white" />

## 1. 개 요
### 1.1 레스토랑 리뷰의 영향력
많은 사람들은 제품의 중요도가 낮고 값이 싸며 브랜드의 차이도 크게 없는 <br>
껌, 초콜릿 등을 구매하고자 할 때 해당 상품에 대한 구체적인 정보를 알아보지 않고 <br>
충동적으로 구매하는 경향이 있습니다. 하지만 고급 레스토랑에서 식사를 할때는 <br>
충동적으로 레스토랑을 방문하기보다는 구체적인 정보를 바탕으로 레스토랑을 방문하는 <br>
경향이 있습니다.여기서 구체적인 정보란 기업에서 제공하는 정보가 아닌, <br>
소비자들의 리뷰(체험)와 댓글 등입니다. <br>
출처: https://m.blog.naver.com/madahm/221409503825
### 1.2 문제 정의
세계에서 가장 유명한 레스토랑인 맥도날드의 리뷰로 미국에 위치한 맥도날드스토어를 방문한 <br>
소비자들이 스토어에 대한 리뷰를 포함하고있는 데이터이다. 이 프로젝트에서는 <br>
맥도날드 리뷰 데이터를 활용해 리뷰, 평점 등 다양한 특징에 따라 긍정 또는 부정을 예측하는 <br>
인공지능 모델을 개발하고자 한다. <br>

## 2. 데이터
### 2.1 원시 데이터
[McDonald'sReviews.csv](https://github.com/yj021225/capstone/blob/0607e92bb928115a5359a3c7774d3adfcd6c747a/BERT/McDonald'sReviews.csv)

| reviewer_id | store_name | category | store_address | latitude | longitude | rating_count | review_time | review | rating |
|-------------|------------|----------|---------------|----------|-----------|--------------|-------------|--------|--------|
| 고유_식별자 | 가게명 | 분류 | 가게_주소 | 위도 | 적도 | 별점_개수 | 리뷰_시간 | 리뷰 | 별점 |
 * 활용할 데이터 예시

| reviewer_id | store_name | category | store_address | latitude | longitude | rating_count | review_time | review | rating |
|-------------|------------|----------|---------------|----------|-----------|--------------|-------------|--------|--------|
| 1 | McDonald's | Fast food restaurant | "13749 US-183 Hwy... | 30.4607176 | -97.7928744 | "1,240" | 3 months ago | "Why does it look like someone spit... | 1 star |
| 2 | McDonald's | Fast food restaurant | "13749 US-183 Hwy... | 30.4607176 | -97.7928744 | "1,240" | 5 days ago | "It'd McDonalds. It is what it is as far... | 4 stars |
| 3 | McDonald's | Fast food restaurant | "13749 US-183 Hwy... | 30.4607176 | -97.7928744 | "1,240" | 5 days ago | "Made a mobile order got to the speaker and checked it in. |
### 2.2 탐색적 데이터 분석
<img src='data/맥도날드 리뷰 가게개수.png'><br>
<img src='data/맥도날드 리뷰 별점별개수.png'><br>
1~5점 척도인 경우에는 분포 <br>
리뷰 문장의 길이 <br>
연도별, 장소별 등등 데이터의 부가정보를 바탕으로 데이터를 탐색 (pandas, matplotlib) <br>
맥도날드 별점 리뷰 분포표 <br>
<img src='data/맥도날드 리뷰 별점 분포표.png'> <br>
맥도날드 주별 리뷰 개수를 나타내는 미국지도 데이터가 없는 하와이나 알래스카등 부속도서는 제외 <br>
<img src='data/맥도날드 주별 리뷰 개수 미국지도.png'> <br>

## 3. 학습 데이터 구축
맥도날드 리뷰 데이터셋에서 'review' 열만 남기고 긍부정을 구분하는 'label'열을 추가한 뒤 <br>
'rating'열 값이 '3 stars' 값이랑 결측 값은 제외하고 '1 star'값이랑 '2 stars'값은 부정을 나타내는 <br>
'0'값을 주고 '4 stars' 값이랑 '5 stars'값은 긍정을 나타내는 '1'값을 부여한다. <br>
- 데이터 예시 <br>

| review | label |
|--------|-------|
 | "Why does it look like someone spit on my food?... | 0 |
 | "I repeat my order 3 times in the drive thru... | 0 | 
 | "We stopped by for a quick breakfast... | 0 |
'label'열 값이 '0'값이 1500개 '1'값이 1500개씩 해서 총 3000개의 샘플 데이터를 구축한다. <br>

## 4. MobileBERT 학습 결과
<img src='data/맥도날드 리뷰_Training_14.png'> <br>
초기 단계에서의 loss는 46,248으로 매우 높은 값을 나타내지만 훈련이 진행될 수록 <br>
loss가 감소하고 있으며 세번째 단계에서는 loss값이 0.24로 매우 감소했다. Accuracy(정확도)는 <br>
학습할 수록 증가했으며, 학습 데이터의 긍부정 예측 정확도는 0.93이 나왔다. <br>
즉 모델이 학습 데이터의 긍정과 부정을 분류하는 것을 올바르게 학습되었다는 것을 확인할 수 있다. <br>

### 4.1 전체 데이터 분석에 적용한 결과값
<img src='data/맥도날드 리뷰_Training_13.png'> <br>
전체 분석 데이터 값에서는 0.92가 나왔다. 정확도가 높게 나와서 믿을만하다. <br> 

## 5. 프로젝트 코드
맥도날드 가게개수 및 주별 리뷰개수 그래프 코드 <br>
<pre>
<code>
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

data = pd.read_csv("McDonald'sReviews.csv", encoding='cp949')

def contains_broken_text(s):
    # 깨진 문자열 목록 (필요에 따라 추가)
    broken_strings = ['�', '占쏙옙']
    # 문자열이 깨진 텍스트를 포함하는지 확인
    return any(broken_string in str(s) for broken_string in broken_strings)

# 데이터 프레임의 각 셀에 대해 깨진 문자열이 있는지 검사하여 행 제거
data_cleaned = data[~data.applymap(contains_broken_text).any(axis=1)]

num = len(data['store_address'].unique().tolist())
# print(' 가게 개수 :', num, '개')
y = ['1 star', '2 stars', '3 stars', '4 stars', '5 stars']
x = data[data['rating'].isin(y)]
a = x['rating'].value_counts()
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(10, 6))
a.plot(kind='bar')
plt.title('리뷰 별점 분포표')
plt.show()

print(a)

b = ['DE', 'PA', 'NJ', 'GA', 'CT', 'MA', 'MD', 'SC', 'NH', 'VA', 'NY', 'NC', 'RI', 'VT', 'KY',
     'TN', 'OH', 'LA', 'IN', 'MS', 'IL', 'AL', 'ME', 'MO', 'AR', 'MI', 'FL', 'TX', 'IA', 'WI',
     'CA', 'MN', 'OT', 'KS', 'WV', 'NV', 'NE', 'CO', 'ND', 'SD', 'MT', 'WA', 'ID', 'WY', 'UT',
     'OK', 'NM', 'AZ', 'AK', 'HI', 'DC']

print(data_cleaned['store_address'].value_counts())
state_info = []
for j in range(len(data_cleaned)):
       address = data_cleaned.iloc[j]['store_address']
       address = str(address)
       isin = 0
       for state in b:
              if state in address:
                     isin = 1
                     state_info.append(state)
                     continue
       if isin==0:
              print(address)
print(len(data_cleaned))
print(len(state_info))

counter = Counter(state_info)

objects = list(counter.keys())
counts = list(counter.values())

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(10, 5))
plt.bar(objects, counts, color='skyblue')

plt.title('주별 리뷰 개수')
plt.xlabel('Objects')
plt.ylabel('Counts')

plt.show()
</code>
</pre>
리뷰 데이터 정리 코드 <br>
<pre>
<code>
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
</code>
</pre>
리뷰 데이터 학습 데이터 구축 코드 <br>
<pre>
<code>
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
</code>
</pre>
맥도날드 주별 리뷰 개수 미국지도 코드 <br>
<pre>
<code>
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import geopandas as gpd
from shapely.geometry import Point
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
# CSV 파일 읽기
data = pd.read_csv("McDonald'sReviews.csv", encoding='cp949', low_memory=False)
# 깨진 문자열 확인 함수 정의
def contains_broken_text(s):
    broken_strings = ['�', '占쏙옙']
    return any(broken_string in str(s) for broken_string in broken_strings)
# 데이터 클리닝
data_cleaned = data[~data.applymap(contains_broken_text).any(axis=1)]
# 주 리스트
b = ['DE', 'PA', 'NJ', 'GA', 'CT', 'MA', 'MD', 'SC', 'NH', 'VA', 'NY', 'NC', 'RI', 'VT', 'KY',
     'TN', 'OH', 'LA', 'IN', 'MS', 'IL', 'AL', 'ME', 'MO', 'AR', 'MI', 'FL', 'TX', 'IA', 'WI',
     'CA', 'MN', 'OT', 'KS', 'WV', 'NV', 'NE', 'CO', 'ND', 'SD', 'MT', 'WA', 'ID', 'WY', 'UT',
     'OK', 'NM', 'AZ', 'AK', 'HI', 'DC']
# 주별 리뷰 개수 계산
state_info = []
for j in range(len(data_cleaned)):
    address = data_cleaned.iloc[j]['store_address']
    address = str(address)
    for state in b:
        if state in address:
            state_info.append(state)
            break
counter = Counter(state_info)
#주별 리뷰 개수를 DataFrame으로 변환
state_review_counts = pd.DataFrame.from_dict(counter, orient='index', columns=['review_count'])
state_review_counts.index.name = 'state'
state_review_counts.reset_index(inplace=True)
# 미국 주 경계 데이터 로드
us_states = gpd.read_file('https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json')
# state 코드 추가
state_abbrev = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
    'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA',
    'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS',
    'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD', 'Massachusetts': 'MA',
    'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT',
    'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM',
    'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK',
    'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD',
    'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA',
    'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'
}
# 이름을 주 약자로 변환
us_states['state'] = us_states['name'].map(state_abbrev)
# 주별 리뷰 개수 데이터를 지오데이터프레임에 병합
us_states = us_states.merge(state_review_counts, left_on='state', right_on='state', how='left')
us_states['review_count'] = us_states['review_count'].fillna(0)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
# 리뷰가 있는 주 필터링
states_with_reviews = us_states[us_states['review_count'] > 0]
# 지도 그리기
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
us_states.boundary.plot(ax=ax)
us_states.plot(column='review_count', ax=ax, legend=False, cmap='Blues')
# 구글 지도 위치 아이콘 이미지 파일 로드
icon_img = plt.imread('data/google_maps_icon.png')
# 리뷰가 있는 주에 위치 아이콘 표시
for idx, row in states_with_reviews.iterrows():
    icon = OffsetImage(icon_img, zoom=0.03)  # 아이콘 크기 조정
    ab = AnnotationBbox(icon, (row.geometry.centroid.x, row.geometry.centroid.y), frameon=False)
    ax.add_artist(ab)
for idx, row in us_states.iterrows():
    plt.annotate(text=row['state'], xy=(row.geometry.centroid.x, row.geometry.centroid.y),
                 ha='center', fontsize=8, color='black')
plt.title('주별 리뷰 개수')
# 컬러바 추가
cax = fig.add_axes([0.85, 0.56, 0.033, 0.25])  # 범례 위치 및 크기 조정
sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=us_states['review_count'].min(), vmax=us_states['review_count'].max()))
sm._A = []  # 빈 데이터 설정
cbar = plt.colorbar(sm, cax=cax)
cbar.set_label('리뷰 개수', rotation=0, labelpad=-50, y=1.1, ha='center', fontsize=12)
plt.show()
</code>
</pre>
Mobile BERT 학습 코드1 <br>
<pre>
<code>
import pandas as pd
from transformers import MobileBertForSequenceClassification, MobileBertTokenizer
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
from transformers import get_linear_schedule_with_warmup, logging
import time
import datetime
path = "McDonald'sReviews_labeled.csv"
df = pd.read_csv(path, encoding="utf-8")
data_X = list(df['review'].values)
labels = df['label'].values
print("*** 데이터 ***")
print("문장"); print(data_X[:5])
print("라벨"); print(labels[:5])
num_to_print = 3
tokenizer = MobileBertTokenizer.from_pretrained('mobilebert-uncased', do_lower_case=True)
inputs = tokenizer(data_X, truncation=True, max_length=256, add_special_tokens=True, padding="max_length")
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']
print("\n\n*** 토큰화 ***")
for j in range(num_to_print):
    print(f"\n{j+1}번째 데이터")
    print("** 토큰 **")
    print(input_ids[j])
    print("** 어텐션 마스크 **")
    print(attention_mask[j])
train, validation, train_y, validation_y = train_test_split(input_ids, labels, test_size=0.1, random_state=2024)
train_masks, validation_masks, _, _ = train_test_split(attention_mask, labels, test_size=0.1, random_state=2024)
batch_size = 8
train_inputs = torch.tensor(train)
train_labels = torch.tensor(train_y)
train_masks = torch.tensor(train_masks)
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
validation_inputs = torch.tensor(validation)
validation_labels = torch.tensor(validation_y)
validation_masks = torch.tensor(validation_masks)
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)
model = MobileBertForSequenceClassification.from_pretrained('mobilebert-uncased', num_labels=2)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, eps=1e-8)
epoch = 4
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=len(train_dataloader) * epoch)
for e in range(0, epoch):
    print(f'\n\nEpoch {e+1} / {epoch}')
    print('Training')
    t0 = time.time()
    total_loss = 0
    model.train()
    for step, batch in enumerate(train_dataloader):
        if step % 50 == 0 and not step == 0:
            elapsed_rounded = int(round(time.time() - t0))
            elapsed = str(datetime.timedelta(seconds=elapsed_rounded))
            print(f'- Batch {step} of {len(train_dataloader)}, Elapsed time: {elapsed}')
        batch_ids, batch_mask, batch_labels = tuple(t for t in batch)
        model.zero_grad()
        outputs = model(batch_ids, token_type_ids=None, attention_mask=batch_mask, labels=batch_labels)
        loss = outputs.loss
        total_loss += loss.item()
        if step % 10 == 0 and not step == 0:
            print(f'step : {step}, loss : {loss.item()}')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    avg_train_loss = total_loss / len(train_dataloader)
    print(f'Average training loss : {avg_train_loss}')
    train_time_per_epoch = str(datetime.timedelta(seconds=(int(round(time.time() - t0)))))
    print(f'Training time of epoch {e} : {train_time_per_epoch}')
    print('\n\Validation')
    t0 = time.time()
    model.eval()
    eval_loss, eval_accuracy, eval_step, eval_examples = 0, 0, 0, 0
    for batch in validation_dataloader:
        batch_ids, batch_mask, batch_labels = tuple(t for t in batch)
        with torch.no_grad():
            outputs = model(batch_ids, token_type_ids=None, attention_mask=batch_mask)
        logits = outputs[0]
        logits = logits.numpy()
        label_ids = batch_labels.numpy()
        pred_flat = np.argmax(logits, axis=1).flatten()
        labels_flat = label_ids.flatten()
        eval_accuracy_temp = np.sum(pred_flat == labels_flat) / len(labels_flat)
        eval_accuracy += eval_accuracy_temp
        eval_step += 1
    print(f'Validation accuracy : {eval_accuracy / eval_step}')
    val_time_per_epoch = str(datetime.timedelta(seconds=int(round(time.time() - t0))))
    print(f'Validation time of epoch {e} : {val_time_per_epoch}')
print('\nSave Model')
save_path = 'mobilebert_model3'
model.save_pretrained(save_path+'.pt')
print('\nFinish')
</code>
</pre>
Mobile BERT 학습 코드2 <br>
<pre>
<code>
import pandas as pd
import numpy as np
from transformers import MobileBertForSequenceClassification, MobileBertTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
model_path = 'mobilebert_model3.pt'
model = MobileBertForSequenceClassification.from_pretrained(model_path)
model.eval()
df = pd.read_csv("McDonald'sReviews_processed.csv", encoding="utf-8")
data_X = list(df['review'].values)
labels = df['label'].values
tokenizer = MobileBertTokenizer.from_pretrained('mobilebert-uncased', do_lower_case=True)
inputs = tokenizer(data_X, truncation=True, max_length=256, add_special_tokens=True, padding="max_length")
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']
batch_size = 8
test_inputs = torch.tensor(input_ids)
test_labels = torch.tensor(labels)
test_masks = torch.tensor(attention_mask)
test_dataset = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size)
test_loss, test_accuracy, test_steps, test_examples = 0, 0, 0, 0
for batch in test_dataloader:
    batch_ids, batch_masks, batch_labels = tuple(t for t in batch)
    with torch.no_grad():
        outputs = model(batch_ids, token_type_ids=None, attention_mask=batch_masks)
    logits = outputs[0]
    logits = logits.numpy()
    label_ids = batch_labels.numpy()
    pred_flat = np.argmax(logits, axis=1).flatten()
    label_flat = label_ids.flatten()
    test_accuracy_temp = np.sum(pred_flat == label_flat) / len(label_flat)
    test_accuracy += test_accuracy_temp
    test_steps += 1
    print(f"Test step : {test_steps}/{len(test_dataloader)}, Temp Accuracy : {test_accuracy_temp}")
avg_test_accuracy = test_accuracy / test_steps
print(f"Total Accuracy : {avg_test_accuracy}")
</code>
</pre>

## 6. 느낀점 및 배운점
리뷰 데이터를 활용해서 긍정과 부정을 예측하는 Mobile BERT 학습을 배울수 있었다. <br>
앞으로는 더욱 다양한 감정들을 분석할 수 있는 딥러닝 기술이 만들어 졌으면 좋겠다.
