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

# 주별 리뷰 개수를 DataFrame으로 변환
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

