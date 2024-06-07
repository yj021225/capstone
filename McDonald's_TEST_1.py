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

# num = len(data['store_address'].unique().tolist())
# # print(' 가게 개수 :', num, '개')
# y = ['1 star', '2 stars', '3 stars', '4 stars', '5 stars']
# x = data[data['rating'].isin(y)]
# a = x['rating'].value_counts()
# plt.rcParams['font.family'] = 'Malgun Gothic'
# plt.rcParams['axes.unicode_minus'] = False
# plt.figure(figsize=(10, 6))
# a.plot(kind='bar')
# plt.title('리뷰 별점 분포표')
# plt.show()
#
# print(a)

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

# day = ['6 hours ago', '8 hours ago', '20 hours ago', '21 hours ago', '22 hours ago', '23 hours ago', 'a day ago',
#        '2 days ago', '3 days ago', '4 days ago', '5 days ago', '6 days ago', 'a week ago', '2 weeks ago',
#        '3 weeks ago', '4 weeks ago', 'a month ago', '2 months ago', '3 months ago', '4 months ago', '5 months ago',
#        '6 months ago', '7 months ago', '8 months ago', '9 months ago', '11 months ago']
# f = data[data['review_time'].isin(day)]
# z = f['review_time'].value_counts()
# print(z)
