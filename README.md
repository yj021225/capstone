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
[McDonald's리뷰 데이터](https://www.kaggle.com/datasets/nelgiriyewithana/mcdonalds-store-reviews/data)

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

## 5. 느낀점 및 배운점
