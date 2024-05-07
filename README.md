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
### 2.2 추출한 데이터
가게명, 분류, 위도, 적도, 별점_개수등 필요없는 칼럼등을 제외하고 <br>
리뷰중에서 식별할 수 없는 글자들을 제거하였다. <br>
### 2.3 추출한 데이터에 대한 탐색적 데이터 분석
1~5점 척도인 경우에는 분포 <br>
리뷰 문장의 길이 <br>
연도별, 장소별 등등 데이터의 부가정보를 바탕으로 데이터를 탐색 (pandas, matplotlib)
맥도날드 별점 리뷰 분포표 <br>
<img src='data/맥도날드 리뷰 별점 분포표.png'>

## 3. 학습 데이터 구축

## 4. MobileBERT 학습 결과

## 5. 느낀점 및 배운점
