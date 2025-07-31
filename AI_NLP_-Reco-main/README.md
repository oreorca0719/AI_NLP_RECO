# AI_NLP_Reco

<div align="center">
  <h2>💳 "당신의 라이프스타일에 맞춘 최고의 카드 추천" RecoAI</h2>
</div>

<div align="center">
  <h3>멋쟁이사자처럼 자연어처리 1팀 Reco</h3>
</div>

## 📋 프로젝트 개요
![image](https://github.com/OreOrca/AI_NLP_-Reco/blob/main/images/%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202024-12-25%20205754.png?raw=true)

### 🎯 프로젝트 목표
고객의 소비 패턴을 분석하거나 대화를 통해 요구사항을 파악하여 가장 높은 혜택을 받을 수 있는 카드를 추천하는 인공지능 챗봇 서비스입니다.

### 💡 주요 기능
- 고객의 소비 내역을 즉시 분석하여 라이프스타일에 맞는 카드 상품 추천
- 대화를 통해 고객의 요구사항을 파악, 이에 맞는 카드 상품 추천

## 📊 데이터셋

### 데이터 출처
- 카드고릴라 신용카드 TOP 100 차트의 상위 50개 카드 정보 데이터
- 이를 활용하여 제작한 모델 학습용 QAcontext Dataset

### 데이터셋 규모
- **카드 혜택 데이터셋**: 총 5,740개(벡터 DB 2740개+Web 검색 DB 3000개)
- 벡터 DB: 2740개(답변 정답 버전 1370개+답변 불가 버전 : 1370개)
- Web 검색: 3000개(답변 정답 버전 : 1500개+답변 불가 버전 : 1500개)

### 데이터셋 구성
- 카드 혜택이 주어지는 업종명, 가맹점명 분류
- 각 카드 혜택의 조건(ex: 전월실적, 연월실적, 최대 할인 한도 등)
-

## 🏗️ 모델 아키텍처
![image](https://github.com/OreOrca/AI_NLP_-Reco/blob/main/images/%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202024-12-25%20214301.png?raw=true)

### 1. BCCard/Llama-3.1-Kor-BCCard-Finance-8B
-BC Card, which is the largest credit card company in Korea, is a question/answer model learned using Korean financial datasets.
-금융 도메인 지식을 기반으로 fine-tuning된 Llama 기반 언어 모델

### 2. intfloat/multilingual-e5-large-instruct
- E5 기반 multilingual 임베딩 모델.
- 24 layers, embedding size 1024, 최대 text 길이 제한 512 tokens.

### 3. monologg/kobert
- 한국어 자연어 처리 작업을 위해 개발된 BERT 모델
- KoBERT 기반으로 문장 분류 모델을 구현

## 🛠️ 시스템 구성

1. **텍스트/작업 분류 시스템**
   - 입력된 텍스트에서 고객의 요구사항 인식 및 추출
   - 필요한 작업 파트로 분류

2. **Web 검색 시스템**
   - 고객 응대에 필수적인 정보 최신화 반영
     - 정확하고 트렌드에 맞는 정보를 전달하여 고객 만족도 상승

3. **Vector DB 검색 시스템**
   - 다양하고 복잡한 정보를 토대로 정확한 답변을 생성하기 위하여 Vector 유사도를 통한 정보 참조

4. **챗봇 시스템**
   - 추천 결과 생성
   - 실시간 대화식 인터페이스

## 📊 실행 결과

### 1. 챗봇 웹사이트
- 고객과의 실시간 대화 방식
- 즉각적인 요구사항 반영

## 🔍 기대 효과

- 고객 문의에 대한 효율적인 응대 가능
    -고객 센터 인원 업무 가중도 완화
- 카드 상품 제안에 대한 용이한 접근
    -가입 고객 수 증대

## 👥 팀 구성
멋쟁이사자처럼 자연어처리 1팀 Reco

------------------------------------------

© 2024 생성형 AI 카드 추천 서비스. All Rights Reserved.
