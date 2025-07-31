import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

import os
import random
import sqlite3
from datetime import datetime, timedelta
import pandas as pd
from langchain_teddynote.tools.tavily import TavilySearch
from tavily import TavilyClient
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from settings import *
from Models import *
from Reward_function import *
from sklearn.metrics.pairwise import cosine_similarity

# 질문(질문에 있는 카드명)과 카드명 유사도 계산 함수
def find_most_similar(question_vector, card_vectors):
    similarities = {
        card: cosine_similarity([question_vector], [vector])[0][0]
        for card, vector in card_vectors.items()
    }
    # 유사도가 높은 카드 반환
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    return sorted_similarities[0]

def get_vectorDB_tool():
    # 벡터 DB 불러오기
    vectordb_embedding_model = load_vector_embedding_model()
    vector_store_path = "/home/work/RECO/Langgraph/Vector_DB/card_data_vector"
    db = FAISS.load_local(vector_store_path, vectordb_embedding_model, allow_dangerous_deserialization=True)

    # 카드명 임베딩 로드
    with open("/home/work/RECO/Langgraph/Vector_DB/card_vectors.pkl", "rb") as f:
        loaded_card_vectors = pickle.load(f)

    return vectordb_embedding_model, loaded_card_vectors, db

def get_web_tool(question, chatbot_model, chatbot_tokenizer):

    rewrite_prompt = PromptTemplate(
        template="""
    질문을 검색에 적합한 키워드 중심의 간결한 쿼리로 재작성해주세요.

    # Examples

    **Input**:
    "지역 기반 디지털 예술 창작 플랫폼과 제휴된 신용카드는 무엇인가요?"
    **Output**:
    "디지털 예술 제휴 신용카드 추천"
    **Input**:
    "전기차 충전소에서 추가 혜택을 제공하는 신용카드는 무엇인가요?"
    **Output**:
    "전기차 충전소 혜택 신용카드 추천"

    # Original Question:
    {question}

    # Reformulated Query:
    """,
        input_variables=["question"]
    )

    # Pipeline 생성
    pipe = pipeline(
        "text-generation",
        model=chatbot_model,
        tokenizer=chatbot_tokenizer,
        max_new_tokens=1024,  # 출력 제한
        temperature=0.1  # 다양성 조정
    )

    hf = HuggingFacePipeline(pipeline=pipe)
    question_rewriter = (rewrite_prompt | hf | StrOutputParser())

    reformulated_query = question_rewriter.invoke({"question": question}).strip()
    query = reformulated_query.split("# Reformulated Query:")[-1].strip()

    return query

def reward_logic(user_id, pred_keyword):
    print(f'here is reward_logic : ')

    card_dir = '/home/work/RECO/Langgraph/Dataset/card_information_data_credit_50_1217.json'
    # user_dir = '/content/drive/MyDrive/멋사자 AI 종합 프로젝트/범준'
    card_data, user_data = dataset_load(card_dir, user_id)

    results = reward_calculator(card_data, user_data, target_month=9, keyword=pred_keyword)
    return results

def classify_category(classifier, category_embeddings, embedding_model, device, question: str):
    # 54개 카테고리
    CATEGORIES = ['LPG충전소', 'OTT', '간편결제', '게임', '공과금', '공연', '기업형슈퍼마켓', '대중교통',
          '대형마트', '드럭스토어', '디지털구독', '렌탈', '리조트', '멤버십구독', '면세점', '미용', '미용실',
          '반려동물', '배달', '백화점', '보험', '세탁소', '쇼핑', '스포츠', '아울렛', '여행', '영화관',
          '온라인서점', '온라인쇼핑', '우체국', '웹툰', '음식점', '의료', '전기차충전소', '제과아이스크림',
          '주유소', '주차장', '차량정비소', '창고형할인매장', '카페', '콘도', '타이어샵', '택시', '테마파크',
          '통신사', '패션', '편의점', '학습지', '학원', '항공사', '해외대중교통', '호텔', '호텔음식점',
          '홈쇼핑']

    classifier.eval()
    with torch.no_grad():
        # (1, 1024)
        # q_emb = embedding_model.encode(question, convert_to_tensor=True).unsqueeze(0).to(device)
        q_emb = embedding_model.embed_query(question)
        q_emb = torch.tensor(q_emb, dtype=torch.float32).unsqueeze(0).to(device)

        logits = classifier(q_emb, category_embeddings)  # (1, 54)
        probs = torch.softmax(logits, dim=1)             # 확률 분포 (1, 54)

        pred_idx = torch.argmax(probs, dim=1).item()     # 0~53
        return CATEGORIES[pred_idx]
    
import pandas as pd
import json
from rapidfuzz import fuzz
from rapidfuzz import process
import re

# JSON 파일을 딕셔너리로 로드하는 함수
def load_json_to_dict(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        print("JSON 파일이 성공적으로 로드되었습니다.")
        return data
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {file_path}")
    except json.JSONDecodeError as e:
        print(f"JSON 디코딩 에러: {e}")
    except Exception as e:
        print(f"알 수 없는 오류 발생: {e}")


# 정규 표현식을 사용하여 문장에 알파벳이 있는지 찾고 대문자로 변환
def capitalize_english_words(text):
    return re.sub(r'\b[a-zA-Z]+\b', lambda match: match.group(0).upper(), text)

# 질문에 상호명이 있으면 상호명을 반환, 없으면 업종명을 다시 반환
# 함수호출 매개변수 : question:질문, category:업종, target_list:업종,상호명 json 파일을 변환한 dictionary 변수
def extract_store(question, category, target_list):
    cut_off_score = 90 # 업종, 상호명 출력 rapidfuzz 점수 기준
    temp_max = 0 # rapidfuzz 점수 최대값 판별용 변수
    result = '' # 상호명을 출력할 경우 rapidfuzz 점수가 최대 값인 상호명
    question = question.replace(' ', '') # 질문의 띄어쓰기 제거
    question = capitalize_english_words(question) # 질문에 있는 알파벳 모두 대문자로

    if len(target_list[category]) != 0: # 업종 하위에 상호명이 있을 경우 상호명 탐색
        # 긴 단어부터 우선순위 설정
        keywords = sorted(target_list[category], key=len, reverse=True)

        for store in keywords:
            store_original = store
            if '(LPG)' in store: store = store.split('(LPG)')[0]

            score = fuzz.partial_ratio(store, question) # question과 상호명으로 rapidfuzz 점수 계산

            if temp_max < score: # rapidfuzz 점수가 제일 높은 상호명 판별
                temp_max = score
                result = store_original

        if temp_max >= cut_off_score: # 기준치 이상이면 특정 상호면 리턴
            return result#, str(temp_max)

        else: # 기준치 이하일때 업종명 리턴
            return category#, result + '/' + str(temp_max)

    else: # 업종 하위에 상호명이 없을 경우 업종명 리턴
        return category#, '100'