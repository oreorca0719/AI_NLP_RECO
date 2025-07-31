import os
import json
import random
import sqlite3
import torch
import pickle
import pandas as pd
from datetime import datetime, timedelta

from transformers import BertTokenizer, BertForTokenClassification

import pandas as pd
import sqlite3
import os


def custom_csv(directory_path = '/home/work/RECO/Langgraph/Dataset'):
    
    # SQLite DB 파일 경로
    sqlite_db_path = os.path.join(directory_path, 'customercard.sqlite')

    # SQLite DB 연결
    conn = sqlite3.connect(sqlite_db_path)

    # 디렉터리 내 모든 customercard_n월.csv 파일 처리
    for file_name in os.listdir(directory_path):
        if file_name.startswith('customercard_') and file_name.endswith('.csv'):
            file_path = os.path.join(directory_path, file_name)

            # CSV 파일 로드
            print(f"Processing file: {file_name}")
            data = pd.read_csv(file_path)

            # 데이터베이스에 저장 (append)
            data.to_sql('customer_card', conn, if_exists='append', index=False)

    # 날짜 기준 정렬을 위해 SQL 쿼리 실행
    sorted_query = """
    CREATE TABLE IF NOT EXISTS customer_card_sorted AS
    SELECT DISTINCT *
    FROM customer_card
    ORDER BY 거래일자 ASC;
    """
    conn.execute(sorted_query)

    # 기존 테이블 대체
    conn.execute("DROP TABLE customer_card")
    conn.execute("ALTER TABLE customer_card_sorted RENAME TO customer_card")

    print(f"데이터가 거래일자 기준으로 오름차순 정렬되었습니다: {sqlite_db_path}")

    # 연결 닫기
    conn.close()

def dataset_load(card_dir, user_id):

    # 카드 혜택 정보 로드
    with open(card_dir, 'r', encoding='utf-8') as f:
        card_data = json.load(f)

    # 디렉터리 경로 및 SQLite DB 파일 경로 설정
    sqlite_db_path = '/home/work/RECO/Langgraph/Dataset/customercard.sqlite'
    
    # SQLite 데이터베이스 연결
    conn = sqlite3.connect(sqlite_db_path)

    # 데이터베이스 테이블 목록 확인
    query = "SELECT name FROM sqlite_master WHERE type='table';"
    tables = pd.read_sql_query(query, conn)
    # print("Tables in the database:")
    # print(tables)

    # 고객 ID에 따라 DB 불러오기
    table_name = 'customer_card'
    query = f"SELECT * FROM {table_name} WHERE 고객ID = '{user_id}';"
    user_data = pd.read_sql_query(query, conn)

    # 연결 닫기
    conn.close()

    return card_data, user_data

# 업종 상호 리스트 로드
def load_business_list(json_path):
    with open(json_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# 업종 리스트를 기반으로 가맹점명과 업종명 추출
def get_random_store(keyword, Industry_store_list):
    if keyword and keyword in Industry_store_list:
        store = random.choice(Industry_store_list[keyword])
        return store, keyword

    random_industry = random.choice(list(Industry_store_list.keys()))
    if Industry_store_list[random_industry]:
        store = random.choice(Industry_store_list[random_industry])
        return store, random_industry
    else:
        return None, None

# 랜덤 날짜 생성 함수
def random_date_in_month(target_month, target_year):
    start_date = datetime(target_year, target_month, 1)

    if target_month == 12:  # December
        end_date = datetime(target_year + 1, 1, 1) - timedelta(days=1)
    else:
        end_date = datetime(target_year, target_month + 1, 1) - timedelta(days=1)

    random_day = random.randint(0, (end_date - start_date).days)
    random_date = start_date + timedelta(days=random_day)

    return random_date

# 더미 데이터 생성
def create_dummy_data(user_data, keyword):
    # 데이터프레임 복사
    dummy_data = user_data.copy()

    # 모든 매출 금액을 0으로 초기화
    dummy_data['매출금액'] = 0

    # keyword에 해당하는 업종명 또는 가맹점명의 소비 내역을 100만 원으로 설정
    keyword_mask = (dummy_data['업종명'] == keyword) | (dummy_data['가맹점명'] == keyword)
    dummy_data.loc[keyword_mask, '매출금액'] = 1_000_000

    return dummy_data
    
def get_benefit(card, transaction, last_usage, daily_usage_tracker, group_max_tracker):
    benefits = 0
    for i in range(1, 17):  # 혜택1부터 혜택16까지
        benefit_key = f'혜택{i}'
        if benefit_key in card:
            benefit_info = card[benefit_key]

            if isinstance(benefit_info, dict):
                # 연간 실적 그룹 확인: 연간 실적 그룹에 속하면 계산 제외
                if 'y_limit_group' in benefit_info and '연간실적' in card:
                    group_key = benefit_info['y_limit_group']
                    if group_key in card['연간실적']:
                        continue

                # 전월 실적 그룹 확인
                max_monthly_benefit = float('inf')
                if 'm_limit_group' in benefit_info and '전월실적' in card:
                    # 전월실적이 문자열일 경우 딕셔너리로 변환
                    if isinstance(card['전월실적'], str):
                        try:
                            card['전월실적'] = json.loads(card['전월실적'])
                        except json.JSONDecodeError:
                            print(f"Error decoding '전월실적' for card: {card['카드이름']}")
                            continue

                    group_key = benefit_info['m_limit_group']
                    group_info = card['전월실적'].get(group_key)
                    if group_info:
                        for max_benefit, range_str in zip(group_info[0], group_info[1]):
                            min_usage, max_usage = map(int, range_str.split('-'))
                            if min_usage * 10_000 <= last_usage <= max_usage * 10_000:
                                max_monthly_benefit = max_benefit
                                break

                # 바우처 혜택 처리
                if 'type' in benefit_info and benefit_info['type'].lower() == '바우처':
                    # 바우처 혜택이 있는 경우에만 처리
                    if last_usage > 2_000_000 and 'discount' in benefit_info:
                        discount_amount = int(benefit_info['discount'][0].replace('W', ''))

                        # 그룹 한도 확인
                        if group_max_tracker.get('y_count', 0) < benefit_info.get('y_count', 1):  # 연간 1회 제한 확인
                            if group_max_tracker['used'] + discount_amount <= max_monthly_benefit:
                                group_max_tracker['used'] += discount_amount
                                group_max_tracker['y_count'] = group_max_tracker.get('y_count', 0) + 1
                                benefits += discount_amount
                    continue  # 바우처 혜택 계산 후 다음 혜택으로 넘어감

                # 일반 혜택 처리
                if 'target' in benefit_info:
                    if transaction['업종명'] in benefit_info['target'] or transaction['가맹점명'] in benefit_info['target']:
                        if 'discount' in benefit_info:
                            discount = benefit_info['discount'][0]

                            # 할인 금액 계산
                            if 'W' in discount:  # 절대 금액
                                discount_amount = int(discount.replace('W', ''))
                            elif '%' in discount:  # 할인율
                                discount_rate = float(discount.strip('%')) / 100
                                discount_amount = transaction['매출금액'] * discount_rate
                            else:
                                discount_amount = 0  # 할인 정보가 올바르지 않을 경우 0으로 설정

                            # `discount_max` 제한
                            discount_max = benefit_info.get('discount_max', float('inf'))
                            discount_amount = min(discount_amount, discount_max)

                            # 하루 최대 횟수 `d_count` 제한
                            d_count = benefit_info.get('d_count', float('inf'))
                            transaction_date = transaction['거래일자'].strftime('%Y-%m-%d')
                            if daily_usage_tracker.get(transaction_date, 0) < d_count:
                                # 그룹 최대 혜택 금액 제한
                                if group_max_tracker['used'] + discount_amount <= max_monthly_benefit:
                                    discount_amount = min(discount_amount, max_monthly_benefit - group_max_tracker['used'])
                                    benefits += discount_amount
                                    daily_usage_tracker[transaction_date] = daily_usage_tracker.get(transaction_date, 0) + 1
                                    group_max_tracker['used'] += discount_amount
    return benefits


def reward_calculator(card_data, user_data, target_month, keyword):

    # User data preprocessing
    user_data['거래일자'] = pd.to_datetime(user_data['거래일자'], errors='coerce', format='%Y.%m.%d')
    user_data['거래월'] = user_data['거래일자'].dt.month

    # 목표 월 데이터 필터링
    if target_month in list(user_data['거래월'].unique()):
        target_user_data = user_data[user_data['거래월'] == target_month]
    else:
        available_months = list(user_data['거래월'].unique())
        closest_month = min(available_months, key=lambda x: abs(x - target_month))
        target_user_data = user_data[user_data['거래월'] == closest_month]
        target_month = closest_month

    # keyword가 None이 아닌 경우 더미 데이터 생성
    if keyword is not None:
        target_user_data = create_dummy_data(target_user_data, keyword)

    # 전월 실적 계산
    last_usage = target_user_data['매출금액'].sum()

    # 전월 실적 출력
    if keyword is None:
        print(f"전월 실적 (last_usage): {last_usage:,.0f}원")

    # 혜택 계산
    card_benefits = []
    for card in card_data:
        daily_usage_tracker = {}
        group_max_tracker = {'used': 0}
        benefit_result = pd.DataFrame(columns=['월', '업종명', '가맹점명', '매출금액', '혜택금액'])
        for _, transaction in target_user_data.iterrows():
            benefit = get_benefit(card, transaction, last_usage, daily_usage_tracker, group_max_tracker)
            if benefit > 0:
                new_row = {
                    '월': transaction.거래월,
                    '업종명': transaction.업종명,
                    '가맹점명': transaction.가맹점명,
                    '매출금액': transaction.매출금액,
                    '혜택금액': benefit
                }
                benefit_result.loc[len(benefit_result)] = new_row

        total_benefit = benefit_result['혜택금액'].sum()
        card_benefits.append({'카드이름': card['카드이름'], '혜택금액': total_benefit, '주요 혜택': card['주요 혜택'], 'URL': card['URL']})

    # 카드 이름으로 중복 제거
    unique_cards = {}
    for card in card_benefits:
        if card['카드이름'] not in unique_cards:
            unique_cards[card['카드이름']] = card

    # 중복 제거 후 혜택 금액으로 정렬
    card_benefits_sorted = sorted(unique_cards.values(), key=lambda x: x['혜택금액'], reverse=True)

    # 상위 3개 카드 추출
    top_3_cards = card_benefits_sorted[:3]

    context = "상위 3개 카드 혜택:\n"
    for rank, card in enumerate(top_3_cards, start=1):
        context += f"{rank}위: {card['카드이름']}, 혜택금액: {card['혜택금액']:,}원\n"
        context += f"주요 혜택: {card['주요 혜택']}\n"
        context += f"카드 발급 URL: {card['URL']}\n"
        context += '\n'

    return context  # 상위 3개 카드 반환

# def load_category_store_dict(filepath = '/home/work/RECO/Langgraph/Dataset/category_store_dict.pkl'):
#   with open(filepath, 'rb') as f:
#     return pickle.load(f)

# 유사 단어를 기준으로 target 단어 찾기
# def find_similar_target(pred_target, category_store_dict):
#     for target, words in category_store_dict.items():
#         if pred_target in words:  # 예측된 target이 유사 단어에 포함되면 기준 target 반환
#             return target
#     return pred_target  # 매칭되지 않으면 원래 예측값 반환

# NER 예측 및 target 단어 추출 함수
# def predict_keyword(question, keyword_model, keyword_tokenizer):

#     category_store_dict = load_category_store_dict()

#     label2id = {'O': 0, 'B-TARGET': 1, 'I-TARGET': 2}
#     id2label = {v: k for k, v in label2id.items()}

#     input_ids = keyword_tokenizer.encode(question, return_tensors="pt")
#     tokens = keyword_tokenizer.convert_ids_to_tokens(input_ids[0])[1:-1]  # CLS와 SEP 제외
#     with torch.no_grad():
#         outputs = keyword_model(input_ids)
#     predictions = torch.argmax(outputs.logits, dim=2)[0].tolist()[1:-1]

#     # BIO 라벨 기반 target 단어 조합
#     target_tokens = []
#     for token, label_id in zip(tokens, predictions):
#         if id2label[label_id] == "B-TARGET" or id2label[label_id] == "I-TARGET":
#             target_tokens.append(token.replace("##", ""))  # 서브워드 처리
#     pred_target = "".join(target_tokens)  # 조합된 target 단어

#     # 유사 단어 매핑을 활용해 target 보정
#     if not pred_target:  # 예측된 target이 비어있으면
#         for word in tokens:
#             for target, words in category_store_dict.items():
#                 if word in words:  # 토큰이 유사 단어에 포함되면 기준 target 반환
#                     return target
#     else:
#         pred_target = find_similar_target(pred_target, category_store_dict)  # 유사 단어 매핑 활용
#     return pred_target
