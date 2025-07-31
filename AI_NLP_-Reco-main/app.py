from flask import Flask, render_template, request, redirect, url_for, jsonify, session
import subprocess
import json
import logging
import sqlite3
import pandas as pd
import io
import base64
from settings import *
from Models import *
from Graph import *
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import random
from datetime import datetime, timedelta

from Reward_function import (
    dataset_load,
    reward_calculator
)

font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
font_prop = fm.FontProperties(fname=font_path)
plt.rc('font', family=font_prop.get_name())
plt.rcParams['axes.unicode_minus'] = False
app = Flask(__name__)
app.secret_key = 'your_secret_key'
users = {"testuser": "password123","admin": "adminpass"}

def process_input(user_input):
    state["messages"].append(HumanMessage(content=user_input))
    try:
        for event in graph.stream(state, config=config):
            for value in event.values():
                state.update(value)

                state["messages"] = value["messages"]
                last_message = value["messages"][-1]

                if isinstance(last_message, AIMessage):
                    return {"response": last_message.content}
                
                if (state['next'] == 'Reward_logic') and (len(state['context'])>2):
                    return {"response": state['context']}
                
        return {"response": "Assistant: Undefined"}
    except Exception as e:
        logging.error(f"{e}")
        return {"response": "오류가 발생했습니다. 다시 시도해주세요."}

graph = None

def load_model():
    global graph
    if graph is None:
        model_name = 'bccard_basic'
        LOAD_DIR = '/home/work/RECO/Models/BCCard_model'
        graph = build_graph(model_name, LOAD_DIR)

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login_post():
    username = request.form.get('username')
    password = request.form.get('password')
    if username in users and users[username] == password:
        session['username'] = username
        return redirect(url_for('home'))
    else:
        return render_template('login.html', error="Invalid username or password")

@app.route('/home')
def home():
    if 'username' in session:
        return render_template('home.html', username=session['username'])
    else:
        return redirect(url_for('login'))

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/chatbot', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        user_input = request.json.get('message')
        if not user_input:
            return jsonify({'error': 'No input provided'}), 400
        try:
            result = process_input(user_input)
            return jsonify(result)
        except json.JSONDecodeError as e:
            logging.error(f"{e}")
            return jsonify({"response": "runner.py에서 유효하지 않은 JSON이 반환되었습니다."}), 500
        except subprocess.CalledProcessError as e:
            logging.error(f"{e.stderr}")
            return jsonify({"response": "runner.py에서 오류가 발생했습니다."}), 500
        except Exception as e:
            logging.error(f"{e}")
            return jsonify({"response": "알 수 없는 오류가 발생했습니다."}), 500
    try:
        conn = sqlite3.connect('/home/work/RECO/Langgraph/Dataset/customercard.sqlite')
        query = """
            SELECT *
            FROM customer_card
            WHERE strftime('%m', REPLACE(거래일자, '.', '-')) = '10'
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        if df.empty:
            logging.warning("데이터가 없습니다.")
            category_summary = pd.DataFrame({'업종명': ['데이터 없음'], '매출금액': [0]})
            top5 = category_summary
            bottom5 = category_summary
        else:
            category_summary = df.groupby('업종명')['매출금액'].sum().reset_index()
            category_summary['매출금액'] = category_summary['매출금액'] / 10000
            sorted_summary = category_summary.sort_values(by='매출금액', ascending=False)
            top5 = sorted_summary.head(5)
            bottom5 = sorted_summary.tail(5)
        graph_data = {
            'labels': category_summary['업종명'].tolist(),
            'data': category_summary['매출금액'].tolist()
        }
        
        card_dir = '/home/work/RECO/Langgraph/Dataset/card_information_data_credit_50_1217.json'  # 가정
        user_id = 'C001'

        card_data, user_data = dataset_load(card_dir, user_id)

        reward_result = reward_calculator(card_data, user_data, 10, None)  # 상위 3개 카드 정보 반환(문자열 형태)

        return render_template(
            'chatbot.html',
            graph_data=graph_data,
            top5=top5.to_dict(orient='records'),
            bottom5=bottom5.to_dict(orient='records'),
            reward_result=reward_result  # 템플릿에 추가 결과 넘김
        )

    except Exception as e:
        logging.error(f"{e}")
        return render_template('chatbot.html', graph_data=None, top5=[], bottom5=[])

if __name__ == "__main__":
    load_model()
    config = RunnableConfig(recursion_limit=50, configurable={"thread_id": "1","checkpoint_id": "test_id"})
    state = {"messages": [],"next": "","context": "","user_id": "C001"}
    app.run(debug=False, port=8121, host = "0.0.0.0")
