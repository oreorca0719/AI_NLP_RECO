# Huggingface 로그인
from huggingface_hub import login
from dotenv import load_dotenv
import os
import json

def get_api_keys():

    load_dotenv(dotenv_path='/home/work/RECO/Langgraph/.env')

    HF_TOKEN = os.getenv("PRHF_TOKEN")
    TRAVILY_KEY = os.getenv("TRAVILY_API_KEY")

    return HF_TOKEN, TRAVILY_KEY

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

