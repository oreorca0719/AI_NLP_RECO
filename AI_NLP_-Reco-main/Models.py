from langgraph.graph import StateGraph, START, END
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import TypedDict, Annotated, Sequence
import pandas as pd
from sqlalchemy import create_engine
import torch
import os
import json
from transformers import BertTokenizer, BertForSequenceClassification
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import BertTokenizer, BertForTokenClassification
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import numpy as np

# Load model directly
def llama_basic(LOAD_DIR, hf_model="meta-llama/Llama-3.1-8B-Instruct"):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if LOAD_DIR and os.path.exists(LOAD_DIR):
        print("empty place")
        model = AutoModelForCausalLM.from_pretrained(LOAD_DIR,
                                                     device_map="auto",
                                                     torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(LOAD_DIR)
    else:
        model = AutoModelForCausalLM.from_pretrained(hf_model,
                                                     device_map="auto",
                                                     torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(hf_model)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    model.to(device)

    return model, tokenizer


def bccard_basic(LOAD_DIR, hf_model="BCCard/Llama-3.1-Kor-BCCard-Finance-8B"):
    print('bccard_basic is loading')
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"device setting: {device}")


    # if LOAD_DIR and os.path.exists(LOAD_DIR):
    #     print('경로를 통해 모델 로드')
    #     tokenizer = AutoTokenizer.from_pretrained(LOAD_DIR)
    #     model = AutoModelForCausalLM.from_pretrained(
    #         LOAD_DIR,
    #         device_map="auto",
    #         torch_dtype=torch.float16,
    #     )
    # else:
    model = AutoModelForCausalLM.from_pretrained(
        LOAD_DIR,
        device_map="auto",
        torch_dtype=torch.float16,
        )

    tokenizer = AutoTokenizer.from_pretrained(LOAD_DIR)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    model.to(device)

    print(f"Pad token: {tokenizer.pad_token}")
    print(f"Padding side: {tokenizer.padding_side}")
    print("Model and tokenizer loaded successfully!")
    return model, tokenizer

def load_selected_model(model_name, LOAD_DIR=None):
    if model_name == "llama_basic":
        return llama_basic(LOAD_DIR=LOAD_DIR)
    elif model_name == "bccard_basic":
        return bccard_basic(LOAD_DIR=LOAD_DIR)
    else:
        raise ValueError("Invalid model name. Choose 'llama' or 'BCCard'.")
    
def load_classifier(classifier_save_path="/home/work/RECO/Langgraph/Trained_models/Classifier/classification_model_kobert.pth",
                    classifier_mapfile = "/home/work/RECO/Langgraph/Trained_models/Classifier/tool_mappings.json"):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print("device : ", device)
    with open(classifier_mapfile, "r") as f:
        mappings = json.load(f)
        tool_to_idx = mappings['tool_to_idx']

    # KoBERT 모델 초기화
    basic_kobert = BertForSequenceClassification.from_pretrained(
        "monologg/kobert",
        num_labels=len(tool_to_idx)
    )

    # 모델 가중치 로드
    try:
        state_dict = torch.load(classifier_save_path, map_location=device)
        basic_kobert.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading model state dict: {e}")
        return None, None, None
        
    basic_kobert.to(device)

    classifier_tokenizer = BertTokenizer.from_pretrained("monologg/kobert")
    print('classifer_loading --------------------')
    return basic_kobert, classifier_tokenizer, tool_to_idx

def load_vector_embedding_model(model_name = "intfloat/multilingual-e5-large-instruct"):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Hugging Face의 사전 학습된 임베딩 모델과 토크나이저 로드
    embedding_model = HuggingFaceEmbeddings(model_name=model_name,
                                            model_kwargs={'device': device},
                                            encode_kwargs={'normalize_embeddings': True},
                                            show_progress=True)
    return embedding_model

# CategoryClassifier 클래스
class CategoryClassifier(nn.Module):
    def __init__(self, emb_dim=1024, hidden_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, q_emb_batch, cat_emb_all):
        batch_size = q_emb_batch.size(0)
        num_categories = cat_emb_all.size(0)
        q_emb_batch = q_emb_batch.unsqueeze(1)
        q_emb_batch = q_emb_batch.expand(batch_size, num_categories, q_emb_batch.size(-1))
        cat_emb_all = cat_emb_all.unsqueeze(0)
        cat_emb_all = cat_emb_all.expand(batch_size, num_categories, cat_emb_all.size(-1))
        x = torch.cat([q_emb_batch, cat_emb_all], dim=-1)
        x = x.view(batch_size * num_categories, -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = x.view(batch_size, num_categories)
        return x

def load_keyword_classifer(model_path, device):
    classifier = CategoryClassifier(emb_dim=1024, hidden_dim=512)
    classifier.load_state_dict(torch.load(model_path + '/category_classifier.pth'))
    classifier.to(device)

    # NumPy 배열을 파일에서 불러오기
    category_embeddings_np = np.load(model_path + '/category_embeddings.npy')

    # NumPy 배열을 PyTorch 텐서로 변환
    category_embeddings = torch.from_numpy(category_embeddings_np).to(device)

    return classifier, category_embeddings