from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
import torch.nn.functional as F

from Tools import *
from Models import *
from settings import *

# 메모리 저장소 생성
memory = MemorySaver()

# 상태 정의
class State(TypedDict):
    messages: Annotated[list, add_messages]
    next: str
    context:str
    user_id: str

def initialize_graph_resources(model_name, LOAD_DIR):
    """
    그래프에서 필요한 리소스를 초기화하는 함수
    """
    global classifier_model, classifier_tokenizer, tool_to_idx, device
    global retriever, chatbot_model, chatbot_tokenizer, keyword_model, keyword_tokenizer
    global vector_embedding_moedel, loaded_card_vectors, db
    global category_classifier, category_embeddings, json_dict

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # Classifier 모델 로드
    classifier_model, classifier_tokenizer, tool_to_idx = load_classifier()
    chatbot_model, chatbot_tokenizer = load_selected_model(model_name, LOAD_DIR)
    vector_embedding_moedel, loaded_card_vectors, db = get_vectorDB_tool()
    category_classifier, category_embeddings = load_keyword_classifer('/home/work/RECO/Models/Category_classifier', device)
    json_dict = load_json_to_dict('/home/work/RECO/Models/Category_classifier/target_list_for_category_n_store.json')

def vectorDB_search_node(state:State) -> State:
    print("==================== VectorDB Searcher ===================")
    print("VectorDB Node Input State:", state)
    last_messages = state['messages'][-1].content
    question_vector = vector_embedding_moedel.embed_query(last_messages)

    most_similar_card, similarity = find_most_similar(question_vector, loaded_card_vectors)
    
    retriever = db.as_retriever(search_kwargs={'filter': {"card_name": most_similar_card}, 'k': 1})
    retrieved_docs = retriever.invoke(last_messages)

    page_contents = [doc.page_content for doc in retrieved_docs]

    state['context'] = page_contents
    print("VectorDB Node Input State:", state)
    return state

def classification_node(state:State) -> State:
    print("==================== Classification ===================")
    print("Classification Node Input State:", state)

    last_messages = state['messages'][-1].content

    inputs = classifier_tokenizer.encode_plus(
        last_messages,
        max_length = 64,
        return_tensors="pt",
        padding='max_length',
        truncation=True,
    ).to(device)

    classifier_model.eval()
    with torch.no_grad():
        outputs = classifier_model(**inputs)  # 로짓(logits) 반환
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)
        predicted_label_idx = torch.argmax(probs, dim=-1).item()

    best_match = [label for label, idx in tool_to_idx.items() if idx == predicted_label_idx][0]
    state['next'] = best_match
    print("Classification Node Output State:", state)
    return state

def web_search_node(state:State) -> State:
    print("==================== Web Searcher ===================")
    print("Web Node Input State:", state)
    last_messages = state['messages'][-1].content

    _, TRAVILY_KEY = get_api_keys()
    tavily_client = TavilyClient(api_key=TRAVILY_KEY)
    query = get_web_tool(last_messages, chatbot_model, chatbot_tokenizer)

    search_results = tavily_client.search(
                query=query,
                max_results=2,
                include_answer=False,
                include_raw_content=False
            )

    contents = [result['content'] for result in search_results['results']]
    contents_output = "\n\n".join([f"content {i + 1} \n{content}" for i, content in enumerate(contents)])

    state['context'] = contents_output
    print("Web Node Output State:", state)
    return state

def reward_logic_node(state:State) -> State:
    print("==================== Reward Logic ===================")
    print("reward logic Input State:", state)
    last_messages = state['messages'][-1].content

    # 업종 상호명 추출
    pred_cat = classify_category(category_classifier, category_embeddings, vector_embedding_moedel, device, last_messages)
    pred_keyword = extract_store(last_messages, pred_cat, json_dict)
    print("업종상호명 : ", pred_keyword)

    reward_context = reward_logic(state['user_id'], pred_keyword)

    state['context'] = reward_context
    print("reward logic Output State:", state)
    return state

def get_user_template(last_message, context):

    user_template = """
질문:
{question}

제공된 정보:
{context}

답변:
"""
    return user_template.format(question=last_message, context=context)

# Step 3: Chatbot node 정의
def chatbot_node(state: State) -> State:
    print("==================== Chatbot ===================")
    print("Chatbot Node Input State:", state)

    messages = state['messages']
    ref_context = state['context']
    tool = state['next']
    last_message = state['messages'][-1].content

    system_template = """
당신은 카드에 대한 정보를 전문적으로 제공하는 AI 금융 비서입니다.
아래 예시와 같이 제공된 정보만을 사용하여 정확하게 답변을 생성해주세요.
제공된 정보와 질문의 연관성이 부족할 경우, "주어진 내용을 바탕으로 답변할 수 없습니다"라고 답변하세요.

---
예시
질문: 간편결제 앱 사용 시 할인 혜택이 큰 신용카드는 무엇인가요?

제공된 정보:
신한카드 Deep On Platinum+ 카드플레이트. 신한카드 Deep On Platinum+. 연회비. MASTER 3만 3천원. UPI 3만원. 간편결제(Pay) 최대 20% 할인 생활서비스 20% 할인. 카드 온라인 카드 신청 시 첫 해 연회비 100% 캐시백 씨티 캐시백 적립 씨티 캐시백 적립 내용 펼치기 온라인쇼핑/간편결제, 해외, 휴대폰요금/스트리밍 영역별 7% 특별캐시백, 월 최대 2만 5천 캐시백 적립 전월 실적 50만원 이상 시 특별 캐시백 적립 내용이 가맹점과 적립혜택/월 한도 안내로 구성되어있습니다. | 구분 | 전월 실적 50만원 이상 시 특별 캐시백 적립 | 연간 사용실적 및 적립 보너스 캐시백 안내로 구성되어 있습니다. 장기카드대출(카드론), 단기카드대출(현금서비스), 연회비, 지방세, 도시가스요금, 포인트 사용분, 해외 사용수수료를 포함한 각종 제수수료, 할인 이용액 및 당사의 모든 할인 서비스 및 무이자할부 이용 금액 캐시백 적립 대상:일시불 및 유이자 할부 금액(매입기준) 특별 캐시백 적립한도를 초과하는 사용액에 대해서는 기본 포인트 적립률 적용(기본 캐시백 적립 한도 없음)되며, 씨티 캐시백 적립 시 기본적립과 특별적립은 중복 적립 불가합니다. 씨티 캐시백 조회 내용 펼치기 온라인 카드 신청 시 첫 해 연회비 100% 캐시백 혜택 좋은 신용카드 추천 5종 순위 2024 월 최대 혜택 4만원) 많이 쓰는 영역 30% 할인 - 삼성id on카드 월 최대 혜택 5만원) 공과금, 생활패턴에 맞는 할인 - 신한 Mr.Life 월 최대 혜택 4.9만원) 빈틈없는 쇼핑 할인 - KB 탄탄대로 올쇼핑 나에게 맞는 할인, 혜택 좋은 신용카드 추천 삼성id on카드 무실적 할인, 전월실적 없는 신용카드 추천 농협 지금 더 페이 카드 전월실적 40만원을 채우면 월 최대 혜택 2.2만원을 받을 수 있는데요, 홈쇼핑, 대형마트, 인터넷 면세점, 온라인쇼핑몰과 같이 다양한 쇼핑영역에서 10% 할인을 제공하고 있습니다. 관리비 할인과 같은 혜택 덕분에 4인 가족이나 아파트에 사시는 분들께 추천해 드리며, 연회비는 낮은 데 비해 다양한 영역에서 골고루 할인 받을 수 있으니 부모님들이 많이 선호하기도 합니다.

답변: 제공된 정보를 바탕으로 간편결제 앱 사용 시 할인 혜택이 큰 신용카드는 다음과 같습니다
1. **신한카드 Deep On Platinum+**
   - 간편결제 최대 20% 할인
2. **삼성id on카드**
   - 무실적 할인
   - 월 최대 혜택 5만원
---
"""

    user_template = get_user_template(last_message, ref_context)

    initial_prompt = [
        {
            "role": "system",
            "content": (
                system_template
            ),
        },
        {
            "role":"user",
            "content":user_template,
        }
    ]

    # Apply chat template
    prompt_message = chatbot_tokenizer.apply_chat_template(
        initial_prompt,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = chatbot_tokenizer(prompt_message,
                       return_tensors="pt",
                       padding=True,
                       truncation=True,
                       return_attention_mask=True
                       ).to(device)
    eos_token_id = [chatbot_tokenizer.eos_token_id, chatbot_tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    outputs = chatbot_model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=6100,# 응답 길이 제한
            temperature=0.7,  # 다양성 조정
            top_p=0.8,  # 높은 확률 토큰에 집중
            eos_token_id=eos_token_id,  # 종료 토큰
            pad_token_id=chatbot_tokenizer.pad_token_id,
        )

    generated_text = chatbot_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 필요한 응답만 추출하는 코드
    start_index = generated_text.rfind("assistant") + len("assistant\n\n")
    response = generated_text[start_index:].strip() if "assistant" in generated_text else generated_text.strip()

    state["messages"].append(AIMessage(content=response))

    print("Chatbot Node Ouput State:", state)

    return state

def route_condition(state):
    if state["next"] == "chatbot":
        return "chatbot"
    elif state["next"] == "VectorDB":
        return "vectordb_searcher"
    elif state["next"] == "Reward_logic":
        return "reward_calculator"
    else:
        return "web_searcher"

def build_graph(model_name, LOAD_DIR):
    initialize_graph_resources(model_name, LOAD_DIR)

    graph_builder = StateGraph(State)

    graph_builder.add_node("classifier", classification_node)
    graph_builder.add_node("chatbot", chatbot_node)
    graph_builder.add_node("web_searcher", web_search_node)
    graph_builder.add_node("vectordb_searcher", vectorDB_search_node)
    graph_builder.add_node("reward_calculator", reward_logic_node)

    graph_builder.add_edge(START, "classifier")
    graph_builder.add_conditional_edges("classifier", route_condition, {
        "web_searcher": "web_searcher",
        "vectordb_searcher":"vectordb_searcher",
        "reward_calculator":"reward_calculator"
    })
    graph_builder.add_edge("web_searcher", "chatbot")
    graph_builder.add_edge("vectordb_searcher", "chatbot")
    graph_builder.add_edge("reward_calculator", END)
    graph_builder.add_edge("chatbot", END)

    graph = graph_builder.compile(checkpointer=memory)
    return graph