import sys
import json
from settings import *
from Models import *
from Graph import *
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage
import logging

# 모델 및 그래프 초기화
model_name = 'bccard_basic'
LOAD_DIR = '/home/work/RECO/Models/BCCard_model'
graph = build_graph(model_name, LOAD_DIR)
print('Graph build 완료')

config = RunnableConfig(
    recursion_limit=10,
    configurable={"thread_id": "1", "checkpoint_id": "test_id"}
)

state = {
    "messages": [],
    "next": "",
    "context": "",
    "user_id": "C001"
}

def process_input(user_input):
    """
    사용자 입력을 처리하여 그래프를 실행하고 결과를 반환합니다.
    """
    # 사용자 입력 추가
    state["messages"].append(HumanMessage(content=user_input))
    
    try:
        # 그래프 실행
        for event in graph.stream(state, config=config):
            for value in event.values():
                print("Event Value:", value)
                state.update(value)

                state["messages"] = value["messages"]
                last_message = value["messages"][-1]

                if isinstance(last_message, AIMessage):
                    return {"response": last_message.content}

                if (state['next'] == 'Reward_logic') and (len(state['context'])>2):
                    return {"response": state['context']}

        # 기본 응답 처리
        return {"response": "Assistant: Undefined"}

    except Exception as e:
        logging.error(f"그래프 실행 중 오류 발생: {e}")
        return {"response": "오류가 발생했습니다. 다시 시도해주세요."}

if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     print(json.dumps({"error": "No input provided"}))
    #     sys.exit(1)

    # user_input = sys.argv[1]
    # result = process_input(user_input)
    # print(json.dumps(result))

    QUESTIONS = "유튜브 프리미엄 혜택 받을 수 있는 카드 추천해줘"
    
    result = process_input(QUESTIONS)
    # print("output : ", result)
    