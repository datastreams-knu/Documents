import os
import requests
from bs4 import BeautifulSoup
from langchain_upstage import UpstageEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_upstage import ChatUpstage
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from concurrent.futures import ThreadPoolExecutor
import re
from datetime import datetime

# Upstage API 키 설정
os.environ["UPSTAGE_API_KEY"] = "up_coecXafSJVG1v17EEZ3lxjFbZ8xcD"

# 모델 초기화
embedding_model = UpstageEmbeddings(model="solar-embedding-1-large")
upstage_llm = ChatUpstage(api_key=os.getenv("UPSTAGE_API_KEY"))

# URL에서 텍스트를 추출하는 함수
def extract_text_from_urls(urls):
    def fetch_text(url):
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            return "\n".join([para.get_text() for para in soup.find_all('p')])
        except Exception as e:
            print(f"Error processing {url}: {e}")
            return ""
    with ThreadPoolExecutor() as executor:
        return list(executor.map(fetch_text, urls))

# 최신 wr_id를 얻는 함수
def get_latest_wr_id():
    url = "https://cse.knu.ac.kr/bbs/board.php?bo_table=sub5_1"
    response = requests.get(url)
    if response.status_code == 200:
        match = re.search(r'wr_id=(\d+)', response.text)
        if match:
            return int(match.group(1))
    return None

# 스크래핑할 URL 목록 생성
now_number = get_latest_wr_id()
urls = [f"https://cse.knu.ac.kr/bbs/board.php?bo_table=sub5_1&wr_id={num}" for num in range(now_number, now_number-30, -1)]

# URL에서 문서 추출
document_texts = extract_text_from_urls(urls)

# 텍스트 분리기
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = [chunk for doc in document_texts for chunk in text_splitter.split_text(doc) if doc]

# FAISS 벡터스토어 초기화
vectorstore = FAISS.from_texts(texts, embedding_model)

# ConversationBufferMemory 초기화
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# ConversationalRetrievalChain 생성
qa = ConversationalRetrievalChain.from_llm(
    llm=upstage_llm,
    retriever=vectorstore.as_retriever(),
    memory=memory
)

def get_ai_message(user_question):
    try:
        # 현재 시간 가져오기
        current_time = datetime.now().strftime("%Y-%m-%d")
        
        # 사용자 질문에 현재 시간 포함
        user_question_with_time = f"{user_question} (현재 시간: {current_time})"
        
        # 대화 기록을 포함하여 질문에 답변
        result = qa({"question": user_question_with_time})
        return result['answer']
    except Exception as e:
        return f"답변을 생성하는 중 오류가 발생했습니다: {e}"
