import os
import requests
from bs4 import BeautifulSoup
from langchain_upstage import UpstageEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_upstage import ChatUpstage
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from concurrent.futures import ThreadPoolExecutor

# Upstage API 키 설정
os.environ["UPSTAGE_API_KEY"] = "up_coecXafSJVG1v17EEZ3lxjFbZ8xcD"

# 모델을 초기화하기 위한 변수들
embedding_model = None
vectorstore = None
upstage_llm = None
qa_chain = None

# URL에서 텍스트를 추출하는 함수
def extract_text_from_url(urls):
    all_texts = []

    def fetch_text(url):
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            paragraphs = soup.find_all('p')
            return "\n".join([para.get_text() for para in paragraphs])
        except Exception as e:
            print(f"Error processing {url}: {e}")
            return None

    # 멀티스레드를 사용하여 여러 URL에서 동시에 텍스트를 추출
    with ThreadPoolExecutor() as executor:
        results = executor.map(fetch_text, urls)

    all_texts = [text for text in results if text]
    return all_texts

# 모델 초기화 함수
def initialize_models(texts):
    global embedding_model, vectorstore, upstage_llm, qa_chain
    if embedding_model is None:
        embedding_model = UpstageEmbeddings(model="solar-embedding-1-large")
    if vectorstore is None:
        vectorstore = FAISS.from_texts(texts, embedding_model)
    if upstage_llm is None:
        upstage_llm = ChatUpstage(api_key=os.getenv("UPSTAGE_API_KEY"))
    if qa_chain is None:
        qa_chain = load_qa_chain(llm=upstage_llm, chain_type="stuff")

# 스크래핑할 URL 목록 생성
now_number = 28233
urls = []
for number in range(now_number, now_number-30, -1):
    urls.append("https://cse.knu.ac.kr/bbs/board.php?bo_table=sub5_1&wr_id=" + str(number))

# URL에서 문서 추출
document_texts = extract_text_from_url(urls)

# 텍스트 분리기 초기화
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# 임베딩할 텍스트 준비
if isinstance(document_texts, list):
    texts = []
    for doc in document_texts:
        if isinstance(doc, str):
            texts.extend(text_splitter.split_text(doc))
        else:
            raise TypeError("리스트 내 각 문서는 문자열이어야 합니다.")
else:
    raise TypeError("document_texts는 문자열 리스트여야 합니다.")

# 텍스트를 이용해 모델 초기화
initialize_models(texts)

# RetrievalQA 체인 생성
qa = RetrievalQA(combine_documents_chain=qa_chain, retriever=vectorstore.as_retriever())

# get_ai_message 함수
def get_ai_message(user_question):
    try:
        ai_message = qa.invoke(user_question)
        return ai_message.get("result")
    except Exception as e:
        return f"답변을 생성하는 중 오류가 발생했습니다: {e}"
