import bs4
from langchain_community.document_loaders import WebBaseLoader
from apscheduler.schedulers.blocking import BlockingScheduler
import logging
from datetime import datetime
import requests
import re
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_upstage import UpstageEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain import hub
from langchain_upstage import ChatUpstage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os

#!!!!!! 크롤링 결과를 split해서 넣는 것까지는 가능. 정확도 개선이 필요함.

# 로깅 설정
logging.basicConfig(filename='crawler.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# 최신 wr_id를 저장할 파일
LAST_ID_FILE = 'last_crawled_id.txt'
all_docs = []
splits = []

load_dotenv()

# Embedding 및 Pinecone 설정
embeddings = UpstageEmbeddings(
    model="solar-embedding-1-large"
)
index_name = 'test'
pinecone_api_key = os.environ.get("PINECONE_API_KEY")

# 최신 wr_id를 가져오는 함수
def get_latest_wr_id():
    url = "https://cse.knu.ac.kr/bbs/board.php?bo_table=sub5_1"
    response = requests.get(url)
    if response.status_code == 200:
        match = re.search(r'wr_id=(\d+)', response.text)
        if match:
            return int(match.group(1))
    return None

# 마지막으로 크롤링한 wr_id를 저장하는 함수
def save_last_crawled_id(wr_id):
    with open(LAST_ID_FILE, 'w') as f:
        f.write(str(wr_id))

# 마지막으로 크롤링한 wr_id를 불러오는 함수
def load_last_crawled_id():
    try:
        with open(LAST_ID_FILE, 'r') as f:
            return int(f.read().strip())
    except FileNotFoundError:
        return None

# 페이지 크롤링 및 처리 함수
def crawl_pages():
    global docs, splits
    try:
        latest_wr_id = get_latest_wr_id()
        if latest_wr_id is None:
            logging.error("Fail to bring recent wr_id.")
            return

        last_crawled_id = load_last_crawled_id()
        
        if last_crawled_id is None or latest_wr_id > last_crawled_id:
            start_id = latest_wr_id
            end_id = last_crawled_id if last_crawled_id else latest_wr_id - 10  # 처음 실행 시 최근 10개만 크롤링
            base_url = "https://cse.knu.ac.kr/bbs/board.php?bo_table=sub5_1&wr_id="
            
            urls = [f"{base_url}{i}" for i in range(start_id, end_id - 1, -1)]
            
            for url in urls :
                loader = WebBaseLoader(
                    web_paths=(url,),
                    bs_kwargs=dict(
                        parse_only=bs4.SoupStrainer(
                            "div",
                            attrs={"id": ["bo_v_con", "bo_v_title"]},
                        )
                    )
                )
                docs = loader.load()
                all_docs.extend(docs)
                
            save_last_crawled_id(latest_wr_id)
            split_documents()
            create_vector_store()
            chain()
        else :
            logging.info("There's no new notifications")

    except Exception as e:
        logging.error(f"Error occurs when crawlling: {str(e)}")

# 문서 분할 함수
def split_documents():
    global splits
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=50,
    )
    splits = text_splitter.split_documents(all_docs)

def create_vector_store():
    global splits, database
    database = PineconeVectorStore.from_documents(splits, embeddings, index_name=index_name)

# 문서 내용을 포맷하는 함수
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Vector Store에 저장 및 Chain 생성 함수
def chain():
    global qa_chain
    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatUpstage()

    retriever = database.as_retriever()

    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()} 
        | prompt 
        | llm 
        | StrOutputParser()
    )
    
    logging.info("QA chain created successfully.")

# def get_ai_message(user_message) :
#     split_documents()
#     create_vector_store()
#     chain()
#     ai_message = qa_chain.invoke({"question" :user_message})
#     return ai_message

# 스케줄러 인스턴스 생성
scheduler = BlockingScheduler()

# 하루에 한 번 크롤링 실행: 매일 00:00에 크롤링
scheduler.add_job(crawl_pages, 'cron', hour=0, minute=0)

# 프로그램 시작 시 즉시 한 번 실행
scheduler.add_job(crawl_pages, 'date', run_date=datetime.now())

logging.info("Start scheduling.")
scheduler.start()
