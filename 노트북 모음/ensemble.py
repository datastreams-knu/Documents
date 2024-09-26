import os
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_upstage import UpstageEmbeddings
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from pinecone import Pinecone
from dotenv import load_dotenv
from langchain_upstage import ChatUpstage
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import Document
from langchain.vectorstores import FAISS

# .env 파일 로드
load_dotenv()

# 환경 변수에서 Pinecone API 키 및 인덱스 이름 가져오기
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
index_name = 'test'

# Pinecone API 설정 및 초기화
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(index_name)

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

# 스크래핑할 URL 목록 생성
now_number = 28226
urls = []
for number in range(now_number, now_number-30, -1):
    urls.append("https://cse.knu.ac.kr/bbs/board.php?bo_table=sub5_1&wr_id=" + str(number))

# URL에서 문서 추출
document_texts = extract_text_from_url(urls)

# 텍스트 분리기 초기화
class CharacterTextSplitter:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text):
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunks.append(text[i:i + self.chunk_size])
        return chunks

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# 텍스트 분리
texts = []
for doc in document_texts:
    if isinstance(doc, str):
        texts.extend(text_splitter.split_text(doc))
    else:
        raise TypeError("리스트 내 각 문서는 문자열이어야 합니다.")

# 1. Sparse Retrieval (TF-IDF)
def initialize_tfidf_model(texts):
    vectorizer = TfidfVectorizer()
    doc_vectors = vectorizer.fit_transform(texts)
    return vectorizer, doc_vectors

vectorizer, doc_vectors = initialize_tfidf_model(texts)

# 2. Dense Retrieval (Upstage 임베딩)
embedding_model = UpstageEmbeddings(model="solar-embedding-1-large")
dense_doc_vectors = np.array(embedding_model.embed_documents(texts))  # 문서 임베딩

# Pinecone에 문서 임베딩 저장
for i, embedding in enumerate(dense_doc_vectors):
    index.upsert([(str(i), embedding.tolist(), {"text": texts[i]})])  # 문서 ID, 임베딩 벡터, 메타데이터 추가

# 사용자 질문에 대한 AI 답변 생성 (앙상블 방식)
def get_best_docs(user_question):
    try:
        # Sparse Retrieval: TF-IDF 벡터화
        query_tfidf_vector = vectorizer.transform([user_question])
        tfidf_cosine_similarities = cosine_similarity(query_tfidf_vector, doc_vectors).flatten()

        # Dense Retrieval: Upstage 임베딩을 통한 유사도 계산
        query_dense_vector = np.array(embedding_model.embed_query(user_question))

        # Pinecone에서 가장 유사한 벡터 찾기
        pinecone_results = index.query(vector=query_dense_vector.tolist(), top_k=10, include_values=True, include_metadata=True)
        pinecone_similarities = [res['score'] for res in pinecone_results['matches']]
        pinecone_docs = [(res['id'], res['score'], res['metadata']) for res in pinecone_results['matches']]

        # TF-IDF에서 상위 10개 문서의 유사도만 가져옵니다.
        top_tfidf_indices = np.argsort(tfidf_cosine_similarities)[-10:][::-1]  # 상위 10개 인덱스
        tfidf_best_docs = [(texts[i], tfidf_cosine_similarities[i]) for i in top_tfidf_indices]
        
        # TF-IDF 유사도 배열 상위 10개로 한정
        tfidf_best_similarities = tfidf_cosine_similarities[top_tfidf_indices]

        # 두 유사도 배열 결합
        combined_similarities = np.concatenate((tfidf_best_similarities, np.array(pinecone_similarities)))

        # 가장 유사한 문서 인덱스 계산
        combined_best_doc_indices = np.argsort(combined_similarities)[-10:][::-1]

        # 결과 문서 목록 생성
        best_docs = []

        # TF-IDF 결과 추가
        for idx in combined_best_doc_indices:
            if idx < len(tfidf_best_docs):
                best_docs.append(tfidf_best_docs[idx])
            else:
                pinecone_index = idx - len(tfidf_best_docs)
                best_docs.append((pinecone_docs[pinecone_index][2]['text'], combined_similarities[idx]))

        return best_docs
    except Exception as e:
        return f"답변을 생성하는 중 오류가 발생했습니다: {e}"

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_answer_from_chain(best_docs) :
    global qa_chain

    doc_texts = [doc[0] for doc in best_docs]
    documents = [Document(page_content=text) for text in doc_texts]

    vector_store = FAISS.from_documents(documents, embedding_model)
    retriever = vector_store.as_retriever()
    llm = ChatUpstage()

    prompt = hub.pull("rlm/rag-prompt")

    qa_chain = (
         {"context": retriever | format_docs, "question": RunnablePassthrough()} 
         | prompt 
         | llm 
         | StrOutputParser()
     )

def get_ai_message(user_question) :
    best_docs = get_best_docs(user_question)
    get_answer_from_chain(best_docs)
    ai_message = qa_chain.invoke({"question" : user_question})
    return ai_message