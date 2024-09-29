import os
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_upstage import UpstageEmbeddings
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from pinecone import Pinecone
from langchain_upstage import ChatUpstage
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import Document
from langchain.vectorstores import FAISS

# Pinecone API 키와 인덱스 이름 선언
pinecone_api_key = 'cd22a6ee-0b74-4e9d-af1b-a1e83917d39e'  # 여기에 Pinecone API 키를 입력
index_name = 'test'

# Upstage API 키 선언
upstage_api_key = 'up_pGRnryI1JnrxChGycZmswEZm934Tf'  # 여기에 Upstage API 키를 입력

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
            text = "\n".join([para.get_text() for para in paragraphs])
            return text, url  # 문서 텍스트와 URL 반환
        except Exception as e:
            print(f"Error processing {url}: {e}")
            return None, url

    with ThreadPoolExecutor() as executor:
        results = executor.map(fetch_text, urls)

    all_texts = [(text, url) for text, url in results if text]
    return all_texts

# 스크래핑할 URL 목록 생성
now_number = 28234
urls = []
for number in range(now_number, now_number-100, -1):
    urls.append("https://cse.knu.ac.kr/bbs/board.php?bo_table=sub5_1&wr_id=" + str(number))

# URL에서 문서 추출
document_texts = extract_text_from_url(urls)

# 텍스트 분리기 초기화
class CharacterTextSplitter:
    def __init__(self, chunk_size=800, chunk_overlap=150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text):
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunks.append(text[i:i + self.chunk_size])
        return chunks

text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=150)

# 텍스트 분리 및 URL 매핑을 확실히 하는 코드
texts = []
doc_urls = []  # 문서와 해당 URL을 매핑하기 위한 리스트
for doc, url in document_texts:
    if isinstance(doc, str):
        split_texts = text_splitter.split_text(doc)
        texts.extend(split_texts)
        doc_urls.extend([url] * len(split_texts))  # 분리된 각 텍스트에 동일한 URL 적용
    else:
        raise TypeError("리스트 내 각 문서는 문자열이어야 합니다.")

# 1. Sparse Retrieval (TF-IDF)
def initialize_tfidf_model(texts):
    vectorizer = TfidfVectorizer()
    doc_vectors = vectorizer.fit_transform(texts)
    return vectorizer, doc_vectors

vectorizer, doc_vectors = initialize_tfidf_model(texts)

# 2. Dense Retrieval (Upstage 임베딩)
embedding_model = UpstageEmbeddings(model="solar-embedding-1-large", api_key=upstage_api_key)  # Upstage API 키 사용
dense_doc_vectors = np.array(embedding_model.embed_documents(texts))  # 문서 임베딩

# Pinecone에 문서 임베딩 저장 (문서 텍스트와 URL을 반드시 포함)
for i, embedding in enumerate(dense_doc_vectors):
    metadata = {
        "text": texts[i],
        "url": doc_urls[i]  # URL을 항상 메타데이터에 포함
    }
    index.upsert([(str(i), embedding.tolist(), metadata)])  # 문서 ID, 임베딩 벡터, 메타데이터 추가

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
        pinecone_docs = [(res['id'], res['score'], res['metadata']['text'], res['metadata'].get('url', 'No URL')) for res in pinecone_results['matches']]

        # TF-IDF에서 상위 10개 문서의 유사도만 가져옵니다.
        top_tfidf_indices = np.argsort(tfidf_cosine_similarities)[-10:][::-1]  # 상위 10개 인덱스
        tfidf_best_docs = [(texts[i], tfidf_cosine_similarities[i], doc_urls[i]) for i in top_tfidf_indices]  # URL 포함

        # 두 유사도 배열 결합
        combined_similarities = np.concatenate((tfidf_cosine_similarities[top_tfidf_indices], np.array(pinecone_similarities)))

        # 가장 유사한 문서 인덱스 계산
        combined_best_doc_indices = np.argsort(combined_similarities)[-10:][::-1]

        # 결과 문서 목록 생성
        best_docs = []

        for idx in combined_best_doc_indices:
            if idx < len(tfidf_best_docs):
                best_docs.append(tfidf_best_docs[idx])
            else:
                pinecone_index = idx - len(tfidf_best_docs)
                best_docs.append((pinecone_docs[pinecone_index][2], combined_similarities[idx], pinecone_docs[pinecone_index][3]))  # 텍스트와 URL

        return best_docs
    except Exception as e:
        return f"답변을 생성하는 중 오류가 발생했습니다: {e}"

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# AI 답변을 사용자에게 전달하기 위한 함수
def get_answer_from_chain(best_docs, user_question):
    global qa_chain

    doc_texts = [doc[0] for doc in best_docs]
    doc_urls = [doc[2] for doc in best_docs]  # URL을 별도로 저장
    documents = [Document(page_content=text, metadata={"url": url}) for text, url in zip(doc_texts, doc_urls)]

    # FAISS 벡터스토어로부터 retriever 설정
    vector_store = FAISS.from_documents(documents, embedding_model)
    retriever = vector_store.as_retriever()

    # Upstage의 LLM과 프롬프트 설정
    llm = ChatUpstage(api_key=upstage_api_key)
    prompt = hub.pull("rlm/rag-prompt")

    # QA 체인 생성
    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # AI 답변 생성 과정에서 retriever가 참조한 문서들 추적
    retriever_docs = retriever.get_relevant_documents(user_question)

    # retriever에서 사용된 문서들의 URL 추출
    retriever_urls = [doc.metadata.get("url", "No URL") for doc in retriever_docs]

    return qa_chain, retriever_urls

# AI 답변과 URL을 함께 반환하는 함수
def get_ai_message(user_question):
    # 가장 유사한 문서 가져오기
    best_docs = get_best_docs(user_question)

    # AI 답변 생성
    qa_chain, retriever_urls = get_answer_from_chain(best_docs, user_question)
    ai_message = qa_chain.invoke({"question": user_question})

    # 참조한 문서의 URL 포함 형식으로 반환
    doc_references = "\n\n".join([f"참고 문서 URL: {url}" for url in retriever_urls if url != 'No URL'])

    # AI의 답변과 참조 URL을 함께 반환
    return f"{ai_message}\n\n{doc_references}"