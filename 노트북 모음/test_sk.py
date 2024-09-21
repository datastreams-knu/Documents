import os
import requests
from bs4 import BeautifulSoup
from langchain_upstage import UpstageEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_upstage import ChatUpstage
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from concurrent.futures import ThreadPoolExecutor
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub

os.environ["UPSTAGE_API_KEY"] = "up_coecXafSJVG1v17EEZ3lxjFbZ8xcD"
embedding_model = UpstageEmbeddings(
    model="solar-embedding-1-large"
)

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

    with ThreadPoolExecutor() as executor:
        results = executor.map(fetch_text, urls)

    all_texts = [text for text in results if text]

    return all_texts

def load_and_split() :
    global texts
    now_number = 28226
    urls = []
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    for number in range(now_number, now_number-20, -1):
        urls.append("https://cse.knu.ac.kr/bbs/board.php?bo_table=sub5_1&wr_id=" + str(number))

    document_text = extract_text_from_url(urls)

    if isinstance(document_text, list):
        texts = []
        for doc in document_text:
            if isinstance(doc, str):
                texts.extend(text_splitter.split_text(doc))
            else:
                raise TypeError("Each document in the list must be a string")
    else:
        raise TypeError("document_text must be a list of strings")

def store_VS() :
    global vectorstore
    vectorstore = FAISS.from_texts(texts, embedding_model)

def format_docs(docs) :
    # 검색한 문서 결과를 하나의 문단으로 합칩니다.
    return "\n\n".join(doc.page_content for doc in docs)

def chaining() :
    global qa_chain
    llm = ChatUpstage(api_key=os.getenv("UPSTAGE_API_KEY"))
    retriever = vectorstore.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")

    qa_chain = load_qa_chain(llm=llm, chain_type="stuff")

    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()} 
        | prompt 
        | llm 
        | StrOutputParser()
    )

def get_ai_message(user_message) :
    load_and_split()
    store_VS()
    chaining()

    ai_message = qa_chain.invoke({"question" : user_message})
    return ai_message