import os
import json
import requests
import re

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache
from PyPDF2 import PdfReader
from line_profiler import LineProfiler
from typing import Any, List, Mapping, Optional

import sqlite3
import sys
sys.modules['sqlite3'] = sys.modules.pop('sqlite3')
import streamlit as st

# 세션 상태 초기화
for key in ["loading_text", "retriever", "vectorstore_cache"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "vectorstore_cache" else {}

st.set_page_config(
    page_title="AK아이에스 복지제도 알리미",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

class LlmClovaStudio(LLM):
    """
    Custom LLM class for using the ClovaStudio API.
    """
    host: str
    api_key: str
    api_key_primary_val: str
    request_id: str

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.host = kwargs.get('host')
        self.api_key = kwargs.get('api_key')
        self.api_key_primary_val = kwargs.get('api_key_primary_val')
        self.request_id = kwargs.get('request_id')

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None
    ) -> str:
        """
        Make an API call to the ClovaStudio endpoint using the specified
        prompt and return the response.
        """
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")

        headers = {
            "X-NCP-CLOVASTUDIO-API-KEY": self.api_key,
            "X-NCP-APIGW-API-KEY": self.api_key_primary_val,
            "X-NCP-CLOVASTUDIO-REQUEST-ID": self.request_id,
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "text/event-stream"
        }

        sys_prompt = """당신은 AK아이에스의 사내 규정 및 업무 가이드에 대해 간결하고 정확한 답변을 제공합니다.
        - 회사의 규정, 정책, 업무 가이드에 대해서만 답변합니다.
        - 제공된 정보(Context)만 사용하고, 추측하거나 추가 정보를 포함하지 않습니다.
        - 질문에 대한 답변이 불가능하면 '잘 모르겠습니다.'라고 답변합니다.
        - 업무와 무관한 질문에는 '업무 관련 질문만 답변 가능합니다.'라고 답변합니다.
        - 감사 인사나 칭찬에는 '감사합니다!'라고만 답변하세요.
        - 여름, 하계와 같이 의미가 비슷한 단어들을 기반으로 질문을 해석하고 답변하세요.
        """

        preset_text = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}]

        request_data = {
            "messages": preset_text,
            "topP": 0.6,
            "topK": 0,
            "maxTokens": 256,
            "temperature": 0.3,
            "repeatPenalty": 1.2,
            "stopBefore": [],
            "includeAiFilters": False
        }

        try:
            response = requests.post(
                self.host + "/testapp/v1/chat-completions/HCX-003",
                headers=headers,
                json=request_data,
                stream=True
            )
            response.raise_for_status()  # 이 줄이 4XX 또는 5XX 오류를 감지합니다.
        except requests.exceptions.HTTPError as e:
            return f"Error in API request: {str(e)}"

        # 스트림에서 마지막 'data:' 라인을 찾기 위한 로직
        last_data_content = ""

        for line in response.iter_lines():
            if line:
                decoded_line = line.decode("utf-8")
                if '"data":"[DONE]"' in decoded_line:
                    break
                if decoded_line.startswith("data:"):
                    last_data_content = json.loads(decoded_line[5:])["message"]["content"]

        return last_data_content

llm = LlmClovaStudio(
    host='https://clovastudio.stream.ntruss.com',
    api_key='NTA0MjU2MWZlZTcxNDJiY4bBoTYomZr3ZI2tRU9B/sr5cjJ/cclnIpCb4trTnTT7',
    api_key_primary_val='n3d70DP5GUuNyofxZhTGBErzwccQtwaB1rVfPZxt',
    request_id='59cf6478-2d5f-42e0-a562-a7a31e623d41' #HCX-003
)

def extract_text_from_pdfs(folder_path, start_page=None, end_page=None):
    all_text = ""
    
    # 폴더 내 모든 파일을 검색하여 PDF 파일만 처리
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            with open(pdf_path, 'rb') as file:
                reader = PdfReader(file)
                text = ""

                if not start_page:
                    start_page = 0
                else:
                    start_page -= 1

                if not end_page or end_page > len(reader.pages):
                    end_page = len(reader.pages)

                for page_num in range(start_page, end_page):
                    text += reader.pages[page_num].extract_text()

                all_text += text  # 각 PDF 파일의 텍스트를 이어붙임
    
    return all_text

def retrieve_docs(text, model_index=0):
    if not text:
        return "No text found in the PDF file."

    model_list = [
        'bespin-global/klue-sroberta-base-continue-learning-by-mnr',
        'BAAI/bge-m3',
        'All-MiniLM-L6-v2',
        'sentence-transformers/paraphrase-MiniLM-L6-v2'
    ]

    embeddings = st.session_state.vectorstore_cache.get('embeddings')
    if embeddings is None:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_list[model_index],
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True},
        )
        st.session_state.vectorstore_cache['embeddings'] = embeddings

    vectorstore_path = f'vectorstore_{model_index}'
    vectorstore = st.session_state.vectorstore_cache.get(vectorstore_path)

    if vectorstore is None:
        if os.path.exists(os.path.join(vectorstore_path, "index")):
            vectorstore = Chroma(persist_directory=vectorstore_path, embedding_function=embeddings)
        else:
            docs = [Document(page_content=text)]
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
            splits = text_splitter.split_documents(docs)
            vectorstore = Chroma.from_documents(splits, embeddings, persist_directory=vectorstore_path)
            vectorstore.persist()

        st.session_state.vectorstore_cache[vectorstore_path] = vectorstore

    return vectorstore.as_retriever()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def rag_chain(question):
    if st.session_state.loading_text is None:
        st.session_state.loading_text = extract_text_from_pdfs('Manual/')

    if st.session_state.retriever is None:
        st.session_state.retriever = retrieve_docs(st.session_state.loading_text)

    retriever = st.session_state.retriever
    if isinstance(retriever, str):
        return f"Error: {retriever}"

    retrieved_docs = retriever.get_relevant_documents(question)
    formatted_context = format_docs(retrieved_docs)
    formatted_prompt = f"Question: {question}\n\nContext: {formatted_context}"

    response = llm.invoke(formatted_prompt)
    return response

def generate_response(question):
    result = rag_chain(question)
    return re.sub(r"^(Answer:|답변:|답변|:|Answer)\s*", "", result)

set_llm_cache(InMemoryCache())
st.title("AK아이에스 복리후생제도 알리미")
st.markdown("현재 챗봇은 **테스트용**으로 운영되고 있습니다. 사용에 참고 바랍니다.")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "무엇을 도와 드릴까요?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    if prompt.strip():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        msg = generate_response(prompt)
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("assistant").write(msg)
    else:
        st.warning("질문을 입력해주세요.")
