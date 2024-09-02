import os
import json
import requests
import re

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache
from PyPDF2 import PdfReader
from typing import Any, List, Mapping, Optional

import streamlit as st

# 세션 상태 초기화
if "loading_text" not in st.session_state:
    st.session_state.loading_text = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "vectorstore_cache" not in st.session_state:
    st.session_state.vectorstore_cache = {}

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

        sys_prompt = """당신은 AK아이에스의 사내 규정 및 업무 가이드에 대한 전문 답변을 제공하는 역할을 맡고 있습니다.
        - 회사의 규정, 정책, 업무 가이드에 대해서만 답변합니다.
        - Context에 있는 내용만을 기반으로 답변하며, 추측하거나 Context 외의 정보를 추가하지 않습니다.
        - 답변은 최대한 간결하고 명확하게, 핵심 정보만을 전달합니다.
        - 질문과 관련 없는 내용이거나 Context에 정보가 부족한 경우, '잘 모르겠습니다.'라고 답변합니다.
        - 일상적인 대화나 업무와 관련 없는 질문에 대해서는 완곡하게 거절합니다.
        - 대화의 흐름을 끊지 않으면서도 친절하게 업무 관련 질문만 답변할 수 있음을 안내합니다.
        - 예를 들어, 업무와 관련 없는 질문이 들어올 경우 다음과 같이 답변합니다: "죄송합니다. 저는 업무에 관련된 내용에만 답변할 수 있습니다.
        - 답변에 대한 감사 인사, 칭찬에 대해서는 '감사합니다!'라고 대답합니다. 부가적인 답변을 생성하지 마십시오.
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

        response = requests.post(
            self.host + "/testapp/v1/chat-completions/HCX-003",
            headers=headers,
            json=request_data,
            stream=True
        )

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
    api_key='NTA0MjU2MWZlZTcxNDJiY6+5UNhJXWh3gqmFLbiMpde7ehpEJAFPwFUIey9lGc0S',
    api_key_primary_val='VhkfehtF14qpXmZPIA6VRw6x1c1eDCXp3P6BfbrG',
    request_id='b9288b57-8e12-45cc-b378-49cc13d8dbb6'
)

def extract_text_from_pdf(pdf_path, start_page=None, end_page=None):
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

    return text

def retrieve_docs(text, model_index=0):
    model_list = [
        'bespin-global/klue-sroberta-base-continue-learning-by-mnr',
        'BAAI/bge-m3',
        'All-MiniLM-L6-v2',
        'sentence-transformers/paraphrase-MiniLM-L6-v2'
    ]

    if 'embeddings' not in st.session_state.vectorstore_cache:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_list[model_index],
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True},
        )
        st.session_state.vectorstore_cache['embeddings'] = embeddings
    else:
        embeddings = st.session_state.vectorstore_cache['embeddings']

    vectorstore_path = 'vectorstore_' + str(model_index)
    os.makedirs(vectorstore_path, exist_ok=True)

    if vectorstore_path not in st.session_state.vectorstore_cache:
        if os.path.exists(os.path.join(vectorstore_path, "index")):
            vectorstore = Chroma(persist_directory=vectorstore_path, embedding_function=embeddings)
        else:
            docs = [Document(page_content=text)]
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
            splits = text_splitter.split_documents(docs)
            vectorstore = Chroma.from_documents(splits, embeddings, persist_directory=vectorstore_path)
            vectorstore.persist()

        st.session_state.vectorstore_cache[vectorstore_path] = vectorstore
    else:
        vectorstore = st.session_state.vectorstore_cache[vectorstore_path]

    return vectorstore.as_retriever()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def rag_chain(question):
    pdf_path = "AK아이에스 복리후생 제도 매뉴얼.pdf"

    if st.session_state.loading_text is None:
        loading_text = extract_text_from_pdf(pdf_path)
        st.session_state.loading_text = loading_text
    else:
        loading_text = st.session_state.loading_text

    if st.session_state.retriever is None:
        retriever = retrieve_docs(loading_text)
        st.session_state.retriever = retriever
    else:
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
    result = re.sub("(Answer:|답변:|답변|:|Answer)", "", result)
    return result

set_llm_cache(InMemoryCache())
st.title("AK아이에스 복리후생제도 알리미")
st.markdown("현재 챗봇은 **테스트용**으로 운영되고 있습니다. 사용에 참고 바랍니다.")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "무엇을 도와 드릴까요?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    if not prompt.strip():
        st.warning("질문을 입력해주세요.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        msg = generate_response(prompt)
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("assistant").write(msg)
