from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from tax_fewshot import answer_examples


def get_llm(model='gpt-4o'):
    llm = ChatOpenAI(model=model)
    return llm


def get_dictionary_chain():
    llm = get_llm()
    dictionary = ["사람을 나타내는 표현 -> 거주자"]
    prompt = ChatPromptTemplate.from_template(f"""
            우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
            만약, 변경하지 않아도 될것 같다면, 그대로 사용자의 질문을 사용해주세요.
            사전: {dictionary}
            
            질문: {{question}}     
    """)
    dictionary_chain = prompt | llm | StrOutputParser()
    return dictionary_chain



def get_retriever():
    embedding = OpenAIEmbeddings(model='text-embedding-3-large')
    index_name='tax-index'
    database = PineconeVectorStore.from_existing_index(
        index_name=index_name, # 해당 인덱스명과 일치하는 기존 데이터베이스를 가져옴.
        embedding=embedding # 사용자의 질문에 해당 임베딩을 적용하여 벡터로 변환함.
    )
    retriever = database.as_retriever(search_kwargs={"k": 3})
    return retriever




###############################




store = {} # 세션별 대화기록 저장소 

def get_session_history(session_id: str) -> BaseChatMessageHistory: # 세션ID에 맞는 대화 기록을 가져오거나, 생성함.
    if session_id not in store:
        store[session_id] = ChatMessageHistory() # 키는 session_id, 밸류는 ChatMessageHistory()객체임.
    return store[session_id]



def get_history_retriever():
    llm = get_llm()
    retriever = get_retriever()
    
    contextualize_q_system_prompt = ( # 대화 기록을 참고하여 사용자의 질문을 재구성하도록 함.
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"), # "히스토리 메시지 키"를 통해 이전 대화기록 전체를 프롬프트에 삽입함.
            ("human", "{input}"), 
        ]
    )
    
    history_aware_retriever = create_history_aware_retriever( # 이전 대화 기록을 참고하는 리트리버를 생성함.
        llm, retriever, contextualize_q_prompt
    )
    
    return history_aware_retriever





def get_rag_chain():    
    llm = get_llm()
    history_aware_retriever = get_history_retriever()

    system_prompt = (
        "당신은 소득세법 전문가입니다. 사용자의 소득세법에 관한 질문에 답변해주세요"
        "아래에 제공된 문서를 활용해서 답변해주시고"
        "답변을 알 수 없다면 모른다고 답변해주세요"
        "답변을 제공할 때는 소득세법 (XX조)에 따르면 이라고 시작하면서 답변해주시고"
        "2-3 문장 정도의 짧은 내용의 답변을 원합니다"
        "\n\n"
        "{context}" 
    ) 
    
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{answer}"),
        ]
    )
    
    few_shot_prompt = FewShotChatMessagePromptTemplate( # LLM에게 답변 예시를 제공해줌.
        example_prompt = example_prompt,
        examples = answer_examples,
    )
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            few_shot_prompt,
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt) # 유사도 검색 문서를 프롬프트 내의 {context}부분에 채워줌."

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain) # 유사도 검색을 위한 리트리버를 해당 체인에 전달함.
    
    conversational_rag_chain = RunnableWithMessageHistory( # 기존 체인이 이전 대화기록을 활용하도록 함.
        rag_chain, # 기존 체인
        get_session_history, # 세션별 대화 저장소
        input_messages_key="input", # 사용자 질문 키
        history_messages_key="chat_history", # 히스토리 메시지 키
        output_messages_key="answer", # AI 답변 저장 키 
    ).pick("answer") # 최종 출력에서 AI답변(answer)만 반환함.
    
    return conversational_rag_chain
    




def get_ai_response(user_message):
    dictionary_chain = get_dictionary_chain()
    rag_chain = get_rag_chain()
    tax_chain = {"input": dictionary_chain} | rag_chain
    
    ai_response = tax_chain.stream( # 체인의 실행 결과를 스트리밍 형식으로 반환함.
        {
            "question": user_message
        },
        config={
            "configurable": {"session_id": "abc123"} # 대화기록을 저장하기 위한 세션 식별자를 설정함.
        }
    )
    
    return ai_response
