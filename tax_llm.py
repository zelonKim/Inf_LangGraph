from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic import hub
from langchain_classic.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


def get_llm(model='gpt-4o'):
    llm = ChatOpenAI(model=model)
    return llm


def get_retriever():
    embedding = OpenAIEmbeddings(model='text-embedding-3-large')
    index_name='tax-index'
    
    database = PineconeVectorStore.from_existing_index(
        index_name=index_name, # 해당 인덱스명과 일치하는 기존 데이터베이스를 가져옴.
        embedding=embedding # 사용자의 질문에 해당 임베딩을 적용하여 벡터로 변환함.
    )
    
    retriever = database.as_retriever(search_kwargs={"k": 3})
    
    return retriever



def get_dictionary_chain():
    dictionary = ["사람을 나타내는 표현 -> 거주자"]
    
    prompt = ChatPromptTemplate.from_template(f"""
            우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
            만약, 변경하지 않아도 될것 같다면, 그대로 사용자의 질문을 사용해주세요.
            사전: {dictionary}
            
            질문: {{question}}     
    """)
    dictionary_chain = prompt | get_llm() | StrOutputParser()
    
    return dictionary_chain



def get_qa_chain():
    prompt = hub.pull("rlm/rag-prompt")
    
    qa_chain = RetrievalQA.from_chain_type( 
        retriever = get_retriever(), 
        chain_type_kwargs = {"prompt": prompt}, 
        llm = get_llm()
    )
    return qa_chain




def get_ai_message(user_message):
    tax_chain = {"query": get_dictionary_chain()} | get_qa_chain()
    
    ai_message = tax_chain.invoke({"question": user_message})
    
    return ai_message['result']
