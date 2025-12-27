from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Demo")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

if __name__ == "__main__":
    mcp.run(transport='stdio')


#########################


# 필요한 라이브러리들을 임포트
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
from langchain_core.tools import tool
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
import os 
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from mcp.server.fastmcp.prompts import base


load_dotenv()

llm = ChatOpenAI(model='gpt-4o')
small_llm = ChatOpenAI(model='gpt-4o-mini')

embeddings = OpenAIEmbeddings(model='text-embedding-3-large')

vector_store = Chroma(
    embedding_function=embeddings,
    collection_name = 'real_estate_tax',
    persist_directory='./real_estate_tax_collection'
)

retriever = vector_store.as_retriever(search_kwargs={'k': 3})

rag_prompt = hub.pull('rlm/rag-prompt')

# 문서 포맷팅을 위한 헬퍼 함수
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 세금 공제액 정보를 검색하기 위한 체인 구성
tax_deductible_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | small_llm
    | StrOutputParser()
)

# 공제액 관련 기본 질문 정의
deductible_question = f'주택에 대한 종합부동산세 과세표준의 공제액을 알려주세요'

# 사용자별 공제액 계산을 위한 프롬프트 템플릿
user_deduction_prompt = """아래 [Context]는 주택에 대한 종합부동산세의 공제액에 관한 내용입니다. 
사용자의 질문을 통해서 가지고 있는 주택수에 대한 공제액이 얼마인지 금액만 반환해주세요

[Context]
{tax_deductible_response}

[Question]
질문: {question}
답변: 
"""

# 프롬프트 템플릿 객체 생성
user_deduction_prompt_template = PromptTemplate(
    template=user_deduction_prompt,
    input_variables=['tax_deductible_response', 'question']
)

# 사용자별 공제액 계산을 위한 체인 구성
user_deduction_chain = (user_deduction_prompt_template
    | small_llm
    | StrOutputParser()
)


#########################


# 세금 공제액 계산을 위한 커스텀 도구
@mcp.tool(
    name='tax_deductible_tool',
    description= """사용자의 부동산 소유 현황에 대한 질문을 기반으로 세금 공제액을 계산합니다.
    
    이 도구는 다음 두 단계로 작동합니다:
    1. tax_deductible_chain을 사용하여 일반적인 세금 공제 규칙을 검색
    2. user_deduction_chain을 사용하여 사용자의 특정 상황에 규칙을 적용

    Args:
        question (str): 부동산 소유에 대한 사용자의 질문
        
    Returns:
        str: 세금 공제액 (예: '9억원', '12억원')
    """
)
def tax_deductible_tool(question: str) -> str:
    # 일반적인 세금 공제 규칙 검색
    tax_deductible_response = tax_deductible_chain.invoke(deductible_question)
    
    # 사용자의 특정 상황에 규칙 적용
    tax_deductible = user_deduction_chain.invoke({
        'tax_deductible_response': tax_deductible_response, 
        'question': question
    })
    
    return tax_deductible


#########################


@mcp.prompt(
    name="house_tax_system_prompt",
    description="""종합부동산세 계산 프롬프트"""
)
def house_tax_system_prompt():
    system_message_content = """당신의 역할은 주택에 대한 종합부동산세를 계산하는 것입니다. 
    사용자의 질문이 들어오면, 사용자의 질문을 바탕으로 종합부동산세를 계산해주세요.
    종합부동산세를 계산하기 위해서는 과세표준을 어떻게 계산할지 파악해야하고, 
    사용자에 질문에 따른 공제액을 파악해야 하고, 
    사용자에 질문에 따른 공정시장가액비율을 파악해야 합니다.
    이 세가지를 파악하고 나면, 종합부동산세를 계산해주세요.
    """
    return base.UserMessage(content=system_message_content)



