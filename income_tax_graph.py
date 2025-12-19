# %%
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model='text-embedding-3-large')

vector_store = Chroma(
    embedding_function = embeddings,
    collection_name = 'income_tax_collection2',
    persist_directory = './income_tax_collection2'
)

retriever = vector_store.as_retriever(search_kwargs={'k':3})

# %%
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from langgraph.graph import StateGraph
    
class AgentState(TypedDict):
    query: str
    context: List[Document]
    answer: str

# %%
def retrieve(state: AgentState):
    query = state['query']
    docs = retriever.invoke(query)
    
    return {'context': docs}

# %%
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model='gpt-4o')

# %%
from langchain import hub

generate_prompt = hub.pull("rlm/rag-prompt")
generate_llm = ChatOpenAI(model='gpt-4o', max_completion_tokens=100) # LLM이 최대 100 토큰까지만 답변을 생성하도록 함.

def generate(state: AgentState):
    query = state['query']
    context = state['context']
    rag_chain = generate_prompt | generate_llm
    
    response = rag_chain.invoke({'question':query, 'context':context})
    
    return {'answer': response.content}



# %%
from typing import Literal

doc_relevance_prompt = hub.pull("langchain-ai/rag-document-relevance")

def check_doc_relevance(state: AgentState) -> Literal['relevant', 'irrelevant']:
    query = state['query']
    context = state['context']
    print(f'query: {query}')
    
    doc_relevance_chain = doc_relevance_prompt  | llm
    response = doc_relevance_chain.invoke({'question':query, 'documents':context})
    print(f'response doc relevance: {response}')
    
    if response['Score'] == 1:
        return 'relevant'
    return 'irrelevant'
    

# %%
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

dictionary = ["사람을 나타내는 표현 -> 거주자"]

rewrite_prompt = PromptTemplate.from_template(f"""
        우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
        사전: {dictionary}
        질문: {{question}}     
""")


def rewrite(state: AgentState):
    query = state['query']
    rewrite_chain = rewrite_prompt | llm | StrOutputParser()
    
    response = rewrite_chain.invoke({'question': query})
    return {'query': response}

# %%
from langchain import hub
from langchain_core.prompts import PromptTemplate

hallucination_prompt = PromptTemplate.from_template(""" 
    You are a teacher tasked with evaluating whether a student`s answer is based on documents or not,
    Given documents, which are excerpts from income tax law, and a student`s answer;
    If the stuendt`s answer is based on documents, respond with "not hallucinated",
    If the student`s answer is not based on documents, respond with "hallucinated".                                          
              
    documents: {documents}
    student_answer: {student_answer}                                      
 """)

hallucination_llm = ChatOpenAI(model='gpt-4o', temperature=0)


def check_hallucination(state: AgentState) -> Literal['hallucinated', 'not hallucinated']:
    answer = state['answer']
    context = state['context']
    context = [doc.page_content for doc in context]
    print(f'context: {context}')
    
    hallucination_chain = hallucination_prompt | hallucination_llm | StrOutputParser()
    response = hallucination_chain.invoke({'student_answer': answer, 'documents': context})
    print(f'hallucination response: {response}')
    
    return response

# %%
from langchain import hub

helpfullness_prompt = hub.pull("langchain-ai/rag-answer-helpfulness")

def check_helpfulness_grader(state: AgentState):
    query = state['query']
    answer = state['answer']

    helpfulness_chain = helpfullness_prompt | llm
    response = helpfulness_chain.invoke({'question': query, 'student_answer': answer})
    print(f'helpfulness response: {response}')
    
    if response['Score'] == 1:
        return 'helpful'
    return 'unhelpful'  


def check_helpfulness(state: AgentState):
    return state

# %%
query = '연봉 5천만원인 거주자의 소득세는 얼마인가요?'
context = retriever.invoke(query)

print('-----document------')
for document in context:
    print(document.page_content)
print('-----document------')

generate_state = {'query': query, 'context': context}
answer = generate(generate_state)
print(f'answer:{answer}')

hallucination_state = {'answer': answer, 'context': context}
helpfulness_state = {'query': query, 'answer': answer}

check_hallucination(hallucination_state)
check_helpfulness(helpfulness_state)

# %%
graph_builder = StateGraph(AgentState)

# %%
graph_builder.add_node('retrieve', retrieve)
graph_builder.add_node('generate', generate)
graph_builder.add_node('rewrite', rewrite)
graph_builder.add_node('check_helpfulness', check_helpfulness)

# %%
from langgraph.graph import START, END

graph_builder.add_edge(START, 'retrieve')

graph_builder.add_conditional_edges(
    'retrieve',
    check_doc_relevance,
    {
        'relevant': 'generate',
        'irrelevant': END
    }
)

graph_builder.add_conditional_edges(
        'generate',
        check_hallucination,
        {
            'not hallucinated': 'check_helpfulness',
            'hallucinated': 'generate'
        }
)

graph_builder.add_conditional_edges(
    'check_helpfulness',
    check_helpfulness_grader,
    {
        'helpful': END,
        'unhelpful': 'rewrite'
    }
)

graph_builder.add_edge('rewrite', 'retrieve')

# %%
graph = graph_builder.compile()
