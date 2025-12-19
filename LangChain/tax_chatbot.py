import streamlit as st
from dotenv import load_dotenv
from tax_llm import get_ai_response

load_dotenv()

st.set_page_config(page_title="소득세 챗봇", page_icon="🤖") # 브라우저 탭의 제목과 아이콘을 설정함.
st.title("💵 소득세 챗봇") # 메인 제목을 화면에 출력함.
st.caption("소득세에 관련된 모든 것을 답변해드립니다.") # 부제를 화면에 출력함.


if 'message_list' not in st.session_state: 
    st.session_state.message_list = [] # 저장 공간을 초기화함.


for message in st.session_state.message_list: # 저장 공간에 있는 이전 대화 기록들을 가져옴.
    with st.chat_message(message["role"]): # 해당 역할에 따른 말풍선 블록을 화면에 출력함.
        st.write(message["content"]) # 해당 내용을 화면에 출력함.


if user_question := st.chat_input(placeholder="소득세에 관련된 궁금한 내용을 질문해주세요."): # 채팅 입력창을 생성함. # 사용자가 입력한 값을 해당 변수에 저장함.
# := 연산자를 통해 값을 변수에 할당하면서 동시에 조건문을 처리함.
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.message_list.append({"role":"user", "content":user_question}) # 저장 공간에 사용자의 질문을 기록함.
    
    
    
    
    with st.spinner("답변을 생성하는 중입니다..."): # 답변을 생성하는 동안 로딩 스피너를 화면에 출력함.
        ai_response = get_ai_response(user_question)
        
        
        
        
        with st.chat_message("ai"):
            ai_message = st.write_stream(ai_response) # 스트리밍 데이터를 실시간으로 화면에 출력함.
        st.session_state.message_list.append({"role":"ai", "content":ai_message})
        
    
    
        
    