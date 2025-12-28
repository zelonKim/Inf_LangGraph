import os
import uuid
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
agent_api_key = os.getenv("AGENT_API_KEY")
agent_api_url = os.getenv("AGENT_API_URL")


st.title('이메일 에이전트')

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())



for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.write(msg['content'])

prompt = st.chat_input('이메일 내용을 입력해주세요')

if prompt:
    st.chat_message('user').write(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    response = requests.post(
        agent_api_url, 
        headers={'Authorization': agent_api_key}, 
        json={'message': prompt,
              'session_id': st.session_state.session_id
              }
        )
    
    st.info(response.json())
    
    st.session_state.messages.append({'role': 'assistant', 'content': response.json()['output']})
    with st.chat_message('assistant'):
        st.write(response.json()['output'])


    