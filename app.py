import streamlit as st
from model import ChatBotUS
import time


st.title("ChatBot")

import streamlit as st

with st.sidebar:
    st.title("Chọn nguồn")
    
    # Sử dụng radio để chọn giữa các nguồn
    source = st.radio("Chọn nguồn", ("Sổ tay sinh viên", "URL khác"))
    
    if source == "Sổ tay sinh viên":
        model = ChatBotUS()  # Gọi ChatBotUS với nguồn mặc định
    elif source == "URL khác":
        user_input = st.text_input("Enter URL:")
        if len(user_input) != 0:
            model = ChatBotUS(url=user_input)  # Gọi ChatBotUS với URL người dùng nhập

        
with st.chat_message('assistant'):    
    st.markdown('Tôi có thể giúp gì cho bạn')

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"],)

if prompt := st.chat_input("What is up?"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    response = model.make_response(question = prompt)
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})