import streamlit as st
from test import get_ai_message

st.set_page_config(page_title="경북대학교 공지사항 챗봇")
st.title("CHAT BOT")
st.caption("궁금한 점에 대해 답변해드립니다!")

if 'message_list' not in st.session_state :
    st.session_state.message_list = []

# 화면에 이전까지의 채팅 내용을 기록함
for message in st.session_state.message_list :
    with st.chat_message(message["role"]) :
        st.write(message["content"])

# 새로운 채팅 내용을 리스트에 추가시킴
if user_question := st.chat_input(placeholder="궁금한 내용들을 말씀해주세요.") :
    with st.chat_message("user") :
        st.write(user_question)
    st.session_state.message_list.append({"role": "user", "content": user_question})

    with st.spinner("답변을 생성하는 중입니다.") :
        ai_message = get_ai_message(user_question)
        with st.chat_message("ai") :
            st.write(ai_message)
        st.session_state.message_list.append({"role": "ai", "content": ai_message})
