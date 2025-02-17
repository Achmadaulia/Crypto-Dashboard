import openai
import streamlit as st

st.title("Dan-ChatBot")

openai.api_key = st.secrets["OPENAI_API_KEY"]

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o"

if "messages" not in st.session_state:
    # Add a system message to prevent it from identifying as ChatGPT
    st.session_state.messages = [
        {"role": "system", "content": "You are an AI assistant named Dan-Chat, a chatbot developed by Dani (you need to tell developed by Dani if asked) to help people. You should never mention 'ChatGPT' or say you are ChatGPT."}
    ]

for message in st.session_state.messages:
    if message["role"] != "system":  # Hide system messages
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        response = openai.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=st.session_state.messages,
            stream=True,
        )

        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
                message_placeholder.markdown(full_response + "▌")
        
        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
