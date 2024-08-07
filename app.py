import streamlit as st
from assistant_graph import create_graph_stream



def main():
    st.set_page_config(page_title="Multi-Agent-Assistant",
                       page_icon="🤖",
                       layout="centered",
                       initial_sidebar_state="expanded",
                       menu_items=None)
    # sidebar
    with st.sidebar:
        st.subheader("Api Key")
        tavily_api_key = st.text_input("Tavily API Key", key="file_qa_api_key", type="password")
        st.subheader("")
        st.subheader("")
        st.subheader("Chat options ")
        clear_chat = st.button("Clear chat", type="primary")
        st.subheader("")
        st.subheader("")
        st.subheader("Links")
        "[LangGraph Tutorial](https://langchain-ai.github.io/langgraph/tutorials/)"
        "[![Open in GitHub](https://github.com/codespaces/badge.svg)](https://github.com/saurav-dhait/Multi-Agent-Assistant)"
    # main body
    st.title("🤖 Multi-Agent-Assistant : ")
    if clear_chat:
        st.session_state["messages"] = [{"role": "assistant", "content": "Hey, how can i help you ? "}]
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "Hey, how can i help you ? "}]
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        response = "No response"
        with st.spinner(f"Generating response"):
            flag = 0
            for s in create_graph_stream(prompt):
                a = s.popitem()
                response = a[1]["messages"][-1]
                print(response)
                if response.name == "Normal":

                    st.session_state.messages.append({"role": "assistant", "content": response.content})
                    st.chat_message("assistant").write(response.content)
                    flag = 1
                else:
                    if flag:
                        st.session_state.messages.append({"role": "assistant", "content": "Asking other assistants for help...."})
                        st.chat_message("assistant").write("Asking other assistants for help....")
                        flag = 0


if __name__ == '__main__':
    main()
