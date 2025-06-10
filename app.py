from langchain_core.messages import AIMessage
import streamlit as st
import os
import sys
from langgraph.checkpoint.memory import MemorySaver
from main import part_1_graph, _print_event, use_groq, stt
import uuid
from system_prompt import system_prompt_summarise_messages
from supabase import create_client
import random


os.environ['GROQ_API_KEY'] = "PLACE YOUR GROQ API KEY HERE"
st.set_page_config(page_title="Country Delight Agent Chat", page_icon="🧑‍💼")
st.title("🧑‍💼 Country Delight Agent Chatbot")

# --- Basic Authentication ---
ALLOWED_NUMBERS = {
    "3641697736",
    "08543578454",
    "05032807606",
    "08622211887",
    "06975813517",
    "3757574986",
    "05053443481",
    "00792862287",
    "02846133617",
    "2714253368"
}

def authenticate():
    st.session_state.authenticated = False
    phone = st.text_input("Enter your registered phone number to continue:", max_chars=15)
    if st.button("Login"):
        if phone.strip() in ALLOWED_NUMBERS:
            st.session_state.authenticated = True
            st.success("Authentication successful! Welcome.")
        else:
            st.error("Invalid phone number. Please try again.")
    return st.session_state.get("authenticated", False)

if "authenticated" not in st.session_state or not st.session_state["authenticated"]:
    if authenticate():
        st.rerun()
    st.stop()

# --- End Authentication ---

if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = ""
if "uploaded_images" not in st.session_state:
    st.session_state.uploaded_images = []

# Input box
user_input = st.chat_input("Ask something...", accept_file=True, file_type=["jpg", "jpeg", "png"])
audio_input = st.audio_input("Speak Here...")
thread_id = str(uuid.uuid4())
st.session_state.thread_id = thread_id
url = "https://bdcbnzufvyfkfqkhkvvy.supabase.co"  # Replace with your Supabase project URL
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJkY2JuenVmdnlma2Zxa2hrdnZ5Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDUzMDQ2MzQsImV4cCI6MjA2MDg4MDYzNH0.gqJ2AKeErJF9nLz7tGRrj3-1g53rYI-TZD1Sk0pvEP0"  # Replace with your Supabase anon key
supabase = create_client(url, key)


if user_input:
    # Save user message
    if len(st.session_state.messages) > 0 :
        st.session_state.memory = use_groq("meta-llama/llama-4-maverick-17b-128e-instruct",system_prompt_summarise_messages,f"content is :{st.session_state.messages}")
        print("\n------------------memory starts from here ------------------\n")
        print(st.session_state.memory)
        print("\n--------------memory ends here -----------------------")
    st.session_state.messages.append({"role": "user", "content": user_input})

    if user_input['files']:
        for i in range(len(user_input['files'])):
            image_file = user_input['files'][i]
            image_file = image_file.read()
            response = supabase.storage.from_("products-test-cd-chatbot").upload(
                    path= f"chat_bot_CD_product_bad_{random.randint(0,1000000)}.jpg",
                    file=image_file,
                file_options={"content-type" : "image/jpeg"}  # Set MIME type (e.g., "image/png")
            )
            image_response = response.full_path
            image_url = f"https://bdcbnzufvyfkfqkhkvvy.supabase.co/storage/v1/object/{image_response}"
            st.session_state.uploaded_images.append(image_url)


    #st.chat_message("user").write(user_input)

    config = {
        "configurable": {
            "user_id": "2",
            "thread_id": st.session_state.thread_id,
        }
    }

    try:
        events = part_1_graph.stream(
            {"messages": [
                {"role": "user", "content": f"last conversation summary: {st.session_state.memory}\n User message: {user_input} \ Product images: {st.session_state.uploaded_images}"}]
                },
            config=config,
            stream_mode="values"
        )
        _printed = set()

        for event in events:
            # DEBUG:
            _print_event(event, _printed)
            message = event.get("messages")
            if message:
                if isinstance(message, list):
                    message = message[-1]
                if isinstance(message, AIMessage):
                    if message.response_metadata['finish_reason'] == "stop":
                        stt(message.text())
                        st.audio("audio.mp3", autoplay=True)
                        st.session_state.messages.append({
                                "role": "assistant",
                                "content": message.text()
                                    })
                else:
                    pass

                #st.chat_message("assistant").write(message.text())

    except Exception as e:
        st.chat_message("assistant").write(f"Error: {e}")
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"Error: {e}"
        })

for msg in st.session_state.messages:
    role = msg.get("role") 
    content = msg.get("content")
    if role == "user":  
        st.chat_message("user").write(content.text)


st.caption("Built with Streamlit & LangGraph | Demo chatbot for Country Delight Agent")
