import streamlit as st
from langchain_core.messages import SystemMessage,HumanMessage,BaseMessage,AIMessage,ToolMessage
import uuid
from chatbot_backend import (
    chatbot,
    ingest_pdf,
    b,
    thread_document_metadata,
)


#  ************************************************************BUTTON UI************************************************************




#*****************************************************Utility function*************************************************************

def generate_thread_id():
    thread_id=uuid.uuid4()
    return thread_id

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
         st.session_state['chat_threads'].append(thread_id)
   


def new_chat():
    thread_id=generate_thread_id()
    add_thread(thread_id)
    st.session_state['thread_id']=thread_id
    st.session_state['message_history']=[]

def load_conversation(thread_id):
    state=chatbot.get_state({'configurable':{'thread_id':thread_id}})
    return state.values.get('messages',[])





# ***********************************************************Session setup*****************************************************
if 'message_history' not in st.session_state:
    st.session_state['message_history']=[]

if 'thread_id' not in st.session_state:
    st.session_state['thread_id']=None

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads']=b

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

if "has_user_message" not in st.session_state:
    st.session_state["has_user_message"] = False


if st.session_state['thread_id'] is not None and st.session_state['thread_id'] != "None":
    add_thread(st.session_state['thread_id'])

thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})
threads = st.session_state["chat_threads"][::-1]









# *****************************************************Sidebar UI*********************************************************************
st.sidebar.title("Chatbot")
if st.sidebar.button("New Chat", use_container_width=True): 
    new_chat()
    

if thread_docs:
    latest_doc = list(thread_docs.values())[-1]
    st.sidebar.success(
        f"Using `{latest_doc.get('filename')}` "
        f"({latest_doc.get('chunks')} chunks from {latest_doc.get('documents')} pages)"
    )
else:
    st.sidebar.info("No PDF indexed yet.")

uploaded_pdf = st.sidebar.file_uploader("Upload a PDF for this chat", type=["pdf"])
if uploaded_pdf:
    if uploaded_pdf.name in thread_docs:
        st.sidebar.info(f"`{uploaded_pdf.name}` already processed for this chat.")
    else:
        with st.sidebar.status("Indexing PDF…", expanded=True) as status_box:
            summary = ingest_pdf(
                uploaded_pdf.getvalue(),
                thread_id=thread_key,
                filename=uploaded_pdf.name,
            )
            thread_docs[uploaded_pdf.name] = summary
            status_box.update(label="✅ PDF indexed", state="complete", expanded=False)
   



st.sidebar.subheader('Past conversation')
if not threads:
    st.sidebar.write("No past conversations yet.")
else:
    for thread_id in st.session_state['chat_threads'][::-1]:
        button_label="new_chat"
        messages=load_conversation(thread_id)

        for mss in messages:
            if isinstance(mss,HumanMessage): 
                button_label =mss.content[:30]+'...'
                break

        if st.sidebar.button(button_label,str(thread_id)):
            st.session_state["thread_id"]=thread_id
      
           
            st.session_state['message_history']=[]
            for mss in messages:
                if isinstance(mss,HumanMessage):
                    role='user'
              
                else:
                    role='assistant'

                st.session_state['message_history'].append({'role':role,'content':mss.content})
            st.session_state["ingested_docs"].setdefault(str(thread_id), {})
                


       





# ***************************************************************UI*****************************************************************
# loading the whole chat_history

st.title("Multi Utility Chatbot")
for messages in st.session_state['message_history']:
   with st.chat_message(messages['role']):
       st.text(messages['content'])



user_input=st.chat_input("Ask about your document or use tools")
if user_input:
    if st.session_state['thread_id'] is None:
        new_chat() 
    st.session_state["has_user_message"] = True
    st.session_state['message_history'].append({'role':'user','content':user_input})
    with st.chat_message('user'):
        st.text(user_input)   

    CONFIG = {
        "configurable": {"thread_id": thread_key},
        "metadata": {"thread_id": thread_key},
        "run_name": "chat_turn",
    } 

    with st.chat_message('assistant'):
        status_holder = {"box": None}

        def ai_only_stream():
           for message_token,metta_data in chatbot.stream({'messages':[HumanMessage(content=user_input)]},
                          config=CONFIG,stream_mode='messages'):
               
                # Lazily create & update the SAME status container when any tool runs
                if isinstance(message_token, ToolMessage):
                    tool_name = getattr(message_token, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )
               


                if isinstance(message_token,AIMessage):
                   yield message_token.content

        ai_response= st.write_stream(ai_only_stream())

    # Finalize only if a tool was actually used
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )
           
    
 
    st.session_state['message_history'].append({'role':'assistant','content': ai_response})

    doc_meta = thread_document_metadata(thread_key)
    if doc_meta:
        st.caption(
            f"Document indexed: {doc_meta.get('filename')} "
            f"(chunks: {doc_meta.get('chunks')}, pages: {doc_meta.get('documents')})"
        )
st.divider()