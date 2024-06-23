import os
import bs4
import gradio as gr
from dotenv import load_dotenv
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

load_dotenv()


os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')


rag_prompt = hub.pull('rlm/rag-prompt')




embedding_model = 'sentence-transformers/all-MiniLM-L6-v2'
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
hf_embedding = HuggingFaceEmbeddings(
    model_name=embedding_model,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)


model_name = 'HuggingFaceH4/zephyr-7b-beta'
task = 'text-generation'

llm = HuggingFacePipeline.from_model_id(
    model_id=model_name,
    task=task,
    pipeline_kwargs=dict(
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
    )
)


rag_chain = None
chat_url = 'No URL Provided!'

def load_url(url):
    global rag_chain
    global chat_url
    loader = WebBaseLoader(web_paths=(url,),
                       bs_kwargs=dict(
                           parse_only=bs4.SoupStrainer(
                               class_=('post-content','post-title','post-header')
                           )
                       ), )
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(documents=splits, embedding=hf_embedding)
    
    retriever = vectorstore.as_retriever()
    
    rag_chain = ({'context': retriever | format_docs, 'question': RunnablePassthrough()}
             | rag_prompt
             | llm
             | StrOutputParser()
             )
    
    chat_url = f'Chat with URL: {url}'
    return 'URL Data loaded in Local Store! Click chat tab to start chatt with the doc!'


def chat_about_url(user_input, history):
    global rag_chain
    if not rag_chain:
        return "Please enter a valid URL first.", history

    response = rag_chain.invoke(user_input)
    
    history.append((user_input, response))
    return "", history

with gr.Blocks() as demo:
    header = gr.Label(label="Enter a URL to chat with the content!")
    choice = gr.Textbox(label='Enter URL')
    output2 = gr.Textbox(label="URL Content Processing Respond")
    submit = gr.Button("Submit")
    submit.click(fn=load_url, inputs=choice, outputs=output2)

    chatbot = gr.Chatbot(height=400)
    msg = gr.Textbox(label="Please ask you question!")
    clear = gr.ClearButton([msg, chatbot])



    msg.submit(chat_about_url, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    demo.launch()