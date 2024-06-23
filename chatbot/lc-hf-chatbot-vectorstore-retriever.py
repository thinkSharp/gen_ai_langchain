from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Goldfish are popular pets for beginners, requiring relatively simple care.",
        metadata={"source": "fish-pets-doc"},
    ),
    Document(
        page_content="Parrots are intelligent birds capable of mimicking human speech.",
        metadata={"source": "bird-pets-doc"},
    ),
    Document(
        page_content="Rabbits are social animals that need plenty of space to hop around.",
        metadata={"source": "mammal-pets-doc"},
    ),
]

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

vectorstore = Chroma.from_documents(documents=documents, embedding=hf)

#print(vectorstore.similarity_search('cat', k=1))

embedding = hf.embed_query('cat')
#print(vectorstore.similarity_search_by_vector(embedding=embedding))
#print(vectorstore.similarity_search_by_vector_with_relevance_scores(embedding=embedding))

retriever = vectorstore.as_retriever(search_kwargs={'k': 1})


model_name = 'google/flan-t5-base'
task = 'text2text-generation'

llm = HuggingFacePipeline.from_model_id(
    model_id=model_name,
    task=task,
    pipeline_kwargs=dict(
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.04
    )
)


message = """
Answer this question using the provided context only.

{question}

Context: {context}
"""

prompt = ChatPromptTemplate.from_messages(
    [ ("human", message) ]
    )

rag_chain = {'context': retriever, 'question': RunnablePassthrough()} | prompt | llm

response = rag_chain.invoke('tell me about dogs')

print(response)

