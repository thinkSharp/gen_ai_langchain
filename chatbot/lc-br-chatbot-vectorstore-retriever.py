from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_aws import BedrockLLM, BedrockEmbeddings
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

model_name = "amazon.titan-text-express-v1"
model_embadding = 'amazon.titan-embed-text-v1'


br_embadding = BedrockEmbeddings(credentials_profile_name='default', model_id=model_embadding)

vectorstore = Chroma.from_documents(documents=documents, embedding=br_embadding)

print(vectorstore.similarity_search('asd', k=1))

embedding = br_embadding.embed_query('cat')
#print(vectorstore.similarity_search_by_vector(embedding=embedding))
#print(vectorstore.similarity_search_by_vector_with_relevance_scores(embedding=embedding))

retriever = vectorstore.as_retriever(search_kwargs={'k': 1})

model_kwargs = {
    'maxTokenCount': 512,
    'stopSequences': [],
    'temperature':0

}

llm = BedrockLLM(credentials_profile_name='default', 
                 model_id=model_name,
                 model_kwargs=model_kwargs)


message = """
Answer this question using the provided context only. If don't know say I don't have it.

{question}

Context: {context}
"""

prompt = ChatPromptTemplate.from_messages(
    [ ("human", message) ]
    )

rag_chain = {'context': retriever, 'question': RunnablePassthrough()} | prompt | llm

response = rag_chain.invoke('tell me about asdf')

print(response)

