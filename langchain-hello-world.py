from langchain_huggingface.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
import torch

from langchain_huggingface.llms import HuggingFacePipeline
# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("translation_en_to_fr", model="facebook/nllb-200-distilled-600M", torch_dtype=torch.bfloat16, src_lang='eng_Latn', tgt_lang='fra_Latn')
hf = HuggingFacePipeline(pipeline=pipe)


template = """Translate the following English to French: {text}"""
prompt = PromptTemplate.from_template(template)

chain = prompt | hf



print(chain.invoke({"text": 'How are you doing today?'}))