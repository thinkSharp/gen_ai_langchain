from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from transformers import pipeline
from utils.factory import factory


template_string = 'Translate the following English text to {ln}: {text}'
prompt_template = PromptTemplate.from_template(template=template_string)

language_codes = {
    'English': 'en',
    'French': 'fr',
    'German': 'de',
    'Spanish': 'es',
    'Italian': 'it',
    'Dutch': 'nl',
    'Portuguese': 'pt',
    'Russian': 'ru',
    'Chinese': 'zh',
    'Japanese': 'ja',
    'Korean': 'ko',
    'Arabic': 'ar'
}

model_id = 'google-t5/t5-small'
model_path = '../.cache/huggingface/hub/models--google-t5--t5-small/snapshots/df1b051c49625cf57a3d0d8d3863ed4d13564fe4'
class HuggingFaceLLM:
    def __init__(self):
        self._translator = {}


    def get_translator(self,language):
        translate = f'translation_en_to_{language}'

        translator = self._translator.get(language)
        if not translator:
            self._translator[language] = pipeline(translate, model=model_id)

hf_llm = HuggingFaceLLM()



def language_translator(target, text):
    translator = hf_llm.get_translator(target)
    llm_chain = prompt_template | translator
    translated_text = llm_chain.run({'ln':target, 'text': text})
    return translated_text

print(language_translator('fr','How are you doing?'))