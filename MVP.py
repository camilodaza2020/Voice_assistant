# !pip install gradio
# !pip install transformers

#Please install this libraries in envs
import gradio as gr
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

#here we act the model bert from opensourcehugging face
tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

#here we create a funtion that convert the text in embbedings and tokens
def answer_question(context, question):
    inputs = tokenizer(question, context, return_tensors="pt")
    outputs = model(**inputs)
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs.input_ids[0][answer_start:answer_end]))
    return answer

#we crete the interface with gradio
context_input = gr.inputs.Textbox(lines=10, placeholder="Put long text or some frases here....")
question_input = gr.inputs.Textbox(lines=2, placeholder="Write the question in this space")

answer_output = gr.outputs.Textbox(label="Result")

iface = gr.Interface(fn=answer_question, inputs=[context_input, question_input], outputs=answer_output, title="Preguntas y Respuestas", description="Responde preguntas basadas en el texto proporcionado.")


#launch the model
iface.launch(share=True, inbrowser=True)

#if you see a link, go and text the model

#_____________________________________________________________________________________________________________________

#Now this code use openai and Lancgchain the model is different

# pip install langchain
# pip install chromadb
# pip install python-magic-bin
# pip install unstructured
# pip install nltk

import numpy as np
import os
import nltk
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import DirectoryLoader
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='root')


os.environ['OPENAI_API_KEY'] = 'Here you should put the KEY of OpenAI'

nltk.download('averaged_perceptron_tagger')

# Your document should be uploaded to Google Colab. You can change this to the path of your uploaded document.
document_path = 'Here you should put the file with information, route'


loader = DirectoryLoader(os.path.dirname(document_path), glob='**/*.txt')
documents = loader.load()
documents = [doc.decode('utf-8') for doc in documents]

#Here we divided the total text
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

#We use Openai for convert in embeddings
embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])

#Thisv ariable gets the embeddings and save in store of Langchain
docsearch = Chroma.from_documents(texts, embeddings)
qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=docsearch)
query = "who is jared "
answer = qa.run(query)
print(answer)

print(docsearch)
