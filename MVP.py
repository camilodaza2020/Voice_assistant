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