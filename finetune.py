

import openai
import os

# Set your OpenAI API key
openai.api_key = "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"

# Load your training data
with open(os.path.join(os.getcwd(), "C:\\Users\\USUARIO\\Desktop\\ChatOpenAI\\envs\\chat\\datos.txt"), "r") as f:
    training_data = f.read()

# Fine-tune the GPT-3 model on your training data
model_engine = "text-davinci-002"
response = openai.Completion.create(
    engine=model_engine,
    prompt=training_data,
    temperature=0.5,
    max_tokens=2048,
    n=1,
    stop=None
)

# Save the finetuned model
with open(os.path.join(os.getcwd(), "finetuned_model.txt"), "w") as f:
    f.write(response.choices[0].text.strip())
