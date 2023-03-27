#Important before we start, install this command in terminal

#pip install whisper
#pip install git+https://github.com/openai/whisper.git
#pip install gradio


import gradio as gr
import whisper

# Load the model
model = whisper.load_model("base")

# Function for transcribing the input audio file
def transcribe(input_audio):
    result = model.transcribe(input_audio.name)
    return result["text"]

# Create Gradio interface
audio_input = gr.inputs.Audio(type="filepath", label="Upload an MP3 or record your voice")
text_output = gr.outputs.Textbox(label="Transcription")

iface = gr.Interface(
    fn=transcribe,
    inputs=audio_input,
    outputs=text_output,
    title="Whisper Audio Transcription",
    description="Upload an MP3 or record your voice, and the Whisper model will transcribe it.",
)

# Launch the interface
iface.launch()
