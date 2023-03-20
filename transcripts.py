#Important before we start, install this command in terminal

#pip install whisper
#pip install git+https://github.com/openai/whisper.git



import whisper

model = whisper.load_model("base")
result = model.transcribe("here you should put the route of the video or presentation made in MP4 or any format")

with open("result.txt", "w") as f:
    print(result["text"], file=f)
