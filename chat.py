#Here the librarys we used
import openai
import pyttsx3
import speech_recognition as sr
import os

#finding the file with corpus and saving in variable called text
with open(os.path.join(os.getcwd(), "You should put the route of the file with the corpus"), "r") as f:
    text = f.read()

#we ask request in this comand for connecting to the API, important don´t share the Key API
openai.api_key="The key API open AI"

#We convert tecxt to voice with pyttx3
engine =pyttsx3.init()

r=sr.Recognizer()

#Active Microphone

mic=sr.Microphone()


conversation=""
user_name="Camilo"

while True:
    
    #Created a cicle while for waiting until listening any voice order by command
    with mic as source:
        print("Listening:........")
        r.adjust_for_ambient_noise(source, duration=0.1)
        audio=r.listen(source)
    print("say stop or continue")
    
    
    try:
        user_input=r.recognize_google(audio)
        print(user_input)
    except:
        continue
    
    #We adjust hyperparameters according with openAI
    conversation += "\n Camilo: " + user_input + "\n AI:"
    response = openai.Completion.create(
    model="text-davinci-002",
    prompt=text,
    temperature=0.7,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    answer= response.choices[0].text.strip()
    conversation += answer
    print(user_input)
    print("AI: "+ answer +"\n")
    engine.say(answer)
    engine.runAndWait()
    break
  
# Once you run the code it should be able to chow the text in terminal and also listening  




    