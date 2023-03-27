#Here the librarys we used
import openai
import pyttsx3
import speech_recognition as sr
import os

#finding the file with corpus and saving in variable called text
with open(os.path.join(os.getcwd(), "C:\\Users\\USUARIO\\Desktop\\GFK\\finetuned_model.txt"), "r") as f:
    text = f.read()

#we ask request in this comand for connecting to the API, important don´t share the Key API
openai.api_key="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"

# Inicializar el engine de pyttsx3 para la voz
engine = pyttsx3.init()

# Inicializar el reconocimiento de voz
r = sr.Recognizer()

# Configurar el micrófono
mic = sr.Microphone()

# Inicializar la conversación
conversation = ""
user_name = "Camilo"

# Función para obtener la respuesta del modelo Fine-Tuned
def get_response(prompt):
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=prompt,
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].text.strip()

# Esperar órdenes de voz
while True:
    # Esperar hasta escuchar una orden de voz
    with mic as source:
        print("Escuchando:........")
        r.adjust_for_ambient_noise(source, duration=0.1)
        audio = r.listen(source)
    print("Decir 'detener' o 'continuar'")
    
    try:
        # Transcribir la orden de voz
        user_input = r.recognize_google(audio)
        print(user_input)
    except:
        continue
    
    # Actualizar la conversación
    conversation += "\n Camilo: " + user_input + "\n AI:"

    # Si el usuario dice "detener", detener el programa
    if "detener" in user_input:
        break

    # Si el usuario dice "continuar", obtener una respuesta del modelo
    if "continuar" in user_input:
        # Obtener la respuesta del modelo Fine-Tuned
        answer = get_response(text)
        conversation += answer
        print(user_input)
        print("AI: "+ answer +"\n")
        engine.say(answer)
        engine.runAndWait()



    