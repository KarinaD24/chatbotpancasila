import requests
import re
import speech_recognition as sr
import os 
import time
import streamlit as st
from playsound import playsound
from gtts import gTTS
from io import BytesIO
import PyPDF2
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

r = sr.Recognizer()

# API DeepSeek Via OpenRouter 
url = "https://openrouter.ai/api/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {os.getenv('API_KEY')}", // DEEPSEEK_API_KEY
    "Content-Type": "application/json",
}

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text.strip() + "\n\n"
    except Exception as e:
        print(f"❌ Gagal mengekstrak teks: {e}")
    
    return text.strip()

def chunking(study_material):
    max_len = 1000 # Max text length in each chunk
    sentences = re.split(r'(?<=[.!?]) +', study_material)
    chunks = []
    curr_chunk = ""

    for s in sentences:
        if len(curr_chunk) + len(s) <= max_len:
            curr_chunk += " " + s
        else:
            chunks.append(curr_chunk.strip())
            curr_chunk = s # Start a new chunk with a new text
    
    if curr_chunk:
        chunks.append(curr_chunk.strip())

    return chunks

def find_chunk(chunks, question):
    # Find the most relevant chunk using Sentence embedding
    model = SentenceTransformer("all-MiniLM-L6-v2") 
    embeddings = model.encode(chunks + [question])
    
    chunk_embeddings = embeddings[:-1]
    question_embedding = embeddings[-1].reshape(1, -1)

    similarities = cosine_similarity(question_embedding, chunk_embeddings)[0]
    best_chunk_index = np.argmax(similarities)

    return chunks[best_chunk_index]

def ask_deepseek(question, study_material):
    # Prompt construction
    data = {
        "model": "openrouter/free", // deepseek/deepseek-r1:free
        "messages": [
            {
                "role": "user", 
                "content": f"Materi pembelajaran: {study_material}\n\nPertanyaan: {question}"
            },
            {
                "role": "system",
                "content": (
                    "Jawablah pertanyaan pengguna sebagai asisten AI Pancasila dan Character Building yang ramah berdasarkan materi pembelajaran..\n"
                    "- Jika pertanyaannya berkaitan dengan konteks yang diberikan, berikan jawaban yang informatif dan relevan.\n"
                    "- Jika pertanyaannya bersifat umum atau tidak terkait dengan konteks, tetaplah merespons secara alami seperti AI percakapan. Bersikaplah ramah.\n"
                    "- Jawaban harus singkat, namun tetap menarik dan bersahabat.\n"
                    "- Jawaban dalam bahasa Indonesia.\n"
                    "- Jika tidak yakin, beri tahu pengguna bahwa kamu tidak memiliki cukup konteks, namun tetap usahakan untuk membantu."
                )
            }
        ]
    }

    # Send request POST the API
    response = requests.post(url, headers=headers, json=data)

    # Check the response to see if the API call was successful.
    if response.status_code == 200: 
        print("\nRAW API response:")
        print(response.json())

        # Get response from the AI
        return response.json().get("choices", [{}])[0].get("message", {}).get("content", "No response.")
    return "Error fetching response."

def ask_ai(question, context, topic):
    all_folder = "data/"
    study_material = ""

    # Pick the selected topic
    topic_folders = {
        "Pancasila": "PA",
        "Character Building": "CB",
    }

    # Read the slides only from the selected topic
    if topic == "All Topics":
        selected_folders = [os.path.join(all_folder, folder) for folder in topic_folders.values()]
    else:
        selected_folders = [os.path.join(all_folder, topic_folders[topic])]

    # Extract text dr pdf yg dr selected folder
    for folder in selected_folders:
        if os.path.exists(folder):  
            for file in os.listdir(folder):
                if file.endswith(".pdf"):  
                    pdfs = os.path.join(folder, file)

                    study_material += extract_text_from_pdf(pdfs) + "\n\n"

    # Chunking the text and find the most relevant text (from the ppt) based on the user's question  
    chunks = chunking(study_material)
    most_relevant_chunk = find_chunk(chunks, question)

    # Additional validation
    if not isinstance(most_relevant_chunk, str) or not most_relevant_chunk.strip():
        most_relevant_chunk = "No relevant information found."

    response = ask_deepseek(question, most_relevant_chunk)

    return response

def textToSpeech(response):
    if not response or response.strip() == "":  # Check response whether it's empty or not
        st.error("Belum menemukan jawaban dari sumber pengetahuan")
        response = "Maaf, saya kurang paham pertanyaan kamu, mohon diulangi."
        return response
    
    audio_bytes = BytesIO()
    tts = gTTS(text=response, lang="id") # Make the speech audio from the response text
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)

    return audio_bytes.read()

def speechToText():
    try:
        with sr.Microphone() as source2: # Activating microphone 
            print("Mic On! Saya mendengar suara Anda")
            st.toast("🎤 Mic On! Silahkan berbicara")
            print("")

            r.adjust_for_ambient_noise(source2, duration=0.2)
            audio2 = r.listen(source2)

            # Convert speech to text in Indonesia
            prompt = r.recognize_google(audio2, language="id-ID") 
            prompt = prompt.lower()

            print("Pertanyaanmu adalah: ", prompt)

            st.session_state.messages.append({"role": "user", "content": prompt})

            st.toast("✅ Suara terdeteksi yeay!")
            
            with st.chat_message("user"):
                st.write(prompt)

            # Get AI response
            response = ask_ai(prompt, st.session_state.context, st.session_state.topic)
            cleaned_response = re.sub(r"[*_#]", "", response) # Clean markdown format
            audio_data = textToSpeech(cleaned_response)

            # Save response to chat history
            st.session_state.messages.append({"role": "assistant", "content": cleaned_response, "audio": audio_data if audio_data else None})
            st.session_state.audio_files.append(audio_data)

            with st.chat_message("assistant"):
                st.write(cleaned_response)
                st.audio(audio_data, format="audio/mp3")

    except Exception as e:
        print(e)
        print("Could not request results; {0}".format(e))

# GUI via streamlit
def main():
    st.set_page_config(page_title="Chatbot Pancasila", layout="wide")

    with st.sidebar:
        col1, col2 = st.columns([0.4, 0.85]) 

        with col1:
            with st.container(): 
                st.image("robot.png", width=75)  

        with col2:
            st.markdown("""
                <div style="display: flex; flex-direction: column; justify-content: center; height: 100%;">
                    <h1 style="margin: 0; padding: 0;">ROBOT PANCASILA @2025</h1>
                    <h4 style="margin: 0; margin-bottom: 10px;">Copyright Widodo Budiharto, Frederick Fios dan Karlina Dwinovera Mulia</h4>
                </div>
            """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown(
            """
            <p style='text-align: left; font-size: 14px; color: gray; margin-bottom: 30px;'>
            ROBOT PANCASILA  is an AI-powered robot from  BINUS University for teaching Pancasila and Character Building
            </p>
            """, 
            unsafe_allow_html=True
        )

        topics = st.sidebar.selectbox(
            "Select Topic", 
            ["All Topics", "Pancasila", "Character Building"], 
            index=0
        )

        st.session_state.topic = topics

        st.markdown(
            """
            <p style='text-align: left; font-size: 14px; color: black; margin-bottom: 5px;'>
                Try asking using your voice!
            </p>
            """, 
            unsafe_allow_html=True
        )

        mic_clicked = st.button("🎤 Speech Mode", key="mic_button", help="Click to start voice input")

        if mic_clicked:
            st.session_state.chat_input = speechToText()
            time.sleep(1)
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = [] 
        st.session_state.audio_files = []  
        st.session_state.context = ""

        # AI Greets user when the chat starts for the first time
        opening_greet = "Hai! Aku ROBOT PANCASILA 🤖 Siap bantu kamu belajar Pancasila dan Character Building. Mau tanya apa hari ini?"
        opening_audio = textToSpeech(opening_greet)

        st.session_state.messages = [{"role": "assistant", "content": opening_greet, "audio": opening_audio}]
        st.session_state.audio_files.append(opening_audio)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and "audio" in msg:
                st.audio(msg["audio"], format="audio/mp3")

    if user_input := st.chat_input("Ask your question..."):
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.write(user_input)

        response = ask_ai(user_input, st.session_state.context, topics)

        # Additional validation
        response = response if response else "Maaf, saya tidak bisa menemukan jawaban."

        cleaned_response = re.sub(r"[*_#]", "", response)   
        audio_data = textToSpeech(cleaned_response)

        st.session_state.messages.append({"role": "assistant", "content": response, "audio": audio_data if audio_data else None})
        st.session_state.audio_files.append(audio_data)

        with st.chat_message("assistant"):
            st.write(response)
            st.audio(audio_data, format="audio/mp3")

main()
