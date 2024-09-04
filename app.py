import streamlit as st
import asyncio
from utils.audio_processing import capture_and_transcribe_audio
from utils.llm_interaction import generate_llm_response
from utils.tts_conversion import convert_text_to_speech, play_audio

st.set_page_config(page_title="Voice Assistant", layout="wide")


if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

st.sidebar.title("Settings")
pitch = st.sidebar.slider("Pitch", -10, 10, 0, 1)
speed = st.sidebar.slider("Speed", -50, 50, 0, 1)
voice = st.sidebar.selectbox("Voice", ["en-US-JennyNeural", "en-US-GuyNeural"])

st.title("🎙️ Voice Assistant")
st.write("Click the button below to start speaking. The assistant will listen, transcribe, respond, and speak back to you.")

status_placeholder = st.empty()
transcription_placeholder = st.empty()
response_placeholder = st.empty()

def update_conversation(user_text, assistant_text):
    st.session_state.conversation_history.append({"User": user_text, "Assistant": assistant_text})

def display_conversation():
    conversation_display = ""
    for entry in st.session_state.conversation_history:
        conversation_display += f"**You**: {entry.get('User', '')}\n\n"
        conversation_display += f"**Assistant**: {entry.get('Assistant', '')}\n\n"
    st.markdown(conversation_display)

if st.button("Start Speaking"):
    status_placeholder.text("Listening... 🎙️")
    image_placeholder=st.image("assets/listening_audio_files/listening_audio.gif", width=150, caption="Listening...") 
    transcribed_text = asyncio.run(capture_and_transcribe_audio())
    image_placeholder.empty()
    if not transcribed_text:
        status_placeholder.text("Please try speaking again.")
    else:
        transcription_placeholder.markdown(f"**You said**: {transcribed_text}")
        status_placeholder.text("Generating response... 🤖")
        
        user_inputs = ' '.join([entry['User'] for entry in st.session_state.conversation_history])
        response = generate_llm_response(transcribed_text, user_inputs)
        response_placeholder.markdown(f"**Assistant**: {response}")

        if pitch >= 0:
            pitch = f"+{pitch}"
        if speed >= 0:
            speed = f"+{speed}"
        
        audio_file = asyncio.run(convert_text_to_speech(response, voice=voice, pitch=f"{pitch}Hz", rate=f"{speed}%"))
        try:
            play_audio(audio_file)
        except Exception as e:
            st.error(f"Error playing audio: {e}")
        
        update_conversation(transcribed_text, response)
        status_placeholder.text("Interaction completed. Click the button to start again.")

display_conversation()
