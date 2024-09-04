import edge_tts
import os
import pygame
import asyncio

async def convert_text_to_speech(text, voice="en-US-JennyNeural"
                                 , rate="+0%", pitch="+0Hz"):
    output_dir = "Testing/audio files"  
    os.makedirs(output_dir, exist_ok=True)  
    
    output_file = os.path.join(output_dir,
                                "tts_response.mp3") 
    communicator = edge_tts.Communicate(text, 
                                        voice=voice, 
                                        rate=rate, pitch=pitch)
    await communicator.save(output_file)
    return output_file
def play_audio(file_path):
    """Play the audio file using pygame."""
    try:
        pygame.mixer.init()

        pygame.mixer.music.load(file_path)

        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)  

    except Exception as e:
        print(f"Error playing audio: {e}")
    finally:
        pygame.mixer.quit()

if __name__ == "__main__":
    text = "Hello , how are you doing today?"
    output_file = asyncio.run(convert_text_to_speech(text))
    print(f"Audio file saved at: {output_file}")
    play_audio(output_file)

conversation_history = "assistant: Hello, how are you doing today?\n"
def speech_to_text(audio_file):
    """Convert audio file to text using STT."""
    transcribed_text = "Hello, how are you doing today?"    
    return transcribed_text

def generate_response(text):
    """Generate a response based on the input text."""
    response = "I'm doing well, thank you for asking."
    return response 

def text_to_speech(text, pitch="+0Hz", rate="+0%"):
    """Convert text to speech using TTS."""
    output_file = asyncio.run(convert_text_to_speech(text, pitch=pitch, rate=rate))
    return output_file  



import streamlit as st

st.title("AI-Powered Voice Assistant")

audio_input = st.file_uploader
("Upload your audio file")

st.text_area("Conversation History", 
value=conversation_history, height=300)

pitch = st.slider("Pitch", -10, 10, 0)
rate = st.slider("Rate", -50, 50, 0)

if st.button("Process"):
    text = speech_to_text(audio_input)
    response = generate_response(text)
    audio_output = text_to_speech(response, 
    pitch=pitch, rate=f"{rate}%")
    st.audio(audio_output)
    conversation_history += f"""You: {text}
    \nAssistant: {response}\n"""