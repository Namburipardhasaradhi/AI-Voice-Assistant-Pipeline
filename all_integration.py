import asyncio
from Models.faster_whisper_stt_tiny import capture_audio, transcribe_audio 
from Models.llm_response import generate  
from ...utils.tts_conversion import convert_text_to_speech  
import edge_tts
from playsound import playsound  
import os  
import time


async def main_interaction_loop():
    conversation_history = []  
    
    while True:
        audio_file = await capture_audio()
        if not audio_file:
            print("Please try speaking again.")
            continue

        transcribed_text = await transcribe_audio(audio_file, language="en")
        if 'watching' in transcribed_text or "Let's go" in transcribed_text:
            print("Unclear transcription, please try again.")
            continue
        print(f"You said: {transcribed_text}\n") 

        if 'stop' in transcribed_text.lower():
            print("Goodbye!")
            break

        conversation_history.append({"User": transcribed_text})
        
        history_context = ' '.join([f"{key}: {value}" for entry in conversation_history for key, value in entry.items()])
        response = generate(
            f"Respond to '{transcribed_text}' based on the following context: {history_context}. Don't exceed more than 20 words.",
            system_prompt="Be concise, helpful, and friendly. Respond briefly with at most 20 words. Always end your response with a friendly note. Only Include Text in the entire response.",
            model="mistralai/Mistral-7B-Instruct-v0.3",
            temperature=0.7,
            chat_template="mistral",
            verbose=False
        )
        print(f"Assistant: {response}\n")

        conversation_history.append({"Assistant": response})

        audio_file = await convert_text_to_speech(response)

        playsound(audio_file)
        print("Audio playback finished.\n")


asyncio.run(main_interaction_loop())