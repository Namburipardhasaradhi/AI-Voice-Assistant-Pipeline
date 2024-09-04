import asyncio
from utils.audio_processing import capture_and_transcribe_audio
from utils.llm_interaction import generate_llm_response
from utils.tts_conversion import convert_text_to_speech, play_audio
import os

async def main_interaction_loop():
    """Main loop for capturing speech, generating responses, and playing audio."""
    conversation_history = [] 

    while True:
        transcribed_text = await capture_and_transcribe_audio()
        if not transcribed_text:
            print("Please try speaking again.")
            continue

        print(f"You said: {transcribed_text}\n") 

        if 'stop' in transcribed_text.lower():
            print("Goodbye!")
            break

        conversation_history.append({"User": transcribed_text})
        
        history_context = ' '.join(
            [f"{key}: {value}" for entry in conversation_history for key, value in entry.items()]
        )
        response = generate_llm_response(transcribed_text, history_context)
        print(f"Assistant: {response}\n")

        conversation_history.append({"Assistant": response})
        audio_file = await convert_text_to_speech(response, rate="+0%", pitch="+0Hz")
        play_audio(audio_file)

        print("Audio playback finished.\n")

if __name__ == "__main__":
    asyncio.run(main_interaction_loop())