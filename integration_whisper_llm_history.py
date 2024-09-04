import asyncio
from collections import deque
from Models.faster_whisper_stt_tiny import capture_audio, transcribe_audio 
from Models.llm_response import generate 
import time

conversation_memory = deque(maxlen=10)

async def main_interaction_loop():
    while True:
        audio_file = await capture_audio()
        if not audio_file:
            print("Please try speaking again.")
            continue

        transcribed_text = await transcribe_audio(audio_file)
        if 'watching' in transcribed_text or "Let's go" in transcribed_text :
            print("Unclear transcription, please try again.")
            continue
        print(f"You said: {transcribed_text}\n") 

        conversation_memory.append(transcribed_text)

        memory_context = " ".join(conversation_memory)

        if 'stop' in transcribed_text or 'Stop' in transcribed_text:
            print("Goodbye!")
            break

        prompt = f"With context '{memory_context}', respond to the latest input: '{transcribed_text}' in at most 20 words."
        response = generate(
            prompt,
            system_prompt="Be concise, helpful, and friendly. Respond briefly with at most 20 words. Always end your response with a positive or friendly note, like a smiley emote.",
            model="mistralai/Mistral-7B-Instruct-v0.3",
            temperature=0.7,
            chat_template="mistral",
            verbose=False
        )

        print(f"Assistant: {response}\n")
        
asyncio.run(main_interaction_loop())