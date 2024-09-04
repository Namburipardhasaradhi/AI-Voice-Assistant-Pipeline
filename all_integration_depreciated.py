import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
from huggingface_hub import InferenceApi
from dspy import DynamicPrompt
import edge_tts
import asyncio
from dotenv import load_dotenv
import time
import os

load_dotenv()
def initialize_stt_model():
    return WhisperModel("small", device="cuda", compute_type="int8_float16")

def capture_audio(duration=5, fs=16000):
    print("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    return audio.flatten()

def apply_vad(audio, threshold=0.5):
    vad_audio = []
    for chunk in np.array_split(audio, len(audio) // int(0.02 * 16000)):
        if np.mean(np.abs(chunk)) > threshold:
            vad_audio.extend(chunk)
    return np.array(vad_audio)

HF_API_KEY = os.getenv("HF_API_KEY")
def initialize_llm_model():
    return InferenceApi(repo_id="mistral", token=HF_API_KEY)

def generate_response(prompt, inference_api):
    response = inference_api(inputs=prompt, parameters={"max_length": 20})
    return response.get('generated_text', '')

def generate_dynamic_prompt(query):
    dp = DynamicPrompt()
    dp.add_prompt(f"Q: {query} A:", weight=1.0)
    dp.add_context("You are a helpful assistant.", weight=0.8)
    return dp.render()

async def text_to_speech(text, voice="en-US-JennyNeural", rate="+0%", pitch="+0%"):
    communicate = edge_tts.Communicate(text, voice=voice, rate=rate, pitch=pitch)
    await communicate.save("output_audio.mp3")

def main():
    stt_model = initialize_stt_model()
    llm_inference_api = initialize_llm_model()

    audio = capture_audio()
    vad_audio = apply_vad(audio)
    text_result = stt_model.transcribe(vad_audio, vad_threshold=0.5)
    transcribed_text = text_result['text']
    print("Transcribed Text:", transcribed_text)

    dynamic_prompt = generate_dynamic_prompt(transcribed_text)
    response = generate_response(dynamic_prompt, llm_inference_api)
    print("LLM Response:", response)

    asyncio.run(text_to_speech(response, voice="en-US-GuyNeural", rate="+5%", pitch="-2%"))

if __name__ == "__main__":
    main()