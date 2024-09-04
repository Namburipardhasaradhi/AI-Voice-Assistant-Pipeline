import sounddevice as sd
import numpy as np
import soundfile as sf
import aiohttp
import asyncio
import time
import os
from dotenv import load_dotenv

load_dotenv()
VAD_THRESHOLD = 0.5  

async def capture_audio(duration=5, filename="audio.flac"):
    fs = 16000  
    print("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.float32)
    sd.wait()  
    print("Recording complete.")

    if vad_filter(audio, threshold=VAD_THRESHOLD):
        sf.write(filename, audio, fs, format='FLAC')
        return filename
    else:
        print("No significant audio detected.")
        return None

def vad_filter(audio, threshold=VAD_THRESHOLD):
    return np.max(np.abs(audio)) > threshold

async def send_audio_in_chunks(filename):
    HF_API_TOKEN = os.getenv("HF_API_KEY")
    HF_API_URL = "https://api-inference.huggingface.co/models/openai/whisper-tiny"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

    async with aiohttp.ClientSession() as session:
        with open(filename, "rb") as f:
            async with session.post(HF_API_URL, headers=headers, data=f) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'text' in data:
                        print("Transcribed Text:", data['text'])
                    else:
                        print("Error or different response format:", data)
                else:
                    print(f"Failed with status code: {response.status}")
                    print(await response.text())

async def main():
    audio_file = await capture_audio()
    if audio_file:
        start  = time.time()
        await send_audio_in_chunks(audio_file)
        end = time.time()
        print("Time taken: ", end-start)
    else:
        print("No valid audio to process.")

asyncio.run(main())