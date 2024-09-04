import sounddevice as sd
import numpy as np
import soundfile as sf
import aiohttp
import asyncio
import webrtcvad
import time
from dotenv import load_dotenv
import os

load_dotenv()

async def capture_audio_vad(duration=5, filename="audio.flac"):
    fs = 16000  
    vad = webrtcvad.Vad(2)  
    buffer_duration = 0.02  
    num_frames = int(fs * buffer_duration)
    
    audio_data = []
    print("Recording with VAD...")
    
    def audio_callback(indata, frames, time, status):
        if status:
            print(f"Status: {status}")  
        audio_chunk = indata[:, 0].astype(np.int16).tobytes()
        
        if len(audio_chunk) == num_frames * 2: 
            if vad.is_speech(audio_chunk, fs):
                audio_data.append(indata.copy())
        else:
            print(f"Unexpected audio chunk size: {len(audio_chunk)} bytes")

    stream = sd.InputStream(samplerate=fs, channels=1, dtype='int16', blocksize=num_frames, callback=audio_callback)
    with stream:
        sd.sleep(int(duration * 1000))
    
    print("Recording complete.")
    
    if audio_data:
        audio = np.concatenate(audio_data, axis=0)
        sf.write(filename, audio, fs, format='FLAC')
        return filename
    else:
        print("No speech detected.")
        return None

async def send_audio_in_chunks(filename):
    HF_API_TOKEN =  os.getenv("HF_API_KEY")
    HF_API_URL = "https://api-inference.huggingface.co/models/openai/whisper-small"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

    if filename:
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
    audio_file = await capture_audio_vad()
    
    if audio_file:
        start= time.time()
        await send_audio_in_chunks(audio_file)
        end = time.time()
        print("Time taken: ", end-start)

asyncio.run(main())