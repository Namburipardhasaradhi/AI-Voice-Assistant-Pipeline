import numpy as np
import soundfile as sf
import asyncio
import time
import os
import speech_recognition as sr
from faster_whisper import WhisperModel

VAD_THRESHOLD = 0.5  

async def capture_audio(duration=5, filename="stt_transcribe.flac"):
    recognizer = sr.Recognizer()
    print("Listening...\n")

    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        audio_data = recognizer.record(source, duration=duration)

    audio = np.frombuffer(audio_data.get_raw_data(), np.float32)
    
    output_dir = "Testing/audio files" 
    os.makedirs(output_dir, exist_ok=True) 
    
    full_filename = os.path.join(output_dir, filename)

    sf.write(full_filename, audio, int(audio_data.sample_rate), format='FLAC')
    if vad_filter(audio, threshold=VAD_THRESHOLD):
        return full_filename
    else:
        print("No significant audio detected. \n")
        return None

def vad_filter(audio, threshold=VAD_THRESHOLD):
    return np.max(np.abs(audio)) > threshold

async def transcribe_audio(filename, language="en"):
    model = WhisperModel("tiny", device="cpu", compute_type="int8")  

    segments, info = model.transcribe(filename, language=language)
    transcribed_text = ""
    for segment in segments:
        transcribed_text += segment.text + " "

    return transcribed_text

async def main():
    audio_file = await capture_audio()

    if audio_file:
        start = time.time()
        await transcribe_audio(audio_file)
        end = time.time()
        print("Time taken: ", end-start)

if __name__ == "__main__":
    asyncio.run(main())