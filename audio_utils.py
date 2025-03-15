# audio_utils.py
import pyaudio
import numpy as np
import wave

RATE = 8000
CHANNELS = 1
FORMAT = pyaudio.paInt16
BYTES_PER_SAMPLE = 2 #16 bit
def record_to_numpy(record_seconds=5, chunk=1024):
    """
    Record audio into a NumPy array with the lowest rate (8000 Hz) and format (paInt8).
    """
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT,
                       channels=CHANNELS,
                       rate=RATE,
                       input=True,
                       frames_per_buffer=chunk)

    print("Recording started...")
    total_samples = int(RATE * record_seconds)
    buffers = []  # List to store raw audio buffers

    # Record chunks into buffers
    for _ in range(0, int(total_samples / chunk)):
        data = stream.read(chunk)
        buffers.append(np.frombuffer(data, dtype=np.int8))

    # Handle remainder if total_samples isn’t a multiple of chunk
    remainder = total_samples % chunk
    if remainder:
        data = stream.read(remainder)
        buffers.append(np.frombuffer(data, dtype=np.int8))

    # Concatenate all buffers into a single NumPy array
    audio_data = np.concatenate(buffers)

    print("Recording finished.")
    stream.stop_stream()
    stream.close()
    audio.terminate()

    return audio_data

def save_numpy_to_wav(audio_data, filename):
    """
    Save a NumPy array as a WAV file.
    """
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(BYTES_PER_SAMPLE)  # 1 byte for int8
    wf.setframerate(RATE)
    audio_buffer = audio_data.tobytes()
    wf.writeframes(audio_buffer)
    wf.close()
    print(f"Audio saved to {filename}")

