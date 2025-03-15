# main.py
#from audio_utils import record_to_numpy, save_numpy_to_wav
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import wave

def simple_spectrogram(audio_data, sample_rate, window_size=256, overlap=128):
    """
    Compute and plot a basic spectrogram using NumPy FFT.
    Adjusted for low sample rate (8000 Hz).
    """
    n_samples = len(audio_data)
    step = window_size - overlap
    n_windows = (n_samples - overlap) // step

    # Initialize spectrogram array
    spectrogram = []
    for i in range(n_windows):
        start = i * step
        end = start + window_size
        window = audio_data[start:end] * np.hanning(window_size)  # Apply Hanning window
        fft_result = np.abs(np.fft.fft(window))[:window_size // 2]  # Positive frequencies only
        spectrogram.append(fft_result)

    spectrogram = np.array(spectrogram).T  # Transpose for plotting
    frequencies = np.fft.fftfreq(window_size, 1 / sample_rate)[:window_size // 2]
    times = np.arange(n_windows) * step / sample_rate

    # Plot
    plt.figure(figsize=(10, 6))
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.pcolormesh(times, frequencies, 10 * np.log10(spectrogram + 1e-10), shading='gouraud')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.title('Spectrogram (8000 Hz, 8-bit)')
    plt.colorbar(label='Intensity (dB)')
    plt.ylim(0, 4000)  # Max frequency is Nyquist (sample_rate / 2 = 4000 Hz)
    plt.tight_layout()

    fig.audio_data = audio_data
    fig.sample_rate = sample_rate

    # Connect click event
    fig.canvas.mpl_connect('button_press_event', on_click)

    plt.show()

def on_click(event):
    """Play audio from the clicked time position."""
    if event.xdata is None:  # Click outside plot
        return

    audio_data = event.canvas.figure.audio_data
    sample_rate = event.canvas.figure.sample_rate
    start_time = event.xdata  # Time in seconds
    start_sample = int(start_time * sample_rate)

    if start_sample < len(audio_data):
        print(f"Playing from {start_time:.2f} seconds...")
        playback_data = audio_data[start_sample:start_sample+int(sample_rate/8)]
        sd.play(playback_data, sample_rate)
        sd.wait()  # Wait for playback to finish

#def recordToFile():
#    # Record audio with lowest rate (8000 Hz) and format (paInt8)
#    audio_data = record_to_numpy(record_seconds=5)

#    # Save to WAV file
#    save_numpy_to_wav(audio_data, filename="lowest_rate_recording.wav")

def load_wav_to_numpy(filename):
    """
    Load a WAV file into a NumPy array using wave and numpy.
    """
    with wave.open(filename, 'rb') as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        n_frames = wf.getnframes()

        raw_data = wf.readframes(n_frames)

        if sample_width == 1:
            dtype = np.int8  # 8-bit signed
        elif sample_width == 2:
            dtype = np.int16
        elif sample_width == 4:
            dtype = np.int32
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")

        audio_data = np.frombuffer(raw_data, dtype=dtype)

        if channels > 1:
            audio_data = audio_data.reshape(-1, channels)

        print(f"Loaded {filename}")
        print(f"Sample Rate: {sample_rate} Hz")
        print(f"Channels: {channels}")
        print(f"Audio Shape: {audio_data.shape}")

        return sample_rate, audio_data

def main():

    # Load the WAV file back into a NumPy array
    sample_rate, loaded_audio = load_wav_to_numpy("h123.wav")

    print("sample rate {sample_rate} ")
    # Verify the data
    print("First 10 samples of loaded audio:", loaded_audio[:10])

    # Generate and display spectrogram
    simple_spectrogram(loaded_audio, sample_rate)

if __name__ == "__main__":
    main()