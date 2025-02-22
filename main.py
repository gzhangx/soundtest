# main.py
from audio_utils import record_to_numpy, save_numpy_to_wav, load_wav_to_numpy
import numpy as np
import matplotlib.pyplot as plt

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
    plt.pcolormesh(times, frequencies, 10 * np.log10(spectrogram + 1e-10), shading='gouraud')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.title('Spectrogram (8000 Hz, 8-bit)')
    plt.colorbar(label='Intensity (dB)')
    plt.ylim(0, 4000)  # Max frequency is Nyquist (sample_rate / 2 = 4000 Hz)
    plt.tight_layout()
    plt.show()

def recordToFile():
    # Record audio with lowest rate (8000 Hz) and format (paInt8)
    audio_data = record_to_numpy(record_seconds=5)

    # Save to WAV file
    save_numpy_to_wav(audio_data, filename="lowest_rate_recording.wav")

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