import wave
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from matplotlib.widgets import SpanSelector

# Load the WAV file
def load_wave_file(file_path):
    with wave.open(file_path, 'rb') as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        signal = wav_file.readframes(-1)
        signal = np.frombuffer(signal, dtype=np.int16)
        return signal, rate, frames

# Plot the waveform and spectrogram side by side
def plot_waveform_and_spectrogram(signal, rate):
    time = np.linspace(0, len(signal) / rate, num=len(signal))

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot the waveform
    ax1.plot(time, signal)
    ax1.set_title("Waveform")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Amplitude")

    # Compute and plot the spectrogram using numpy.fft
    spectrogram_data = compute_spectrogram(signal, rate)
    extent = [0, len(signal) / rate, 0, rate / 2]  # Time and frequency extent
    ax2.imshow(10 * np.log10(spectrogram_data), aspect='auto', extent=extent, origin='lower')
    ax2.set_title("Spectrogram (numpy.fft)")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Frequency [Hz]")

    plt.tight_layout()
    return fig, (ax1, ax2)

# Compute the spectrogram using numpy.fft
def compute_spectrogram(signal, rate, window_size=1024, overlap=512):
    hop_size = window_size - overlap
    num_windows = (len(signal) - window_size) // hop_size + 1

    # Initialize the spectrogram array
    spectrogram = np.zeros((window_size // 2, num_windows))

    # Apply a Hann window
    window = np.hanning(window_size)

    for i in range(num_windows):
        start = i * hop_size
        end = start + window_size
        segment = signal[start:end] * window  # Apply windowing
        fft_result = np.fft.fft(segment)[:window_size // 2]  # Take only positive frequencies
        spectrogram[:, i] = np.abs(fft_result)  # Store magnitude

    return spectrogram

# Callback function for selecting a section
def onselect(xmin, xmax):
    global selected_section
    selected_section = (xmin, xmax)
    print(f"Selected section: {xmin:.2f}s to {xmax:.2f}s")

# Play the selected section using sounddevice
def play_selected_section(signal, rate, start_time, end_time):
    start_sample = int(start_time * rate)
    end_sample = int(end_time * rate)
    selected_signal = signal[start_sample:end_sample]
    sd.play(selected_signal, rate)
    sd.wait()  # Wait until playback is finished

# Compute and plot the FFT of the selected section
def plot_fft(signal, rate, start_time, end_time):
    start_sample = int(start_time * rate)
    end_sample = int(end_time * rate)
    selected_signal = signal[start_sample:end_sample]

    # Compute the FFT
    n = len(selected_signal)
    fft_result = np.fft.fft(selected_signal)
    fft_freqs = np.fft.fftfreq(n, d=1/rate)

    # Plot the FFT magnitude
    plt.figure(figsize=(10, 4))
    plt.plot(fft_freqs[:n // 2], np.abs(fft_result[:n // 2]))
    plt.title("FFT of Selected Section")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.grid()
    plt.show()

# Main function
def main():
    file_path = "h123.wav"  # Replace with your WAV file path
    signal, rate, frames = load_wave_file(file_path)
    fig, (ax1, ax2) = plot_waveform_and_spectrogram(signal, rate)

    # Add span selector for selecting a section
    span = SpanSelector(ax1, onselect, 'horizontal', useblit=True)

    plt.show()

    # After closing the plot, play the selected section and compute FFT
    if 'selected_section' in globals():
        start_time, end_time = selected_section
        play_selected_section(signal, rate, start_time, end_time)
        plot_fft(signal, rate, start_time, end_time)

if __name__ == "__main__":
    main()