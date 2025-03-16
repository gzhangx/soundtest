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
def plot_waveform_and_spectrogram(signal, rate, title):
    time = np.linspace(0, len(signal) / rate, num=len(signal))

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot the waveform
    ax1.plot(time, signal)
    ax1.set_title(f"{title} - Waveform")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Amplitude")

    # Compute and plot the spectrogram using numpy.fft
    spectrogram_data = compute_spectrogram(signal, rate)
    extent = [0, len(signal) / rate, 0, rate / 2]  # Time and frequency extent
    ax2.imshow(10 * np.log10(spectrogram_data), aspect='auto', extent=extent, origin='lower')
    ax2.set_title(f"{title} - Spectrogram (numpy.fft)")
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

    # Extract the selected section
    start_sample = int(xmin * rate)
    end_sample = int(xmax * rate)
    selected_signal = signal[start_sample:end_sample]

    # Plot the selected section in a new window
    plot_selected_section(selected_signal, rate, xmin, xmax)

# Plot the selected section in a new window
def plot_selected_section(selected_signal, rate, start_time, end_time):
    fig, (ax1, ax2) = plot_waveform_and_spectrogram(selected_signal, rate, f"Selected Section ({start_time:.2f}s to {end_time:.2f}s)")
    plt.show()

    # Play the selected section
    play_selected_section(selected_signal, rate)

# Play the selected section using sounddevice
def play_selected_section(signal, rate):
    sd.play(signal, rate)
    sd.wait()  # Wait until playback is finished

# Main function
def main():
    global signal, rate
    file_path = "h123.wav"  # Replace with your WAV file path
    signal, rate, frames = load_wave_file(file_path)
    fig, (ax1, ax2) = plot_waveform_and_spectrogram(signal, rate, "Full Audio")

    # Add span selector for selecting a section
    span = SpanSelector(ax1, onselect, 'horizontal', useblit=True)

    plt.show()

if __name__ == "__main__":
    main()