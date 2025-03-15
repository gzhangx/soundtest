import wave
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
import sounddevice as sd

# Load the wave file
def load_wave_file(file_path):
    with wave.open(file_path, 'rb') as wav_file:
        sample_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        audio_data = wav_file.readframes(n_frames)
        audio_data = np.frombuffer(audio_data, dtype=np.int16)
    return sample_rate, audio_data

# Perform FFT
def perform_fft(data):
    n = len(data)
    fft_result = fft(data)
    freq = np.fft.fftfreq(n, d=1/sample_rate)
    return freq[:n//2], fft_result[:n//2]

# Adjust FFT using mouse
def adjust_fft(freq, fft_result, original_data):
    fig, ax = plt.subplots()
    magnitude = np.abs(fft_result)
    line, = ax.plot(freq, magnitude, lw=2, picker=True, pickradius=5)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude')

    # Store the original FFT for reference
    original_fft = fft_result.copy()

    # Function to update the graph and audio
    def update_audio():
        modified_fft = np.zeros(len(original_data), dtype=complex)
        modified_fft[:len(fft_result)] = fft_result
        modified_fft[len(fft_result):] = np.conj(fft_result[::-1])
        modified_data = np.real(ifft(modified_fft)).astype(np.int16)
        sd.play(modified_data, samplerate=sample_rate)
        sd.wait()

    # Event handler for mouse clicks and drags
    def on_pick(event):
        ind = event.ind[0]
        point = line.get_data()
        x, y = point[0][ind], point[1][ind]

        def on_motion(event):
            if event.inaxes != ax:
                return
            new_y = event.ydata
            if new_y is None:
                return
            fft_result[ind] = fft_result[ind] * (new_y / magnitude[ind])
            magnitude[ind] = new_y
            line.set_ydata(magnitude)
            fig.canvas.draw_idle()

        def on_release(event):
            fig.canvas.mpl_disconnect(motion_id)
            fig.canvas.mpl_disconnect(release_id)
            update_audio()

        motion_id = fig.canvas.mpl_connect('motion_notify_event', on_motion)
        release_id = fig.canvas.mpl_connect('button_release_event', on_release)

    fig.canvas.mpl_connect('pick_event', on_pick)
    plt.show()

# Main function
def main(file_path):
    global sample_rate
    sample_rate, audio_data = load_wave_file(file_path)
    freq, fft_result = perform_fft(audio_data)
    adjust_fft(freq, fft_result, audio_data)

if __name__ == "__main__":
    file_path = 'h123.wav'  # Replace with your wave file path
    main(file_path)