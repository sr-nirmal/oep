import threading
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import time
from pvrecorder import PvRecorder

class Audio:
    def __init__(self):
        # Initialize audio-related attributes
        self.audio_xs = []
        self.audio_ys = []
        self.threshold = []
        self.time = 0
        self.lock = threading.Lock()

        self.device_index = -1  # -1 uses the default audio input device
        self.frame_length = 512
        self.volume_threshold = 100  # Define a threshold for the volume
        self.audio_input = PvRecorder(device_index=self.device_index, frame_length=self.frame_length)

    def calculate_rms(self, frame):
        # Calculate RMS volume
        rms = np.sqrt(np.mean(np.square(frame)))
        return rms

    def audio_thread(self):
        self.audio_input.start()
        above_threshold_count = 0  # Counter for consecutive readings above threshold
        below_threshold_count = 0  # Counter for consecutive readings below threshold

        while True:
            try:
                # Read the audio input frame
                self.audio_frame = self.audio_input.read()
            except Exception as e:
                print("Error reading audio frame:", e)
                continue  # Skip iteration if reading fails

            with self.lock:
                # Calculate the volume
                volume = self.calculate_rms(self.audio_frame)
                self.audio_xs.append(self.time)
                self.audio_ys.append(volume)
                self.threshold.append(self.volume_threshold)
                self.time += 0.5

                # Check if the volume is above or below the threshold
                if volume > self.volume_threshold:
                    above_threshold_count += 1
                    below_threshold_count = 0  # Reset the below-threshold counter
                else:
                    below_threshold_count += 1
                    above_threshold_count = 0  # Reset the above-threshold counter

                # Adjust threshold if consecutive counts exceed 500 iterations
                if above_threshold_count > 500:
                    self.volume_threshold += np.mean(self.audio_ys[-500:])  # Aggregate based on recent data
                    above_threshold_count = 0  # Reset the counter after adjusting

                elif below_threshold_count > 500:
                    self.volume_threshold = max(0, self.volume_threshold - np.mean(self.audio_ys[-500:]))  # Decrease with a limit
                    below_threshold_count = 0  # Reset the counter after adjusting

            time.sleep(0.01)  # Delay for stability


    def run(self):
        
        audio_thread = threading.Thread(target=self.audio_thread)
        audio_thread.start()

       
        st.title("Real-Time Audio Volume Plot")
        plot_area = st.empty() 

        max_time = 30  

        while True:
           
            with self.lock:
                fig, ax = plt.subplots()
                ax.set_title("Real-Time Audio Volume Plot")
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Volume")

              
                if len(self.audio_xs) > 0 and self.audio_xs[-1] > max_time:
                    self.audio_xs = [x - self.time + max_time for x in self.audio_xs if x >= self.time - max_time]
                    self.audio_ys = self.audio_ys[-len(self.audio_xs):]
                    self.threshold = self.threshold[-len(self.audio_xs):]

                ax.plot(self.audio_xs, self.audio_ys, lw=2, label="Volume")
                ax.plot(self.audio_xs, self.threshold, lw=2, color="red", label="Threshold")
                ax.set_xlim(max(self.time - max_time, 0), self.time) 
                ax.legend(loc="upper right")

                
                plot_area.pyplot(fig)

            time.sleep(0.001)  


if __name__ == "__main__":
    object = Audio()
    object.run()
