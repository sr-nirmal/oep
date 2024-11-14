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
        self.volume_threshold = 6000  # Define an initial threshold for the volume
        self.audio_input = PvRecorder(device_index=self.device_index, frame_length=self.frame_length)
        self.count_limit = 500  # Threshold adjustment counter limit

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

                
                if volume > self.volume_threshold:
                    above_threshold_count += 1
                    below_threshold_count = 0  
                else:
                    below_threshold_count += 1
                    above_threshold_count = 0 

               
                if (above_threshold_count > self.count_limit or below_threshold_count > self.count_limit):
                    
                    recent_data = self.audio_ys[-self.count_limit:]
                    new_average_volume = np.mean(recent_data)
                    volume_std = np.std(recent_data)

                    
                    self.volume_threshold = (new_average_volume + volume_std * 0.5)*1.5

            time.sleep(0.01) 

    def plot(self):
        audio_thread = threading.Thread(target=self.audio_thread)
        audio_thread.start()

        st.title("Real-Time Audio Volume Plot")
        plot_area = st.empty()

        max_time = 300  

        while True:
            with self.lock:
                
                fig, ax = plt.subplots()
                ax.set_title("Real-Time Audio Volume Plot")
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Volume")

              
                # if len(self.audio_xs) > 0 and self.audio_xs[-1] > max_time:
                    
                
                plot_x = [i for i in range(int(self.time), int(self.time + max_time))]
                
                audio_plot = self.audio_ys[-len(plot_x):]
                threshold_plot = self.threshold[-len(plot_x):]
                print(len(audio_plot), len(threshold_plot), len(plot_x))

                temp_l = len(audio_plot)
                try:

                    
                    ax.plot(plot_x[:temp_l], audio_plot, lw=2, label="Volume")
                    ax.plot(plot_x[:temp_l], threshold_plot, lw=2, color="red", label="Threshold")
                    ax.set_xlim(max(plot_x[0], 0), plot_x[-1])
                    ax.legend(loc="upper right")
                except:
                    pass
                    

                
                plot_area.pyplot(fig)

            time.sleep(0.1)  

if __name__ == "__main__":
    audio_object = Audio()
    audio_object.plot()
