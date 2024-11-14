import cv2
import threading
import random
from cvzone.HandTrackingModule import HandDetector
from loop_detect.detect import VideoLoopFinder
from matplotlib.animation import FuncAnimation
import time
import streamlit as st
import numpy as np
from pvrecorder import PvRecorder
from gaze_direction.face_Direction import face_tracking
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style




USER_FACE_WIDTH = 140
DEFAULT_WEBCAM = 0


MIN_DETECTION_CONFIDENCE = 0.8
MIN_TRACKING_CONFIDENCE = 0.8


LEFT_EYE_OUTER_CORNER = [33]
LEFT_EYE_INNER_CORNER = [133]
RIGHT_EYE_OUTER_CORNER = [362]
RIGHT_EYE_INNER_CORNER = [263]

UPPER_LIP = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
LOWER_LIP = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]

head_indices_pos = [1, 33, 61, 199, 263, 291]

lefteye_bottom_indices_pos = [160, 159, 158]
lefteye_top_indices_pos = [144, 145, 153]
lefteye_iris_center_indices_pos = [468]
lefteye_leftcorner_indices_pos = [158, 153]
lefteye_rightcorner_indices_pos = [160, 144]

righteye_bottom_indices_pos = [380, 374, 373]
righteye_top_indices_pos = [385, 386, 387]
righteye_iris_center_indices_pos = [473]
righteye_leftcorner_indices_pos = [387, 373]
righteye_rightcorner_indices_pos = [380, 385]
movement_threshold = 0.01


class Video:
    def __init__(self):
     
        self.detector = HandDetector(detectionCon=0.8, maxHands=1)
       
        self.video = cv2.VideoCapture(0)  # 0 for default camera
        
        self.running = True
        
        self.finger_counts = []
        self.frame = None
        self.lock = threading.Lock()

        #loop
        self.current_frames = []
        self.FRAME = 500
        self.check_for_loop = True
        self.is_loop = False
        self.loop_status = ''

        #captcha
        self.frame = None
        self.captcha_running = False
        self.detector = HandDetector(maxHands=1)
        self.timelimit = 15
        self.lock = threading.Lock()
        self.show_text = ''
        self.remainingTime = 0
        self.required_gesture = 0

        #captcha and loop
        self.loop_score = 0
        self.loop_check_count = 0

    def video_stream(self):
       
        while self.running:
            success, frame = self.video.read()
            if not success:
                print("Error accessing video feed.")
                break
            
            
            hands, img = self.detector.findHands(frame, flipType=False)
            with self.lock:
                self.frame = img
                # if (len(self.current_frames) < self.FRAME and self.captcha_running == False):  
                #     self.current_frames.append(img)
            
            # cv2.namedWindow("Live Video Feed", cv2.WINDOW_NORMAL)
            # cv2.imshow("Live Video Feed", frame)
            
            # Press 'q' to stop the video thread
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False

        self.video.release()
        cv2.destroyAllWindows()

    def loop_thread(self):
        INIT = 0
        print("loop detection thread started.....")
        while (True):

            if(self.check_for_loop):
                
                with self.lock:  
                    self.loop_status = "Checking for loop"
                    if (len(self.current_frames) >= self.FRAME and self.captcha_running == False):
                        self.loop_check_count += 1
                        duplicate_frames = self.loop_finder.find_duplicates(self.current_frames)
                        isLooped, loop_frame_count = self.loop_finder.get_vaild_duplicates(duplicate_frames)
                        self.loop_score += loop_frame_count
                        if(isLooped and self.captcha_running == False):
                            self.is_loop = True
                            self.start_captcha_thread()
                            self.loop_status = "loop found"
                        else:
                            print("analysis -> ", self.loop_score, self.loop_check_count, self.loop_score // self.loop_check_count)
                        print("loop ->", isLooped)
                        INIT += self.FRAME
                        self.current_frames = []  
                    else:
                        pass
            else:
                with self.lock:
                    self.loop_status = "No checking"
    def plot(self):
        video_thread = threading.Thread(target=self.video_stream)
        video_thread.start()

        # loop_thread = threading.Thread(target=self.loop_thread)
        # loop_thread.start()
    
        
        st.title("Real-Time Finger Count")
        st.text(self.loop_status)

        
        video_placeholder = st.empty()
        plot_placeholder = st.empty()
        label_placeholder = st.empty()

        while self.running:
            
          
            try:
                frame = self.frame
                
                hands, img = self.detector.findHands(frame, flipType=False)
                finger_count = 0
                if hands:
                    hand = hands[0]
                    finger_count = self.detector.fingersUp(hand).count(1)  
                    self.finger_counts.append(finger_count)
                else:
                    self.finger_counts.append(0)
                
                
                frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB) 
                video_placeholder.image(frame, use_column_width=True)

                
                plot_placeholder.line_chart(self.finger_counts)
                label_placeholder.metric(label="Loop detection status", value=self.loop_status)
                temp = face_tracking(self.frame)
                # for i in temp:
                #     print(i)

                
                if len(self.finger_counts) > 50:
                    self.finger_counts.pop(0)

                
                time.sleep(0.1)
            except:
                pass
   

# Instantiate and start the Video class with threads
if __name__ == "__main__":
    video_stream = Video()
    video_stream.plot()
