import cv2
import threading
import random
from cvzone.HandTrackingModule import HandDetector
from loop_detect.detect import VideoLoopFinder
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
class OEP():  # Inherit from VideoLoopFinder
    def __init__(self):
        #init
        self.loop_finder = VideoLoopFinder()
        self.input = True
        
        #loop
        self.current_frames = []
        self.FRAME = 500
        self.check_for_loop = True
        self.is_loop = False
        
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

        #tracking
        self.tracking = True

        self.mesh_points = None
        self.landmarks = None
        self.nose_2D_point = None

        self.face_looks = ''
        self.eye_looks = ''
        self.movement_detected = ''
        self.strike_count = ''

        self._SHOW_GAZE = False
        self._SHOW_EYE = False
        self._SHOW_LIPS = False

        self.angle_x = 0
        self.angle_y = 0
        self.movement_count = 0

        self.current_lip = 0
        self.prev_lip = 0

        #streamlit 
        # st.set_page_config(page_title="Streamlit WebCam App")
        # st.title("Webcam Display Steamlit App")
        # self.frame_placeholder = st.empty()
        # self.stop_button_pressed = st.button("Stop")
        # self.check_captcha = st.button("Check for captcha")

        #audio processing
        self.device_index = -1  # -1 uses the default audio input device
        self.frame_length = 512
        self.volume_threshold = 100  # Define a threshold for the volume
        self.audio_input = PvRecorder(device_index=self.device_index, frame_length=self.frame_length)
        self.audio_alert = ""
        self.check_audio = True
        self.audio_frame = None
        self.AUDIO_FRAME = 500
        self.current_audio = 0


        #plotting
        self.time = 0
        self.audio_xs = []
        self.audio_ys = []
        self.MAX_POINTS = 100
        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(1,1,1)

    def loop_thread(self):
        INIT = 0
        print("loop detection thread started.....")
        while (True):

            if(self.check_for_loop):
                
                with self.lock:  
                    if (len(self.current_frames) >= self.FRAME and self.captcha_running == False):
                        self.loop_check_count += 1
                        duplicate_frames = self.loop_finder.find_duplicates(self.current_frames)
                        isLooped, loop_frame_count = self.loop_finder.get_vaild_duplicates(duplicate_frames)
                        self.loop_score += loop_frame_count
                        if(isLooped and self.captcha_running == False):
                            self.is_loop = True
                            self.start_captcha_thread()
                        else:
                            print("analysis -> ", self.loop_score, self.loop_check_count, self.loop_score // self.loop_check_count)
                        print("loop ->", isLooped)
                        INIT += self.FRAME
                        self.current_frames = []  
                    else:
                        pass

            


    def generate_required_gesture(self):
        return random.randint(1, 5)
    

    def calculate_rms(self, frame):
    # Calculate RMS (Root Mean Square) to measure volume
        rms = np.sqrt(np.mean(np.square(frame)))
        return rms


    def captcha_thread(self):
        
        initialTime = time.time()
        stateResult = False
        requiredGesture = self.generate_required_gesture()
        timeLimit = self.timelimit
    
        #captcha check starting - pause loop check

        
        with self.lock:
            self.show_text = f'Show {requiredGesture} fingers'
            self.remainingTime = 0
            self.check_for_loop = False
            self.required_gesture = requiredGesture
            self.captcha_running = True
            self._SHOW_EYE = False
            self._SHOW_GAZE = False
            self._SHOW_LIPS = False
        print("initialized captcha thread.....", self.required_gesture)
        while(self.captcha_running):
            timer = time.time() - initialTime
            
            
            with self.lock:
                self.remainingTime = timeLimit - timer
                
                
                try:
                    hands, img = self.detector.findHands(self.frame)

                    if(self.remainingTime <= 0):
                        self.show_text = "Rejected! Time Over"
                        
                        break
                    if(hands and stateResult is False):
                        hand = hands[0]
                        fingers = self.detector.fingersUp(hand)
                        fingersCount = fingers.count(1)

                        self.show_text = f'Fingers: {fingersCount}'

                       

                        if(fingersCount == requiredGesture):
                            self.show_text = "Accepted!"
                            
                            break
                except Exception as e:
                    print("Error in captcha-> ", e)
            
        

        # captcha check over - resume loop check
        with self.lock:
            self.show_text = ''
            self.check_for_loop = True
            self.captcha_running = False
        pass

    def start_captcha_thread(self):

        
        print("starting captcha thread....")
        captcha_thread = threading.Thread(target = self.captcha_thread)
        captcha_thread.start()
        print("captcha thread started.... ", self.captcha_running)


    def extract_frames(self, video_path=0):
        video = cv2.VideoCapture(video_path)
        # video.set(3, 800)  # Width
        # video.set(4, 800)  # Height
        

        # loop_thread = threading.Thread(target=self.loop_thread)
        # loop_thread.start()

        audio_thread = threading.Thread(target=self.audio_thread)
        audio_thread.start()

        tracking_thread = threading.Thread(target=self.tracking_thread)
        tracking_thread.start()

        # time_thread = threading.Thread(target=self.time_thread)
        # time_thread.start()

        while (self.input):
            try:
                success, frame = video.read()
                # print("length of current Frame -> ", len(self.current_frames))
                try:
                    img_h, img_w = frame.shape[:2]
                except:
                    continue
                with self.lock:
                    self.frame = frame

                    # try catch for audio
                    try:
                        self.audio_frame = self.audio_input.read() 
                    except:
                        self.audio_frame = None
                    
                    if (len(self.current_frames) < self.FRAME and self.captcha_running == False):  
                        self.current_frames.append(frame)
                        
                    else:
                        pass 
                
                if(self.show_text):
                    cv2.putText(frame, f"show {self.required_gesture} fingers", (200, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 4)
                    cv2.putText(frame, self.show_text, (200, 200), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 4)
                    cv2.putText(frame, f"(remaining time ->  {round(self.remainingTime, 2)})", (200, 300), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 4)
                    
                if(self.check_for_loop):
                    cv2.putText(frame, f"checking for loop ({self.loop_check_count})", (10, 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 225), 1)

                if(self.captcha_running):
                    cv2.putText(frame, "captcha checking", (10, 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 225), 1)
                    if(self.is_loop):
                        cv2.putText(frame, "loop detected", (700, 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 225), 1)
                    else:
                        cv2.putText(frame, "regular check", (700, 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 225), 1)

                if(self.check_for_loop):
                    cv2.putText(frame, f"{self.audio_alert} ({self.volume_threshold})", (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 225), 1)


                if (self._SHOW_EYE and self.mesh_points is not None):
                    
                    (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(self.mesh_points[lefteye_iris_center_indices_pos])
                    (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(self.mesh_points[righteye_iris_center_indices_pos])
                    center_left = np.array([l_cx, l_cy], dtype=np.int32)
                    center_right = np.array([r_cx, r_cy], dtype=np.int32)
                    cv2.circle(
                        frame, center_left, int(l_radius), (255, 0, 255), 2, cv2.LINE_AA
                    )  # Left iris
                    cv2.circle(
                        frame, center_right, int(r_radius), (255, 0, 255), 2, cv2.LINE_AA
                    )  # Right iris
                    cv2.circle(
                        frame, self.mesh_points[LEFT_EYE_INNER_CORNER][0], 3, (255, 255, 255), -1, cv2.LINE_AA
                    )  # Left eye right corner
                    cv2.circle(
                        frame, self.mesh_points[LEFT_EYE_OUTER_CORNER][0], 3, (0, 255, 255), -1, cv2.LINE_AA
                    )  # Left eye left corner
                    cv2.circle(
                        frame, self.mesh_points[RIGHT_EYE_INNER_CORNER][0], 3, (255, 255, 255), -1, cv2.LINE_AA
                    )  # Right eye right corner
                    cv2.circle(
                        frame, self.mesh_points[RIGHT_EYE_OUTER_CORNER][0], 3, (0, 255, 255), -1, cv2.LINE_AA
                    )  # Right eye left corner

                    cv2.putText(frame, f"Eye Direction: {self.eye_looks}", (30, 80), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

                if (self._SHOW_GAZE and self.nose_2D_point is not None and self.mesh_points is not None):
                    
                    p1 = self.nose_2D_point
                    p2 = (
                        int(self.nose_2D_point[0] +self.angle_y * 10),
                        int(self.nose_2D_point[1] -self.angle_x * 10),
                    )

                    cv2.line(frame, p1, p2, (255, 0, 255), 3)

                    for point in self.mesh_points[head_indices_pos]:
                        cv2.circle(frame, tuple(point), 1, (0, 255, 0), -1)

                    cv2.putText(frame, f"Face Direction: {self.face_looks}", (30, 80), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)


                if (self._SHOW_LIPS and self.landmarks is not None):
                    
                    upper_lip_pts = np.array([(int(self.landmarks[i].x * img_w), int(self.landmarks[i].y * img_h)) for i in UPPER_LIP])
                    lower_lip_pts = np.array([(int(self.landmarks[i].x * img_w), int(self.landmarks[i].y * img_h)) for i in LOWER_LIP])
                    
                    #upper lip line
                    for i in range(len(upper_lip_pts) - 1):
                        cv2.line(frame, tuple(upper_lip_pts[i]), tuple(upper_lip_pts[i+1]), (0, 255, 0), 2)
                    cv2.line(frame, tuple(upper_lip_pts[-1]), tuple(upper_lip_pts[0]), (0, 255, 0), 2)
                    
                    #lower lip line
                    for i in range(len(lower_lip_pts) - 1):
                        cv2.line(frame, tuple(lower_lip_pts[i]), tuple(lower_lip_pts[i+1]), (0, 255, 0), 2)
                    cv2.line(frame, tuple(lower_lip_pts[-1]), tuple(lower_lip_pts[0]), (0, 255, 0), 2)

                    # display lip detection
                    cv2.putText(frame, f"Lip Movement Count: {self.movement_count}", (30, 80), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)


                
                cv2.namedWindow("footage", cv2.WINDOW_NORMAL)
                cv2.imshow("footage", frame)
                # if(frame is not None):
                #     self.frame_placeholder.image(frame,channels="RGB")

                key = cv2.waitKey(1)
                with self.lock:
                    if (key == ord(' ')  and self.captcha_running == False):
                        self._SHOW_EYE = False
                        self._SHOW_GAZE = False
                        self._SHOW_LIPS = False
                        self.start_captcha_thread()
                    
                    if key == ord('f'):
                        
                        self._SHOW_EYE = False
                        self._SHOW_GAZE = True
                        self._SHOW_LIPS = False
                
                    if key == ord('e'):
                        
                        self._SHOW_EYE = True
                        self._SHOW_GAZE = False
                        self._SHOW_LIPS = False

                    if key == ord('l'):
                        
                        self._SHOW_EYE = False
                        self._SHOW_GAZE = False
                        self._SHOW_LIPS = True

                if key == ord('q'):
                    break
            except Exception as e:
                print("Error handled in main thread-> ", e)
            
        video.release()
        cv2.destroyAllWindows()
    
    def audio_thread(self):
        
        temp_count = 0
        temp = 0
        flag = 1
        avg = 0
        self.audio_input.start()
        while(self.check_audio and self.audio_frame is not None):

            # if(temp_count > self.AUDIO_FRAME):
            #     self.volume_threshold = avg / self.AUDIO_FRAME
            try:
               
                with self.lock:
                    volume = self.calculate_rms(self.audio_frame)
                    self.current_audio = volume
                    avg += volume
                    # print(round(volume, 2), temp_count, self.AUDIO_FRAME)
                
                    if volume > self.volume_threshold:
                        self.audio_alert = "Noise in close proximity detected"
                      
                        if(flag == 1):
                            temp_count += 1
                        else:
                            temp_count = 0
                            flag = 1
                    else:
                        self.audio_alert = "No noise detected"
                        if(flag == 0):
                            temp_count += 1
                        else:
                            temp_count = 0
                            flag = 0
            except Exception as e:
                print("Error handled in audio thread-> ", e)
        pass
    def tracking_thread(self):
        #self.frame
        while(self.tracking):
            if(self.frame is not None):
                temp = face_tracking(self.frame)
                if(temp is not None):
                    with self.lock:
                        self.prev_lip = self.current_lip
                        self.mesh_points,self.landmarks,self.face_looks,self.eye_looks,self.movement_detected, self.strike_count, self.nose_2D_point, self.angle_x, self.angle_y, self.current_lip = temp
                        print(self.movement_count, self.movement_detected, self.current_lip - self.prev_lip)
                        if(self.current_lip - self.prev_lip > movement_threshold):
                            
                            self.movement_count += 1
            pass
    
    # def time_thread(self):

    #     while(True):
    #         with self.lock:
    #             self.time += 1
    #             time.sleep(1)
    
    # def plot(self):
    #     pass
        

    # def animate(self, i):
    #     # Update the plot with current audio data
    #     with self.lock:
    #         self.audio_xs.append(self.time)
    #         self.audio_ys.append(self.current_audio)

    #         # Limit the plot to the last `MAX_POINTS` data points
    #         self.audio_xs = self.audio_xs[-self.MAX_POINTS:]
    #         self.audio_ys = self.audio_ys[-self.MAX_POINTS:]

    #     # Clear and redraw the plot
    #     self.ax1.clear()
    #     self.ax1.plot(self.audio_xs, self.audio_ys)

        




object = OEP()
threading.Thread(target=object.extract_frames, daemon=True).start()

filename = r'./loop_detect/videos/orig_vid_3.avi'

object.extract_frames()

