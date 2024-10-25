import pylab
import imageio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import time 
import pdb
import collections
import pandas as pd
import imageio
from PIL import Image
import cv2
import threading
from cvzone.HandTrackingModule import HandDetector





class VideoLoopFinder:

	def __init__(self, RES = 32):
		self.RES = 32
		self.OPT_VALUE = 1
		self.MAX_LIMIT = 1000
		self.THRESHOLD = 10 // self.OPT_VALUE
		self.seen_frames = {}
		self.duplicate_frames = {}
		self.current_frames = []

		self.finish = True
		self.lock = threading.Lock()


		


	
	def ahash(self, frame, res = 64):
		i = Image.fromarray(frame)
		i = i.resize((res,res), Image.ANTIALIAS).convert('L')
		pixels = list(i.getdata())
		avg = sum(pixels)/len(pixels)
		bits = "".join(map(lambda pixel: '1' if pixel < avg else '0', pixels))
		hexadecimal = int(bits, 2).__format__('016x').upper()
		return hexadecimal

	def find_duplicates(self, vid, res=32):
	
		all_frames = len(vid)
		print(all_frames)

		for x in range(0, all_frames//self.OPT_VALUE, self.OPT_VALUE): 
			
			frame = vid[x]

			# if x % 1000 == 0:
			# 	print("frame count: ",x,"\t",round(x*1.0/all_frames,3)*100,'%')

			
			hashed = self.ahash(frame, res)
			
			if self.seen_frames.get( hashed, None):
				
				self.duplicate_frames[hashed].append(x)
			else:
				
				self.seen_frames[hashed] = x
				self.duplicate_frames[hashed] = [x]
		# for i in self.duplicate_frames:
		# 	print(i)
		duplicates = [abs(self.duplicate_frames[x][0] - self.duplicate_frames[x][-1]) for x in self.duplicate_frames if len(self.duplicate_frames[x]) > 1]
		print(duplicates)
		return duplicates

	def get_vaild_duplicates(self, duplicate_frames):
		s = 0
		loop_frame_count = 0
		for x in duplicate_frames:
			if(x > self.THRESHOLD):
				loop_frame_count += 1
			
		
		print("duplicate frame count ->  ", loop_frame_count)

		if(len(self.seen_frames) > self.MAX_LIMIT):
			self.seen_frames.clear()
			self.duplicate_frames.clear()
		if(loop_frame_count > self.THRESHOLD):
			print("LOOP DETECTED")
			self.seen_frames.clear()
			self.duplicate_frames.clear()
			return [True, loop_frame_count]
		else:
			print("NO LOOP")
		return [False, loop_frame_count]
		
	# def extract_frames(self, video_path = 0):
	# 	video = cv2.VideoCapture(video_path)
	# 	frames = []
	# 	INIT = 0
	# 	success, frame = video.read()

	# 	while success:
	# 		cv2.namedWindow("footage", cv2.WINDOW_NORMAL)
	# 		cv2.imshow("footage", frame)
	# 		self.current_frames.append(frame)
			

			
	# 		if cv2.waitKey(1) & 0xFF == ord('q'):
	# 			break
			
	# 		success, frame = video.read()
		
	# 	video.release()
	# 	cv2.destroyAllWindows()
	# 	self.finish = False
	
	# def loop_thread(self):

	# 	INIT = 0
	# 	while(self.finish):

	# 		if len(self.current_frames) > INIT + self.FRAME:
	# 			duplicate_frames = self.find_duplicates(self.current_frames)
	# 			isLooped = self.get_vaild_duplicates(duplicate_frames)

	# 			INIT += self.FRAME
	# 			self.current_frames = []
	# 		else:
	# 			pass

		
	

# loop_finder = VideoLoopFinder()
# filename = r'./videos/looped_vid_1.mp4'
# input_thread = threading.Thread(target = loop_finder.extract_frames, args =())
# loop_thread = threading.Thread(target = loop_finder.loop_thread, args = ())

# input_thread.start()
# loop_thread.start()
	

print("detect imported")