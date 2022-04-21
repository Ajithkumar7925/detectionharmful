from flask import Flask,render_template
import numpy as np
import pyautogui
from PIL import Image, ImageGrab
import time
import cv2
import os
#import mail
import time
import schedule
import ctypes                                   
import smtplib 
from email.mime.multipart import MIMEMultipart 
from email.mime.text import MIMEText 
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email import encoders

app = Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
	return render_template('/index.html')

@app.route('/stop')
def stop():
	return render_template('index.html')

@app.route('/monitor')
def monitor():
	directory = os.getcwd()
	LABELS = ["Weapon"]
	weightsPath = directory + '/weight/yolov3_training_2000.weights'
	configPath = directory + '/cfg/yolov3_testing.cfg'
	net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
	print("loaded YOLO from disk...")
	threshold = 0.3
	confidence_value = 0.5
	def Mbox(title, text, style):
		'''
		Custom Mesgsage box
		'''
		return ctypes.windll.user32.MessageBoxW(0, text, title, style)

	def takeScreenShot():
		'''
		Ths function help to takescreen shrt. 
		'''
		image = pyautogui.screenshot()
		image = cv2.cvtColor(np.array(image),
				 cv2.COLOR_RGB2BGR)
		#cv2.imshow('sample', image)
		cv2.imwrite('takenimage.jpg', image)
		return image

	def report_send_mail(label, image_path):
		'''
		This function sends mail
		'''
		try:
			with open(image_path, 'rb') as f:
				img_data = f.read()
			fromaddr = "yolotechnology04324554@gmail.com"
			toaddr = "yolotechnology04324554@gmail.com"

			msg = MIMEMultipart() 
			msg['From'] = fromaddr 
			msg['To'] = toaddr 
			msg['Subject'] = "Alert"
			body = label
			msg.attach(MIMEText(body, 'plain'))  # attach plain text
			image = MIMEImage(img_data, name=os.path.basename(image_path))
			msg.attach(image) # attach image
			s = smtplib.SMTP('smtp.gmail.com', 587) 
			s.starttls() 
			s.login(fromaddr, "YOLOTECHNOLOGY04324554") 
			text = msg.as_string() 
			s.sendmail(fromaddr, toaddr, text) 
			s.quit()
			return render_template('/index.html',msg='Mail send successfully')
			# print("Mail send successfully")
		except:
			return render_template('/index.html',msg='Some thing went wrong')
			# print("Some thing went wrong")

	def detection():
		image = takeScreenShot()
		#image = cv2.imread('images.jpg')
		(H, W) = image.shape[:2]
		ln = net.getLayerNames()
		ln = net.getUnconnectedOutLayersNames()
		blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
			swapRB=True, crop=False)
		net.setInput(blob)
		start = time.time()
		layerOutputs = net.forward(ln)
		end = time.time()
		print("[INFO] YOLO took {:.6f} seconds".format(end - start))
		boxes = []
		confidences = []
		classIDs = []
		for output in layerOutputs:
			for detection in output:
				scores = detection[5:]
				classID = np.argmax(scores)
				confidence = scores[classID]

				if confidence > confidence_value:
					box = detection[0:4] * np.array([W, H, W, H])
					(centerX, centerY, width, height) = box.astype("int")
					x = int(centerX - (width / 2))
					y = int(centerY - (height / 2))
					boxes.append([x, y, int(width), int(height)])
					confidences.append(float(confidence))
					classIDs.append(classID)
		idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_value,
			threshold)
		if len(idxs) > 0:
			for i in idxs.flatten():
				(x, y) = (boxes[i][0], boxes[i][1])
				(w, h) = (boxes[i][2], boxes[i][3])
				print(LABELS[classIDs[i]])
				label = LABELS[classIDs[i]]
				image_path = 'takenimage.jpg'
				report_send_mail(label, image_path)
				Mbox('Warnings!!! ', label, 2)

	schedule.every(30).seconds.do(detection)
	while True:
		schedule.run_pending()
		time.sleep(1)

if __name__ == '__main__':
	app.run(debug=True)
