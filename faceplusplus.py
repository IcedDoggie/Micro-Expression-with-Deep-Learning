import requests
import json
import base64
import cv2

def face_landmarking(image):
	key = "6iJfwlLNpz_Yn8OU-knyRytWT8ic_5wB"   
	secret = "dPEUSxH_WfwBTlh5sXRD-e9mj_ltJoLj"
	http_url ="https://api-us.faceplusplus.com/facepp/v3/detect"

	#if using image from URL
	#url = "http://images2.chictopia.com/photos/Thefoxandfern/1491318141/gold-mini-dress-free-people-dress-black-ankle-boots-steve-madden-boots_400.jpg"
	#parameters = {"api_key": key, "api_secret": secret, "image_url": url}


	with open(image,"rb") as imageFile:
	    # print(type(imageFile))
	    image_str = base64.b64encode(imageFile.read())
	data = {"api_key":key,"api_secret":secret,"image_base64":image_str,"return_landmark":0,"return_attributes":"gender,age,smiling,headpose,facequality,blur,eyestatus,ethnicity"}
	response = requests.post(http_url, data=data)



	# print(response.status_code)

	# get response data as a python object.
	data = response.json()

	# print out some data info
	# print(data)
	# print(len(data['faces']))
	# print(data['faces'][0]['attributes']['age']['value'])
	# print(data'fa[ces'][0]['face_rectangle'])
	# item = data['faces'][0]['face_rectangle']
	# print(item)	
	rect = data['faces'][0]['face_rectangle']
	w, h, left, top = rect['width'], rect['height'], rect['left'], rect['top']

	# image = cv2.imread(image)
	# cropped_image = image[top:top+h, left:left+w]
	# cropped_image = cv2.resize(cropped_image, (280,340))


	return w, h, left, top

# rect = face_landmarking('006_05562.jpg')
# w, h, left, top = rect['width'], rect['height'], rect['left'], rect['top']
# image = cv2.imread('006_05562.jpg')
# cropped_image = image[top:top+h, left:left+w]
# cv2.imshow('cropped_image', cropped_image)
# cv2.waitKey(0)
