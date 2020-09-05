# AirDraw-n-Match-A-Webcam-Paint-Digit-Recognition-Program

## How to use
![Detection Panel 1](images/detection%20panel%201.PNG)
![Detection Panel 2](images/detection%20panel%202.PNG)

In the first detection panel, you will see a text saying, "Draw: 9". The number "9" is randomly generated, and you have to draw "9" on the canvas.
As I've coded the 'object' as a green bottle-cap, there will be a text at the top of the webcam saying, "Object Found". There will be another text at the lower region of the detection panel where it shows the object's area. The larger the area is, the larger the pen size will be. To increase or decrease the pen-size you will have to bring the green object closer to the webcam (to increase) or take further from the webcam (to decrease). 
![Canvas-Nine](images/canvas.PNG)

Draw by moving the green object, it is quite hard at the beginning but gets easier eventually.
Press 'q' to stop the video and then the output image will be analyzed using the MNIST data set to predict the digit that you have drawn. 
If the prediction and the given number "Draw: 9" matches, you will get a point.

![Result](images/quit%20menu.PNG)

You can either continue or close the program. Clicking the Continue button will run the 'webcam paint' again and increase the trial number.
The result panel shows the given number (randomly generated) and the prediction of your drawing and your achieved points with the trials.

## Code Explanation
```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import easygui
```
Install all required modules and import them. 

### Digit Recognition
```python
mnist = tf.keras.datasets.mnist #28*28 image of handwritten of 0-9 
(x_train, y_train),(x_test,y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test,axis = 1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten()) 
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax)) # softmax for probability distribution
model.compile(optimizer = "adam" , loss = 'sparse_categorical_crossentropy' , metrics = ['accuracy'] )
model.fit(x_train,y_train,epochs = 3 )
predictions = model.predict([x_test])
```
We used the MNIST data set, it has 70,000 images of handwritten images in 28 by 28 pixels.

### Webcam Paint
```python
trials=0
points=0
```
Before we start our Webcam Paint portion, we have to create two variables 'trials' and 'points'.
```python
# random number generator
n=random. randint(0,9) 
num= 'Draw: '+str(n)

# capturing video
cap = cv2.VideoCapture(0) 

# choosing object color
low_green = np.array([20, 85, 70])
high_green = np.array([102, 255, 255])

# creating an image to draw 
img=np.zeros((512,512,3),np.int8)

# choosing the pen color
pen_color=(256,256,256)   

# 'trials' increases whenever it enters the outer loop
trials+=1
```

So, to show the "Draw: 9" on the detection panel we have to create a random number generator. Here, in random.randint, the arguments are set to 0 to 9 so that it can only generate integers from 0 to 9. The 'num' is a string data to use on the detection panel.
To capture a video, you need to create a VideoCapture object. Its argument can be either the device index or the name of a video file. Device index is just the number to specify which camera. Normally one camera will be connected (as in my case). So I simply pass 0 (or -1). You can select the second camera by passing 1 and so on. After that, you can capture frame-by-frame. But at the end, don’t forget to release the capture.
Choose a color range (threshold value) for the colored object detection.
For details, [Check this out](https://www.geeksforgeeks.org/python-thresholding-techniques-using-opencv-set-1-simple-thresholding/).

In the inner loop,
```python
cv2.imshow('Canvas',img)
ret, frame = cap.read()
frame = cv2.flip(frame,1)
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, low_green , high_green)
contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.putText(frame, num, (20, 80), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255,10,0), 1, cv2.LINE_AA)
```
Show the created image 'img' and name it 'Canvas'. 
cap.read() returns a bool (True/False). If frame is read correctly, it will be True. So you can check end of the video by checking this return value.
Mirror the image using cv2.flip(frame,1) to draw with ease. 
Convert color to hsv and then create a mask with the threshold value (low_green, high_green) to detect the object.
Use cv2.findContours to show the detected green object. Contours can be explained simply as a curve joining all the continuous points (along the boundary), having same color or intensity. The contours are a useful tool for shape analysis and object detection and recognition.
I used cv2.putText to show the 'Draw: 9' to see which number the user have to draw.

![Pen Size](images/pensize.PNG)

```python
for c in contours:
        area = cv2.contourArea(c)
        if area > 600:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255,10,0), 2)
            pen_size=2
            if  1500 < area <2500:
                pen_size=3
            elif 2500 < area<3500:
                pen_size=4
            elif 3500 < area <4500:
                 pen_size=5
            elif 4500 < area<5500:
                 pen_size=6
            elif 5500 < area<6500:
                 pen_size=7
            elif 6500 < area < 7500:
                 pen_size=8
            elif 7500 < area < 8500:
                 pen_size=9
            elif 8500 < area:
                 pen_size=10
            cv2.circle(img,(x,y),pen_size,pen_color,-1)
            cv2.circle(frame,(x,y),pen_size,pen_color,2)
            cv2.putText(frame, "Object Found", (250, 80), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255,10,0), 1, cv2.LINE_AA)
            cv2.putText(frame, str(area), (250, 400), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,10,0), 1, cv2.LINE_AA)          
```
For each countours, we will use the cv2.contourArea to get the area of the countor. If the area is greater than 60 it will create a bounding box. Moreover, the pensize depends on the area of the countour. 
After the last elif we draw circles on the 'Canvas' according to the value of 'x' and 'y'. We also add a circle on the detection panel which works as a nib.
At last, in the for loop we write some texts like "Object Found" and the area of the object.
```python
cv2.imshow('Detection Panel', frame)
if cv2.waitKey(10) == ord('q'):
    break  
```
To see the detection panel I used cv2.imshow('Detection Panel', frame).
To close the Webcam Paint program or stop the video capturing we have to press 'q', which breaks the whole loop.
```python
cv2.imwrite("Output Image.jpg",img)
cap.release()
cv2.destroyAllWindows()
```
To finish the Webcam Paint section we'll realease the capture. We can save the output image using cv2.imwrite.

### Merging Webcam Paint & Digit Recognition (predicting the digit)
