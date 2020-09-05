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

Choose a color range (threshold value) for the colored object detection.
For details, [Check this out](https://www.geeksforgeeks.org/python-thresholding-techniques-using-opencv-set-1-simple-thresholding/)
