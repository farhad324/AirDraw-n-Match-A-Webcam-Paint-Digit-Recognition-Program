# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 17:13:39 2020

@author: farhad324
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import easygui

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

trials=0
points=0

while True:
    n=random. randint(0,9) 
    num= 'Draw: '+str(n)

    cap = cv2.VideoCapture(0)

    # Select object color range 
    low_green = np.array([20, 85, 70])
    high_green = np.array([102, 255, 255])

   #Create an image for the canvas
    img=np.zeros((512,512,3),np.int8)

   #Choose pen color (I chose white)
    pen_color=(256,256,256)
    
    trials+=1
    while True:
                
        cv2.imshow('Canvas',img)
        ret, frame = cap.read()
        frame = cv2.flip(frame,1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, low_green , high_green)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.putText(frame, num, (20, 80), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255,10,0), 1, cv2.LINE_AA)
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
        cv2.imshow('Detection Panel', frame)
        if cv2.waitKey(10) == ord('q'):
            break
    cv2.imwrite("Output Image.jpg",img)
    cap.release()
    cv2.destroyAllWindows()
    
    ## Changing image resolution and color ##
    
    image = cv2.imread("Output Image.jpg")
    image = image[25:420, 25:450]
    image = ~image
    cv2.imwrite("img_inv.jpg",image)
    img_array = cv2.imread("img_inv.jpg", cv2.IMREAD_GRAYSCALE)
    img_array = cv2.bitwise_not(img_array)
    img_size = 28
    new_array = cv2.resize(img_array, (img_size,img_size))
    plt.imshow(new_array, cmap = plt.cm.binary)
    plt.title("Test Image")
    plt.show()
    user_test = tf.keras.utils.normalize(new_array, axis = 1)
    predicted = model.predict([[user_test]])
    a = predicted[0][0]
    for i in range(0,10):
        b = predicted[0][i]
        print("Probability Distribution for",i,b)

    print("The Predicted Value is",np.argmax(predicted[0]))

    draw=str(n)
    prediction=str(np.argmax(predicted[0]))
    
    if n==np.argmax(predicted[0]):
        points+=1
        message="Given number was: "+draw+"\nPredicted number is: "+prediction+"\n\nYour number matched!!!\n\nPoints: "+str(points)+"\n\nTrials: "+str(trials)       
        permission = easygui.buttonbox(message, 'Result', choices=(['Close', 'Continue']))
        
    else:
        message="Given number was: "+draw+"\nPredicted number is: "+prediction+"\n\nYour number didn't match!!!\n\nPoints: "+str(points)+"\n\nTrials: "+str(trials)
        permission = easygui.buttonbox(message, 'Result', choices=(['Close', 'Continue']))
            
    if permission!='Continue':
        break
