#pylint:disable=no-member

import cv2 as cv
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

img = cv.imread(os.path.join(script_dir, '../Resources/Photos/trixia ni.jpg'))
cv.imshow('1 person', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray People', gray)

haar_cascade = cv.CascadeClassifier(os.path.join(script_dir, 'haar_face.xml'))

faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)

print(f'Number of faces found = {len(faces_rect)}')

for (x,y,w,h) in faces_rect:
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

cv.imshow('Detected Faces', img)



cv.waitKey(0)