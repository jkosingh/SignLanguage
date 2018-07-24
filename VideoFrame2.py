from skimage.measure import compare_mse as mse
import argparse
import imutils
import cv2
# import cv
import os
import keyboard
import glob
import numpy as np
from skimage.measure import compare_ssim as ssim
from skimage.transform import resize
import math, operator
import matplotlib.pyplot as plt
import scipy
#from PIL import Image

dir = "\Users\justi\Downloads\Sign-Language-Translator-master\SignRecognition\TEST_VIDS\Learn ASL Alphabet Video.mp4"


# def mse(imageA, imageB):
#     # the 'Mean Squared Error' between the two images is the
#     # sum of the squared difference between the two images;
#     # NOTE: the two images must have the same dimension
#     err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
#     err /= float(imageA.shape[0] * imageA.shape[1])
#
#     # return the MSE, the lower the error, the more "similar"
#     # the two images are
#     return err

vidcap = cv2.VideoCapture(dir)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 330)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 440)
success, image = vidcap.read()
count = 0  # initialize counter
while success:
    vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 1000))  # Change multiple on count for number of frames taken
    cv2.imwrite("\Users\justi\Downloads\Sign-Language-Translator-master\SignRecognition\DATA\EXPORT\ frame%d.jpg" % count,
                image)  # save frame as JPEG file
    success, image = vidcap.read()
    print('Read a new frame: ', success, ' ', count)
    count += 1
    if count > 60:  # Here because idk why ASL video never ends...
        success = False
    comp = False

if success == False:
    print('DONE SPLICING VIDEO INTO FRAMES')
    Y_data = []  # Frames from video
    export = glob.glob("\Users\justi\Downloads\Sign-Language-Translator-master\SignRecognition\DATA\EXPORT\*.jpg")
    for pegs in export:
        frames = cv2.imread(pegs)
        Y_data.append(frames)  # array to hold exported pictures
    print('Frames to compare array size', len(Y_data))

    X_data = []  # Database sign langauge pictures
    #database = glob.glob("\Users\justi\Downloads\Sign-Language-Translator-master\SignRecognition\DATA\EXPORT\*.jpg")
    database = glob.glob("\Users\justi\Downloads\Sign-Language-Translator-master\SignRecognition\DATA\extractor\*.png")
    for myFile in database:
        image = cv2.imread(myFile)
        X_data.append(image)
    print('Database Array size:', len(X_data))
    print('DONE CREATING ARRAYS TO COMPARE')
    comp = True
else:
    print('I Suck')
    comp = False

if comp == True:
    print('NOW WILL COMPARE ARRAYS')
    tolerance = 0.0036
    tolerance2 = 19440
    counted = 0
    deleted = 0
    m = 0
    s = 0
    for i in range(0, len(Y_data)):
        WantToDelete = True
        for j in range(0, len(X_data)):
            Databased = cv2.cvtColor(X_data[j], cv2.COLOR_BGR2GRAY)
            Exported = cv2.cvtColor(Y_data[i], cv2.COLOR_BGR2GRAY)
            #Databased = X_data[j]
            #Exported = Y_data[i]
            Exported = resize(Exported, Databased.shape, mode='constant')
            #widthd, heightd = Databased.shape
            #widthe, heighte, = Exported.shape
            #print('Database shape', widthd, ' ', heightd)
            #print('Exported shape', widthe, ' ', heighte)
            #basewidth = width
            #wpercent = (basewidth/float(Exported.size[0]))
            #hsize = int((float(Exported.size[1])*float(wpercent)))
            #Exported = Exported.resize((widthd, heightd))
            #print('After Database shape', widthd, ' ', heightd)
            #print('After Exported shape', widthe, ' ', heighte)
            s = ssim(Exported, Databased)
            m = mse(Exported, Databased)
            if s > tolerance:
                counted += 1
                print("Files still in folder: ", counted, "SSIM Value: ", s, "MSE Value: ", m)
                WantToDelete = False
                break
        if WantToDelete:
            # delete picture from database
            print("Deleting: ", "{}".format(export[i]), "SSIM Value: ", s, "MSE Value: ", m)
            os.remove("{}".format(export[i]))
            deleted += 1
        # print("Files deleted: ", deleted)
    # print Databased.shape
    # print Exported.shape
    # s = ssim(original, original)
    # print("MSE: ", m)
    # print ("SSIM: ", s)
else:
    print('I still suck')
