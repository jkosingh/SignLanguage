import numpy as np
import cv2
from matplotlib import pyplot as plt
import glob
import os
from skimage.measure import compare_ssim as ssim
import skimage.transform
from skimage import io, color
#https://www.pyimagesearch.com/2014/09/15/python-compare-two-images/
dir = "D:/Summer2018/ENSC482/JustinCode/Sign-Language-Translator/SignRecognition/TEST_VIDS/Learn ASL Alphabet Video.mp4"
vidcap = cv2.VideoCapture(dir)
success,image = vidcap.read()
count = 0  # initialize counter
while success:
  vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))    # Change multiple on count for number of frames taken
  cv2.imwrite("D:/Summer2018/ENSC482/JustinCode/Sign-Language-Translator/SignRecognition/DATA/EXPORT/frame%d.jpg" % count, image)     # save frame as JPEG file
  success,image = vidcap.read()
  print('Read a new frame: ', success, ' ', count)
  count += 1
  if count > 100: #Here because idk why ASL video never ends...
    success = False
  comp = False

if success == False:
  print('DONE SPLICING VIDEO INTO FRAMES')
  Y_data = [] # Frames from video
  export = glob.glob("D:/Summer2018/ENSC482/JustinCode/Sign-Language-Translator/SignRecognition/DATA/EXPORT/*.jpg")
  for pegs in export:
    frames = cv2.imread(pegs)
    Y_data.append(frames) #array to hold exported pictures

#cap = cv2.VideoCapture(0)
#i = 0
#while(True):
    # Capture frame-by-frame
    #ret, frame = cap.read()

    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   # i = i+1
    # Display the resulting frame
    #cv2.imshow('frame',frame)
    #naming = 'frame%s.png' %(i)
    #plt._imsave(naming, frame)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
      #  break
# When everything done, release the capture
#cap.release()
#cv2.destroyAllWindows()

#filenames = [img for img in glob.glob("D:/Summer2018/ENSC482/Segmentation/VideoFrame/*.png")]

#filenames.sort() # ADD THIS LINE

#images = []
#j=0
#for img in filenames:
    #j = j + 1
    #n = cv2.imread(img)
    #images.append(n)

    #gray = cv2.cvtColor(n, cv2.COLOR_BGR2GRAY)
   # ret2, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # noise removal
    #kernel = np.ones((3, 3), np.uint8)
    #opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # sure background area
    #sure_bg = cv2.dilate(opening, kernel, iterations=3)
    # Finding sure foreground area
    #dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    #ret3, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    # Finding unknown region
   # sure_fg = np.uint8(sure_fg)
    #unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    #ret4, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    #markers = markers + 1

    # Now, mark the region of unknown with zero
    #markers[unknown == 255] = 0
    #plt._imsave('D:/Summer2018/ENSC482/Segmentation/VideoFrame/export/new%s.png'%j, markers)

    #os.remove('frame%d.png'%j)



def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def compare_images(imageA, imageB, title):
    # compute the mean squared error and structural similarity
    # index for the images
    m = mse(imageA, imageB)
    s = ssim(imageA, imageB, multichannel=True)
    if(m < 20000 and s> 0.03):
        # setup the figure
        print("found a match !!!!!!!!!!")
        fig = plt.figure(title)
        plt.suptitle("MSE: %.2f, SSIM: %.10f" % (m, s))

        # show first image
        ax = fig.add_subplot(1, 2, 1)
        plt.imshow(imageA, cmap=plt.cm.gray)
        plt.axis("off")

        # show the second image
        ax = fig.add_subplot(1, 2, 2)
        plt.imshow(imageB, cmap=plt.cm.gray)
        plt.axis("off")

        # show the images
        plt.show()
    else:
        print("Not a match")
filenames2 = [img for img in glob.glob("D:/Summer2018/ENSC482/Segmentation/VideoFrame/data/*.jpg")]

#filenames2.sort() # ADD THIS LINE

filenames3 = [img for img in glob.glob("D:/Summer2018/ENSC482/Segmentation/VideoFrame/export/*.png")]

#filenames3.sort()

images2 = []
j2=0
for img in Y_data:
    # load the images -- the original, the original + contrast,
    #original = cv2.imread(img)
    original = img[150:550,200:1100]
    original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    #####################################################


    #################################################

    for img2 in filenames2:
        contrast = cv2.imread(img2)
        contrast = cv2.flip(contrast,1)

        # convert the images to grayscale

        contrast = cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY)

        original = skimage.transform.resize(original, contrast.shape, mode='constant')

        # initialize the figure
        #fig = plt.figure("Images")
        images = ("Original", original), ("Contrast", contrast)

        # loop over the images
        #for (i, (name, image)) in enumerate(images):
            # show the image
            #ax = fig.add_subplot(1, 3, i + 1)
            #ax.set_title(name)
            #plt.imshow(image, cmap=plt.cm.gray)
            #plt.axis("off")

        # show the figure
            #plt.show()

        # compare the images
        #compare_images(original, original, "Original vs. Original")
        compare_images(original, contrast, "Original vs. Contrast")

