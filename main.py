import cv2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import imutils
import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt


path = os.listdir("C://Users/nh tech/Documents/Data/Training")
classes={'no_tumor':0,'pituitary_tumor':1}

X = []
Y = []
print(classes)
def modify(image):

        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        im_blur=cv2.GaussianBlur(image, (5,5),0)
        #im_blur=cv2.Canny(im_blur, 120,170)
        im_threshold=cv2.threshold(im_blur,45,255,cv2.THRESH_BINARY)[1]
        im_threshold=cv2.erode(im_threshold, None , iterations=2)
        im_thresholde = cv2.dilate(im_threshold, None, iterations=2)
        #contour = cv2.findContours(im_thresholde.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        #contour = imutils.grab_contours(contour)
        #c = max (contour, key=cv2.contourArea)
        #extreme_pnts_left = tuple(c[c[:, :,0].argmin()][0])
        #extreme_pnts_right = tuple(c[c[:, :, 0].argmax()][0])
        #extreme_pnts_top = tuple(c[c[:, :, 1].argmin()][0])
        #extreme_pnts_bot = tuple(c[c[:, :, 1].argmax()][0])
        #new_imag = im[extreme_pnts_top[1]:extreme_pnts_bot[1], extreme_pnts_left[0]:extreme_pnts_right[0]]
        img=cv2.resize(im_thresholde,(200,200))
        return img




for cls in classes:
    print(cls)

    pth = "C://Users/nh tech/Documents/Data/Training/"+cls
    #print(pth)
    for j in os.listdir(pth):
        pth = "C://Users/nh tech/Documents/Data/Training/" + cls
        pth=pth+'/'+j
        print(pth)
        im=cv2.imread(pth)
        #print(pth)

        img=modify(im)


        X.append(img)
        Y.append(classes[cls])


print('process done!')

Y=np.array(Y)
X=np.array(X)


print(pd.Series(Y).value_counts())

print(X.shape)
#visualize data

plt.imshow(X[3], cmap ='gray')
#RESHAING THE SIZE OF THE array X to two dimensional
X_updated=X.reshape(len(X),-1)

print(X_updated.shape)

#Splitting the data

x_train,x_test,y_train,y_test= train_test_split(X_updated, Y, random_state=10 , test_size=.20)

print(x_train.shape)
print(x_test.shape)

#Rescaling features in order to get values between 0 and 1, we devide our data over 255 as the maximum pixel value is 255
print(x_train.max(), x_train.min())
print(x_test.max(),x_test.min())
x_train=x_train/255
x_test=x_test/255

print(x_train.max() , x_train.min())

print(x_test.max() , x_test.min())

#training the model

import warnings

warnings.filterwarnings('ignore')#ignoring weak warnings that make me feel unconfortable
lg = LogisticRegression(C=0.01)
lg.fit(x_train,y_train)

print(lg)
sv=SVC(kernel='rbf',gamma='scale')
sv.fit(x_train,y_train)
print(sv)
#Evaluation of the model
print("Regressoin: \nTraining Score :", lg.score(x_train,y_train))
print("Testing Score: ",lg.score(x_test,y_test))
print("Support Vector: \nTraining Score:",sv.score(x_train, y_train))
print("Testing Score:",sv.score(x_test, y_test))

#Making predictions
pred = sv.predict(x_test)
print(np.where(y_test!=pred))
print(pred[1],pred[2])
print(y_test[1],y_test[2])
#Testing the model
dec = {0: 'No tumor', 1: 'Positive Tumor'}


plt.figure(figsize=(12, 8))

p=os.listdir("C://Users/nh tech/Documents/Data/Testing")

c = 1
#testing the model with the no tumor data set
for i in os.listdir("C://Users/nh tech/Documents/Data/Testing/no_tumor/")[:9]:

    plt.subplot(3, 3, c)

    img = cv2.imread("C://Users/nh tech/Documents/Data/Testing/no_tumor/" + i , 0)
    im_blur = cv2.GaussianBlur(img, (5, 5), 0)
    # im_blur=cv2.Canny(im_blur, 120,170)
    im_threshold = cv2.threshold(im_blur, 45, 255, cv2.THRESH_BINARY)[1]
    im_threshold = cv2.erode(im_threshold, None, iterations=2)
    img1 = cv2.dilate(im_threshold, None, iterations=2)


    imgR = cv2.resize(img1 , (200,200))

    imgR = imgR.reshape(1, -1) / 255

    p = sv.predict(imgR)

    plt.title(dec[p[0]])

    plt.imshow(img, cmap='gray')

    plt.show()

    c+=1

#WE see that the model succeeded in testing how good it is working for the no tumor tsting set
#Now w test our model on the positive tumor data set
plt.figure(figsize=(12,8))
k = 1

for i in os.listdir("C://Users/nh tech/Documents/Data/Testing/pituitary_tumor/")[:16]:

    plt.subplot(4,4, k)

    img = cv2.imread("C://Users/nh tech/Documents/Data/Testing/pituitary_tumor/" + i , 0)
    im_blur = cv2.GaussianBlur(img, (5, 5), 0)
    # im_blur=cv2.Canny(im_blur, 120,170)
    im_threshold = cv2.threshold(im_blur, 45, 255, cv2.THRESH_BINARY)[1]
    im_threshold = cv2.erode(im_threshold, None, iterations=2)
    img1 = cv2.dilate(im_threshold, None, iterations=2)

    imgR = cv2.resize(img1 , (200,200))

    imgR = imgR.reshape(1, - 1) / 255

    p = sv.predict(imgR)

    plt.title(dec[p[0]])

    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.show()

    k+=1



