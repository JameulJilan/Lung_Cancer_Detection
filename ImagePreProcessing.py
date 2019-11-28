import cv2
import numpy as np
import os


def display(image, name, value):
    resized = cv2.resize(image, (0, 0), fx=value, fy=value)
    cv2.imshow(name, resized)


def contrastStretching(img):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image = cv2.equalizeHist(image)
    return image


def createBinaryImage(img):
    ret, binaryImage = cv2.threshold(
        img, 0, 255, cv2.THRESH_OTSU)
    return binaryImage


def morphologicalOperation(img):
    image = img.copy()
    kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    image = cv2.erode(image, kernal)
    return image


def findBorder(img):
    image = img.copy()
    kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    erodiedImage = cv2.morphologyEx(image, cv2.MORPH_ERODE, kernal)
    # pixel=[]
    # returnImage=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    for i in range(erodiedImage.shape[0]):
        # count=0
        for j in range(erodiedImage.shape[1]):
            # if image[i,j]==255 and erodiedImage[i,j]==255 and count==0:
            #     pixel.append([i,j])
            #     count+=1
            image[i, j] = image[i, j]-erodiedImage[i, j]
    # image=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    # print(pixel)
    # for i in range(len(pixel)):
    #     cv2.circle(image,(pixel[i][0],pixel[i][1]), 1, (0,255,0), -1)
    # image=cv2.morphologyEx(image,cv2.MORPH_DILATE,kernal)
    # kernel = np.ones((5,5),np.uint8)
    # image=cv2.dilate(image,kernal,iterations = 1)
    return image


def imagePreProcessing(img):
    # display(img, "Input Image", .25)
    image = contrastStretching(img)
    # display(image, "After ContrastStreatching", .25)
    image = createBinaryImage(image)
    # display(image, "Binary Image", .25)
    image = morphologicalOperation(image)
    display(image, "After Closing", .25)
    border = findBorder(image)
    display(border, "BorderImage", .25)
    return image


# def ROIExtraction():
#     print(os.listdir('../MedicalData/masks/left lung'))
# img=cv2.imread("JPCLN001.bmp",cv2.IMREAD_COLOR)
# imagePreProcessing(img)
# cv2.waitKey(0)
# ImageList = os.listdir("../MedicalData/NoduleBmp154Images/")
# for i in range(len(ImageList)):
#     img = cv2.imread("../MedicalData/NoduleBmp154Images/" +
#                      ImageList[i], cv2.IMREAD_COLOR)
#     image = imagePreProcessing(img)
#     cv2.imwrite("../MedicalData/BinaryImage/"+ImageList[i], image)
# ROIExtraction()
