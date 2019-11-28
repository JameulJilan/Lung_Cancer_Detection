import ImagePreProcessing
import imageio
import cv2
import math
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


def calcutateHistogram(img):
    xValue = [0 for x in range(img.shape[0])]
    for j in range(img.shape[1]):
        for i in range(img.shape[0]):
            if img[j][i] == 255:
                xValue[i] += 1
    plt.plot(xValue)
    plt.title("Histogram Of Binary Image")
    plt.xlabel("X Cordinate Value")
    plt.ylabel("Number Of White Pixel")
    plt.show()
    image = img.copy()
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    cv2.line(image, (image.shape[0]//2, 0),
             (image.shape[0]//2, image.shape[1]), (0, 0, 255), 8)
    ImagePreProcessing.display(image, "Left And Right Lung Area", .25)


def returnPixelList(pixel):
    pixel = pixel.replace("[", "")
    pixel = pixel.split("]")
    del pixel[(len(pixel)-1)]
    pixel1 = []
    for i in range(len(pixel)-1):
        string = pixel[i]
        string = string.split(", ")
        pixel1.append([int(string[0]), int(string[1])])
    pixel1.append([pixel[(len(pixel)-1)].split(", ")[0],
                   pixel[(len(pixel)-1)].split(", ")[1]])
    return pixel1


def calculateLeftLungArea(image):
    img = image.copy()
    img = img[0:img.shape[1], img.shape[0]//2:img.shape[0]]
    resized = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_AREA)
    return resized


def calculateRightLungArea(image):
    img = image.copy()
    img = img[0:img.shape[1], 0:img.shape[0]//2]

    resized = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_AREA)
    return resized


def ROIExtraction(image):
    print(os.listdir('../MedicalData/masks/left lung'))


def findBorder(img):
    image = img.copy()
    kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    erodiedImage = cv2.morphologyEx(image, cv2.MORPH_ERODE, kernal)
    pixel = []
    for i in range(erodiedImage.shape[0]):
        for j in range(erodiedImage.shape[1]):
            image[i, j] = image[i, j]-erodiedImage[i, j]
    return image


def getRightLungBorderPixel(image):
    pixelValueToReturn = []
    hitRetio = 0.0
    with open("./ImageData/RightLungBorder.txt", "r") as file:
        for line in file:
            hitScore = 0.0
            misScore = 0.0
            pixel = returnPixelList(line)
            for i in range(len(pixel)-1):
                if image[pixel[i][0]][pixel[i][1]] == 255:
                    hitScore += 1
                else:
                    misScore += 1
            if (hitScore/(hitScore+misScore)) > hitRetio:
                hitRetio = (hitScore/(hitScore+misScore))
                pixelValueToReturn = pixel.copy()
    return pixelValueToReturn


def getLeftLungBorderPixel(image):
    pixelValueToReturn = []
    hitRetio = 0.0
    with open("./ImageData/LeftLungBorder.txt", "r") as file:
        for line in file:
            hitScore = 0.0
            misScore = 0.0
            pixel = returnPixelList(line)
            for i in range(len(pixel)-1):
                if image[pixel[i][0]][pixel[i][1]] == 255:
                    hitScore += 1
                else:
                    misScore += 1
            if (hitScore/(hitScore+misScore)) > hitRetio:
                hitRetio = (hitScore/(hitScore+misScore))
                pixelValueToReturn = pixel.copy()
    return pixelValueToReturn


def calculateMeanStd(OrginalImage, ImageForSearch):
    count = 0
    sumValue = 0
    # OrginalImage = cv2.resize(
    #     OrginalImage, (1024, 1024), interpolation=cv2.INTER_AREA)

    for i in range(1024):
        for j in range(1024):
            if OrginalImage[j][i] != 255:
                sumValue += OrginalImage[j][i]
                count += 1

    avgValue = sumValue//count
    belowAvarageList = []
    belowAvgSum = 0
    stdValue = 0
    for i in range(1024):
        for j in range(1024):
            if OrginalImage[j][i] != 255:
                stdValue += pow((avgValue-OrginalImage[j][i]), 2)
                if OrginalImage[j][i] < avgValue:
                    belowAvarageList.append(OrginalImage[j][i])
                    belowAvgSum += OrginalImage[j][i]
    stdValue = math.sqrt(stdValue/count)
    belowAvgAvg = belowAvgSum/len(belowAvarageList)
    belowStd = 0
    for i in range(len(belowAvarageList)):
        belowStd += pow((belowAvgAvg-belowAvarageList[i]), 2)
    belowStd = math.sqrt(belowStd/len(belowAvarageList))
    targetValue = avgValue-stdValue-belowStd
    print([avgValue, stdValue, belowAvgAvg, belowStd, targetValue])
    for i in range(1024):
        for j in range(1024):
            if OrginalImage[j][i]>=targetValue-belowStd/2 and OrginalImage[j][i]<=targetValue+belowStd/2:
                ImageForSearch[j][i]=255
    return ImageForSearch


def getRegionOfInterest(OrginalImage):
    PreProcessedImage = ImagePreProcessing.imagePreProcessing(
        OrginalImage.copy())
    # LeftProcessedImage = calculateLeftLungArea(PreProcessedImage)
    # LeftBounbaryImage = findBorder(LeftProcessedImage)
    # LeftLungBorderPixel = getLeftLungBorderPixel(LeftBounbaryImage)
    ResizedPreProcessedImage = cv2.resize(PreProcessedImage, (1024, 1024),
                                          interpolation=cv2.INTER_AREA)
    OrginalImage = cv2.resize(OrginalImage, (1024, 1024),
                              interpolation=cv2.INTER_AREA)
    BorederOfResizedImage = findBorder(ResizedPreProcessedImage)
    kernel = np.ones((5, 5), np.uint8)
    BorederOfResizedImage = cv2.dilate(
        BorederOfResizedImage, kernel, iterations=1)
    LeftLungBorderPixel = getLeftLungBorderPixel(BorederOfResizedImage)
    RightLungBorderPixel = getRightLungBorderPixel(BorederOfResizedImage)
    # ImagePreProcessing.display(BorederOfResizedImage, "Border", .5)
    ComparingImage = ResizedPreProcessedImage.copy()
    ResizedPreProcessedImage = cv2.cvtColor(
        ResizedPreProcessedImage, cv2.COLOR_GRAY2RGB)
    OrginalImage = cv2.cvtColor(OrginalImage, cv2.COLOR_BGR2GRAY)
    for i in range(len(LeftLungBorderPixel)-1):
        cv2.circle(
            ResizedPreProcessedImage, (LeftLungBorderPixel[i][1], LeftLungBorderPixel[i][0]), 1, (0, 0, 255), 2)
    for i in range(len(RightLungBorderPixel)-1):
        cv2.circle(
            ResizedPreProcessedImage, (RightLungBorderPixel[i][1], RightLungBorderPixel[i][0]), 1, (0, 0, 255), 2)
    # ImagePreProcessing.display(ResizedPreProcessedImage, "BorderImage", .5)

    ImageForSearch = [[255 for x in range(1024)] for y in range(1024)]
    NewImageForSearch = [[255 for x in range(1024)] for y in range(1024)]

    LeftLung = cv2.imread(LeftLungBorderPixel[len(
        LeftLungBorderPixel)-1][0].replace("'", ""), cv2.IMREAD_GRAYSCALE)
    RightLung = cv2.imread(RightLungBorderPixel[len(
        RightLungBorderPixel)-1][0].replace("'", ""), cv2.IMREAD_GRAYSCALE)
    for i in range(ComparingImage.shape[0]):
        for j in range(ComparingImage.shape[1]):
            if ComparingImage[i][j] == 0 and LeftLung[i][j] == 255:
                ImageForSearch[i][j] = 0
                NewImageForSearch[i][j] = NewImageForSearch[i][j] - \
                    OrginalImage[i][j]
            if ComparingImage[i][j] == 0 and RightLung[i][j] == 255:
                ImageForSearch[i][j] = 0
                NewImageForSearch[i][j] = NewImageForSearch[i][j] - \
                    OrginalImage[i][j]

    ImageForSearch=calculateMeanStd(NewImageForSearch, ImageForSearch)
    # cv2.imwrite("Template.bmp",ImageForSearch)
    ImageForSearch = np.array(ImageForSearch)
    # img = Image.fromarray(ImageForSearch)
    cv2.imwrite("Template.bmp", ImageForSearch)
    # plt.imshow(img, 'gray')
    # plt.show()
    # img.save('test.png')
    # calculateMeanStd(NewImageForSearch, ImageForSearch)
    # return ImageForSearch
