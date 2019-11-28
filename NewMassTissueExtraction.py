import NewRegionOfInterestExtraction
import cv2
import numpy as np
import os


def findMassTissue(OrginalImage, name):
    OrginalImage = cv2.resize(
        OrginalImage, (1024, 1024), interpolation=cv2.INTER_AREA)
    img_rgb = cv2.imread(
        './TH7.5/Template.bmp', cv2.IMREAD_COLOR)
    img_rgb = np.array(img_rgb)
    resizeList = [8, 16, 32, 64, 128]

    # Convert it to grayscale
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    imageGray = img_rgb.copy()
    TemplateList = os.listdir("../TumourRegion")
    Flag = False
    for i in range(len(TemplateList)):
        # Read the template
        templateTemp = cv2.imread('../TumourRegion/'+TemplateList[i], 0)

        template = templateTemp

        w, h = template.shape[::-1]

        # Perform match operations.
        res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)

        # Specify a threshold
        threshold = 0.80

        # Store the coordinates of matched area in a numpy array
        loc = np.where(res >= threshold)

        if len(loc[0]) > 0 and len(loc[1]) > 0:
            Flag = True
            # print(len(loc))

        # Draw a rectangle around the matched region.
        for pt in zip(*loc[::-1]):
            cv2.rectangle(OrginalImage, pt,
                          (pt[0] + w, pt[1] + h), (0, 0, 255), 1)
            cv2.rectangle(imageGray, pt,
                          (pt[0] + w, pt[1] + h), (0, 0, 255), 1)
        for j in range(len(resizeList)):
            template = cv2.resize(
                templateTemp, (resizeList[j], resizeList[j]), interpolation=cv2.INTER_AREA)

            # Store width and heigth of template in w and h

            w, h = template.shape[::-1]

            # Perform match operations.
            res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)

            # Specify a threshold
            threshold = 0.80

            # Store the coordinates of matched area in a numpy array
            loc = np.where(res >= threshold)

            if len(loc[0]) > 0 and len(loc[1]) > 0:
                Flag = True
                print(TemplateList[i])

            # Draw a rectangle around the matched region.
            for pt in zip(*loc[::-1]):
                cv2.rectangle(OrginalImage, pt,
                              (pt[0] + w, pt[1] + h), (0, 0, 255), 1)
                cv2.rectangle(imageGray, pt,
                              (pt[0] + w, pt[1] + h), (0, 0, 255), 1)

        # loc = None
    NewRegionOfInterestExtraction.ImagePreProcessing.display(
        OrginalImage, "Detected ( "+name+" )", .50)
    # if Flag == True:
    #     # RegionOfIntrestExtartion.ImagePreProcessing.display(
    #     #     OrginalImage, "Detected ( "+name+" )", .50)
    #     print("not found "+name)
    # else:
    #     # RegionOfIntrestExtartion.ImagePreProcessing.display(
    #     #     OrginalImage, "No Region Found ( "+name+" )", .50)
    # cv2.imwrite("./TH7.5/"+"Detected(Orginal) " + name, OrginalImage)
    # cv2.imwrite("./TH7.5/"+"Detected(Binary) "+name, imageGray)


# ImageList = os.listdir("../MedicalData/NoduleBmp154Images/")
# for i in range(len(ImageList)):
#     print(str(ImageList[i]))
#     img = cv2.imread(
#         "../MedicalData/NoduleBmp154Images/"+ImageList[i], cv2.IMREAD_COLOR)
#     RegionOfIntrestExtartion.getRegionOfInterest(img)
#     findMassTissue(img, ImageList[i])
#     print(str("\n"))
img = cv2.imread(
    "../MedicalData/NoduleBmp154Images/JPCLN007.bmp", cv2.IMREAD_COLOR)
NewRegionOfInterestExtraction.getRegionOfInterest(img)
# findMassTissue(img, "JPCLN002.bmp")
# cv2.waitKey(0)
