"""student name:Hrushikesh Ubale
"""

import cv2
import numpy as np


pathImage = "2.jpg"
heightImg = 700
widthImg = 700
questions = 5
choices = 5
ans = [1, 2, 0, 2, 4]


def reorder(myPoints):

    myPoints = myPoints.reshape((4, 2)) # REMOVE EXTRA BRACKET
    #print(myPoints)
    myPointsNew = np.zeros((4, 1, 2), np.int32) # NEW MATRIX WITH ARRANGED POINTS
    add = myPoints.sum(1)
    #print(add)
    #print(np.argmax(add))
    myPointsNew[0] = myPoints[np.argmin(add)]  #[0,0]
    myPointsNew[3] =myPoints[np.argmax(add)]   #[w,h]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] =myPoints[np.argmin(diff)]  #[w,0]
    myPointsNew[2] = myPoints[np.argmax(diff)] #[0,h]

    return myPointsNew

def rectContour(contours):

    rectCon = []
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i) #find out area of each contour 
        if area > 50:
            peri = cv2.arcLength(i, True) #closed arc length  
            approx = cv2.approxPolyDP(i, 0.02 * peri, True) #resolution 0.02*perameter gives number of corner points
            if len(approx) == 4:
                rectCon.append(i)
    rectCon = sorted(rectCon, key=cv2.contourArea,reverse=True)#arrange all the rectangle points on basis of area in decending order 
    #print(len(rectCon))
    return rectCon

def getCornerPoints(cont):
    peri = cv2.arcLength(cont, True) # LENGTH OF CONTOUR
    approx = cv2.approxPolyDP(cont, 0.02 * peri, True) # APPROXIMATE THE POLY TO GET CORNER POINTS
    return approx

def splitBoxes(img):
    rows = np.vsplit(img,5)#vertical split
    boxes=[]
    for r in rows:
        cols= np.hsplit(r,5)#horizontal split
        for box in cols:
            boxes.append(box)
    return boxes



def showAnswers(img,myIndex,grading,ans,questions=5,choices=5):
     secW = int(img.shape[1]/questions) #section width
     secH = int(img.shape[0]/choices)

     for x in range(0,questions):
         myAns= myIndex[x]
         cX = (myAns * secW) + secW // 2
         cY = (x * secH) + secH // 2
         if grading[x]==1:
            myColor = (0,255,0)
            #cv2.rectangle(img,(myAns*secW,x*secH),((myAns*secW)+secW,(x*secH)+secH),myColor,cv2.FILLED)
            cv2.circle(img,(cX,cY),50,myColor,cv2.FILLED)
         else:
            myColor = (0,0,255)
            #cv2.rectangle(img, (myAns * secW, x * secH), ((myAns * secW) + secW, (x * secH) + secH), myColor, cv2.FILLED)
            cv2.circle(img, (cX, cY), 50, myColor, cv2.FILLED)

            # CORRECT ANSWER
            myColor = (0, 255, 0)
            correctAns = ans[x]
            cv2.circle(img,((correctAns * secW)+secW//2, (x * secH)+secH//2),
            20,myColor,cv2.FILLED)



count = 0

img = cv2.imread(pathImage)
img = cv2.resize(img, (widthImg, heightImg))  # RESIZE IMAGE
imgFinal = img.copy()
#imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)  # CREATE A BLANK IMAGE FOR TESTING DEBUGGING IF REQUIRED
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # CONVERT IMAGE TO GRAY SCALE
imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # ADD GAUSSIAN BLUR
imgCanny = cv2.Canny(imgBlur, 10, 70)  # APPLY CANNY detect the edges

## FIND ALL COUNTOURS
imgContours = img.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
imgBigContour = img.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # FIND ALL CONTOURS
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)  # DRAW ALL DETECTED CONTOURS
rectCon = rectContour(contours)  # FILTER FOR RECTANGLE CONTOURS
biggestPoints = getCornerPoints(rectCon[0])  # GET CORNER POINTS OF THE BIGGEST RECTANGLE
gradePoints = getCornerPoints(rectCon[1])  # GET CORNER POINTS OF THE SECOND BIGGEST RECTANGLE

if biggestPoints.size != 0 and gradePoints.size != 0:

            # BIGGEST RECTANGLE WARPING
    biggestPoints = reorder(biggestPoints)  # REORDER FOR WARPING
    cv2.drawContours(imgBigContour, biggestPoints, -1, (0, 255, 0), 20)  # DRAW THE BIGGEST CONTOUR
    pts1 = np.float32(biggestPoints)  # PREPARE POINTS FOR WARP
    pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])  # PREPARE POINTS FOR WARP
    matrix = cv2.getPerspectiveTransform(pts1, pts2)  # GET TRANSFORMATION MATRIX
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))  # APPLY WARP PERSPECTIVE gives bird eye view image

            # SECOND BIGGEST RECTANGLE WARPING
    cv2.drawContours(imgBigContour, gradePoints, -1, (255, 0, 0), 20)  # DRAW THE BIGGEST CONTOUR
    gradePoints = reorder(gradePoints)  # REORDER FOR WARPING
    ptsG1 = np.float32(gradePoints)  # PREPARE POINTS FOR WARP
    ptsG2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])  # PREPARE POINTS FOR WARP
    matrixG = cv2.getPerspectiveTransform(ptsG1, ptsG2)  # GET TRANSFORMATION MATRIX
    imgGradeDisplay = cv2.warpPerspective(img, matrixG, (325, 150))  # APPLY WARP PERSPECTIVE

            # APPLY THRESHOLD
    imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)  # CONVERT TO GRAYSCALE
    imgThresh = cv2.threshold(imgWarpGray, 170, 255, cv2.THRESH_BINARY_INV)[1]  # APPLY THRESHOLD AND INVERSE

    boxes = splitBoxes(imgThresh)  # GET INDIVIDUAL BOXES
    #cv2.imshow("Split Test ", boxes[3])
    #print(cv2.countNonZero(boxes[1]),cv2.countNonZero(boxes[2])) #can see pixel difference
    countR = 0
    countC = 0
    myPixelVal = np.zeros((questions, choices))  # TO STORE THE NON ZERO VALUES OF EACH BOX
    ##calculating non zero pixel values of box
    for image in boxes:
                # cv2.imshow(str(countR)+str(countC),image)
        totalPixels = cv2.countNonZero(image)
        myPixelVal[countR][countC] = totalPixels
        countC += 1
        if (countC == choices): countC = 0;countR += 1
    print(myPixelVal)
            # FIND THE USER ANSWERS AND PUT THEM IN A LIST
    myIndex = []
    for x in range(0, questions):
        arr = myPixelVal[x]
        myIndexVal = np.where(arr == np.amax(arr))
        print(myIndexVal[0])
        myIndex.append(myIndexVal[0][0])
            # print("USER ANSWERS",myIndex)

            # COMPARE THE VALUES TO FIND THE CORRECT ANSWERS
    grading = []
    for x in range(0, questions):
        if ans[x] == myIndex[x]:
            grading.append(1)
        else:
            grading.append(0)
            # print("GRADING",grading)
    score = (sum(grading) / questions) * 100  # FINAL GRADE
            # print("SCORE",score)

            # DISPLAYING ANSWERS
    #showAnswers(imgWarpColored, myIndex, grading, ans)  # DRAW DETECTED ANSWERS
    
    imgRawDrawings = np.zeros_like(imgWarpColored)  # NEW BLANK IMAGE WITH WARP IMAGE SIZE
    showAnswers(imgRawDrawings, myIndex, grading, ans)  # DRAW ON NEW IMAGE
    invMatrix = cv2.getPerspectiveTransform(pts2, pts1)  # INVERSE TRANSFORMATION MATRIX
    imgInvWarp = cv2.warpPerspective(imgRawDrawings, invMatrix, (widthImg, heightImg))  # INV IMAGE WARP

            # DISPLAY GRADE
    imgRawGrade = np.zeros_like(imgGradeDisplay, np.uint8)  # NEW BLANK IMAGE WITH GRADE AREA SIZE
    cv2.putText(imgRawGrade, str(int(score)) + "%", (70, 100)
                , cv2.FONT_HERSHEY_COMPLEX, 3, (0, 255, 0), 3)  # ADD THE GRADE TO NEW IMAGE
    invMatrixG = cv2.getPerspectiveTransform(ptsG2, ptsG1)  # INVERSE TRANSFORMATION MATRIX
    imgInvGradeDisplay = cv2.warpPerspective(imgRawGrade, invMatrixG, (widthImg, heightImg))  # INV IMAGE WARP

            # SHOW ANSWERS AND GRADE ON FINAL IMAGE
    imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWarp, 1, 0)
    imgFinal = cv2.addWeighted(imgFinal, 1, imgInvGradeDisplay, 1, 0)

cv2.imshow("original", img)
cv2.imshow("Gray", imgGray)
cv2.imshow("Edges", imgCanny)
cv2.imshow("Contours", imgContours)
cv2.imshow("Biggest Contour",imgBigContour)
cv2.imshow("Threshold",imgThresh)
cv2.imshow("Bird eye view",imgWarpColored)
cv2.imshow("Final Result", imgFinal)

cv2.waitKey(0)


