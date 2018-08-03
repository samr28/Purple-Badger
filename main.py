import side_view_pipeline
import top_view_pipeline
import cv2
import numpy

accuracyList = [[], []]

def processDart(img, pipeline, centerTargetVal, orientation):
    pipeline.process(img)
    contours = pipeline.filter_contours_output

    height, width, channcel = img.shape

    if(orientation == 0):
        cv2.line(img, (int(width/2), 0), (int(width/2), height), (0,0,255), 3)
    if(orientation == 1):
        cv2.line(img, (0, int(height/2)), (width, int(height/2)), (0,0,255), 3)

    if(not contours):
        return
    points = numpy.vstack(contours).squeeze()
    avg = getAverage(points, orientation)

    accuracy = (1 - abs((avg - centerTargetVal)/centerTargetVal)) * 100
    
    if(orientation == 0):
        cv2.line(img, (int(avg), 0), (int(avg), height), (0,255,0), 3)
        accuracyList[0].append(accuracy)

    if(orientation == 1):
        cv2.line(img, (0, int(avg)), (width, int(avg)), (0,255,0), 3)
        accuracyList[1].append(accuracy)

    # Draw found dart in blue
    cv2.drawContours(img, contours, -1, (255, 0, 0), 2)

    return avg

def getAverage(points, orientation):
    pointCount = 0
    total = 0
    for point in points:
        total += point[orientation]
        pointCount += 1
    avg = total/pointCount

    return avg
    
yList = []
xList = []

yHeight = None
yWidth = None
yChannels = None
yCap = cv2.VideoCapture('yTest.avi')

while(yCap.isOpened()):
    ret, frame = yCap.read() # 1 frame
    if(ret == False):
        break
    
    if(not yHeight and not yWidth and not yChannels):
        yHeight, yWidth, yChannels = frame.shape
    pipeline = top_view_pipeline.Pipeline()
    y = processDart(frame, pipeline, yHeight/2, 1)

    cv2.imshow('frame',frame)

    if(y is not None):
        yList.append(y)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

xHeight = None
xWidth = None
xChannels = None
xCap = cv2.VideoCapture('xTest.avi')

while(xCap.isOpened()):
    ret, frame = xCap.read()
    if(ret == False):
        break
    if(not xHeight and not xWidth and not xChannels):
        xHeight, xWidth, xChannels = frame.shape
    pipeline = side_view_pipeline.Pipeline()
    x = processDart(frame, pipeline, xWidth/2, 0)
    cv2.imshow('frame',frame)

    if(x is not None):
        xList.append(x)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break


finalWidth = 500
finalHeight = 500
blankImage = numpy.zeros((finalWidth,finalHeight,3), numpy.uint8)
bHeight, bWidth, bChanncel = blankImage.shape

cv2.circle(blankImage,(int(finalWidth/2),int(finalHeight/2)),20,(0,0,255),1)
cv2.circle(blankImage,(int(finalWidth/2),int(finalHeight/2)),60,(0,0,255),1)
cv2.circle(blankImage,(int(finalWidth/2),int(finalHeight/2)),100,(0,0,255),1)
cv2.circle(blankImage,(int(finalWidth/2),int(finalHeight/2)),140,(0,0,255),1)

xError = (numpy.mean(xList) - (xWidth / 2)) / numpy.mean(xList)
yError = (numpy.mean(yList) - (yHeight / 2)) / numpy.mean(yList)

print("xError: ", xError)
print("yError: ", yError)

print("centerX: ", (xWidth/2))
print("centerY: ", (yHeight/2))

xOffset = xError * finalWidth
yOffset = yError * finalHeight
print(f"xOffset: {xOffset}, yOffset: {yOffset}")
xPos = finalWidth/2 + xOffset
yPos = finalHeight/2 + yOffset

print(f"centerX: {finalWidth/2}, centerY: {finalHeight/2}")
print(f"xPos: {xPos}, yPos: {yPos}")

cv2.circle(blankImage,(int(xPos), int(yPos)),18,(0,255,0),-1)

# Draw final accuracy text
finalAccuracy = round((numpy.mean(accuracyList[0]) + numpy.mean(accuracyList[1])) / 2, 2)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(blankImage,f"Overall Accuracy: {finalAccuracy}%",(30,50), font, 0.8, (255,255,255),1,cv2.LINE_AA)

cv2.destroyAllWindows()
cv2.imshow("image", blankImage)
cv2.waitKey()

xCap.release()
yCap.release()
cv2.destroyAllWindows()