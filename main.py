import horizontalPipeline
import cv2
import numpy

# Load an color image
img = cv2.imread('test3.png', 1)

# Run image through pipeline
hPipeline = horizontalPipeline.HorizontalPipeline()
hPipeline.process(img)
contours = hPipeline.filter_contours_output
points = numpy.vstack(contours).squeeze()

pointCount = 0
xTotal = 0
for point in points:
    xTotal += point[0]
    pointCount += 1

height, width, channels = img.shape
xAvg = xTotal/pointCount
centerX = width/2
xAccuracy = 100 - abs((xAvg - centerX)/centerX) * 100

# Draw x average line
cv2.line(img, (int(xAvg), 0), (int(xAvg), height), (255,255,255), 1)

# Draw center circle
cv2.circle(img,(int(width/2),int(height/2)),18,(0,0,255),3)

# Display image
cv2.imshow('image',img)

# Draw accuracy text
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'Horizontal Accuracy: ' + str(round(xAccuracy, 2)),(30,50), font, 0.8, (255,255,255),1,cv2.LINE_AA)

# Draw found dart in blue
shape = img.copy()
cv2.drawContours(shape, contours, -1, (255, 0, 0), 2)
cv2.imshow("Edges", shape)

# Quit program with keypress
cv2.waitKey(0)