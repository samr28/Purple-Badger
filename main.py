import top_view_pipeline
import cv2
import numpy
import threading
import queue
import time
import json

cam0Frames = queue.Queue()
out = cv2.VideoWriter("topview.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 187, (320,240))

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
        cv2.imshow("frame", img)
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
    cv2.imshow("frame", img)

    return accuracy

def getAverage(points, orientation):
    pointCount = 0
    total = 0
    for point in points:
        total += point[orientation]
        pointCount += 1
    avg = total/pointCount

    return avg

accuracyList = [[], []]

def captureCam0():
    cap = cv2.VideoCapture("sideview.avi")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_FPS, 187)

    t_end = time.time() + 5
    while time.time() < t_end:
        ret, frame = cap.read()
        cam0Frames.put(frame)
        if(ret == False):
            break

def saveFrames(frames):
    while(not frames.empty()):
        out.write(cam0Frames.get())

def processFrames():
    i = 0
    while(not cam0Frames.empty()):
        if(cam0Frames.qsize() == 1):
            break
        frame = cam0Frames.get()
        #write to video file
        pipeline = top_view_pipeline.Pipeline()
        if(i > 278 and i < 286):
            singleLocation = processDart(frame, pipeline, 240/2, 1)
            if(singleLocation is not None):
                accuracyList[0].append(singleLocation)
            cv2.waitKey()

        i += 1

def printJsonResults(x, y):
    result = {
        'name': "Shot",
        'x': x,
        'y': y
    }
    print(json.dumps(result))



captureCam0()
processFrames()
printJsonResults(numpy.mean(accuracyList[0]), numpy.mean(accuracyList[0]))
# saveFrames(cam0Frames)
