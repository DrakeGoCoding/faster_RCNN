import cv2
import imutils
import numpy as np
import time
import time
import os

from image_detector import ImageDetector

from imutils.video import FileVideoStream

vid_path = 'test.mp4'
result_path = 'test_predicted.mp4'

v = FileVideoStream(vid_path)
vs = v.start()


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    # return the list of (x, y)-coordinates
    return coords

#
# detector = dlib.get_frontal_face_detector()  # replace with detector
detector = ImageDetector(model='trained_model/snapshot_model_20180404.npz')
#
# face_cascade_default = cv2.CascadeClassifier('other/default_cascade.xml')  # remove this

default_size = (int(v.stream.get(3)), int(v.stream.get(4)))

size = default_size

result = cv2.VideoWriter(os.path.splitext(result_path)[0] + '.avi',
                         cv2.VideoWriter_fourcc(*'MPEG'),
                         10, size)

time.sleep(2.0)

frame_rate = 5
prev = time.time()

while vs.more():
    # grab the frame from the video stream, resize it, and convert it
    # to grayscale
    time_elapsed = time.time() - prev
    if time_elapsed < 1. / frame_rate:
        continue
    else:
        prev = time.time()

    frame = vs.read()

    if frame is None:
        break

    frame = cv2.resize(frame, size)

    # Replace this with your detect
    faceRects = detector.detect(np.swapaxes(frame.astype(np.float32), 0, 2))
    # faceRects = face_cascade_default.detectMultiScale(
    #     frame, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30),
    #     flags=cv2.CASCADE_SCALE_IMAGE)
    # cv2.rectangle(frame, (20, 20), (100, 100),
    #               (235, 168, 58), 2)

    for (_, fX, fY, fW, fH) in faceRects:
        # extract the face ROI
        faceROI = frame[fY:fY + fH, fX:fX + fW]
        cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH),
                      (0, 255, 0), 2)
        # apply eyes detection to the face ROI
        # eyeRects = detectors["eyes"].detectMultiScale(
        #     faceROI, scaleFactor=1.1, minNeighbors=10,
        #     minSize=(15, 15), flags=cv2.CASCADE_SCALE_IMAGE)
        # # apply smile detection to the face ROI
        # smileRects = detectors["smile"].detectMultiScale(
        #     faceROI, scaleFactor=1.1, minNeighbors=10,
        #     minSize=(15, 15), flags=cv2.CASCADE_SCALE_IMAGE)
    result.write(frame)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

result.release()
cv2.destroyAllWindows()
vs.stop()
