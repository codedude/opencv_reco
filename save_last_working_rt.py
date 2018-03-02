#! /usr/bin/python2.7
# -*- coding: utf-8 -*-
#

try:
    import sys, time, io
    import numpy as np
    import RPi.GPIO as GPIO
    import picamera
    from picamera.array import PiRGBArray
    import matplotlib.pyplot as plt
    import cv2
except ImportError as e:
    print("Failed loading modules : {0}".format(e))
    sys.exit(2)


def main():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(17,GPIO.IN)

    GPIO.setup(27,GPIO.OUT)
    GPIO.setup(23,GPIO.OUT)
    GPIO.setup(22,GPIO.OUT)

    freq= 50
    gr = GPIO.PWM(27, freq)
    gg = GPIO.PWM(22, freq)
    gb = GPIO.PWM(23, freq)

    prev_input = 0
    # Create the in-memory stream
    stream = io.BytesIO()
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

    camera = picamera.PiCamera()
    camera.resolution = (320, 240)
    camera.framerate = 25
    rawCapture = PiRGBArray(camera, size=(320, 240))

    time.sleep(0.5)

    i = 0

    cv2.namedWindow('face', cv2.WINDOW_NORMAL)

#Train svm
    #Get samples
    nmoods = [
        ['normal/img_0.jpg', 'normal/img_1.jpg', 'normal/img_2.jpg',
        'normal/img_3.jpg', 'normal/img_4.jpg', 'normal/img_5.jpg',
        'normal/img_6.jpg', 'normal/img_7.jpg', 'normal/img_8.jpg',
        'normal/img_9.jpg', 'normal/img_10.jpg', 'normal/img_11.jpg',
        'normal/img_12.jpg'],
        ['smile/img_0.jpg', 'smile/img_1.jpg', 'smile/img_2.jpg',
        'smile/img_3.jpg', 'smile/img_4.jpg', 'smile/img_5.jpg',
        'smile/img_6.jpg', 'smile/img_7.jpg', 'smile/img_8.jpg',
        'smile/img_9.jpg', 'smile/img_10.jpg', 'smile/img_11.jpg',
        'smile/img_12.jpg', 'smile/img_13.jpg', 'smile/img_14.jpg',
        'smile/img_15.jpg', 'smile/img_16.jpg', 'smile/img_17.jpg',
        'smile/img_18.jpg', 'smile/img_19.jpg'],
        ['angry/img_0.jpg', 'angry/img_1.jpg', 'angry/img_2.jpg',
        'angry/img_3.jpg', 'angry/img_4.jpg', 'angry/img_5.jpg',
        'angry/img_6.jpg', 'angry/img_7.jpg', 'angry/img_8.jpg',
        'angry/img_9.jpg', 'angry/img_10.jpg', 'angry/img_11.jpg',
        'angry/img_12.jpg', 'angry/img_13.jpg']
    ]

    n_normal = [0] * len(nmoods[0])
    n_happy = [1] * len(nmoods[1])
    n_angry = [2] * len(nmoods[2])
    #Grayscale, (160, 120)
    moods = [cv2.imread(m, 0) for mood in nmoods for m in mood]
    moods = np.array(moods)
    for i in range(len(moods)):
        moods[i] = cv2.resize(moods[i], (80, 60));

    moods = np.array([[c  for r in img for c in r] for img in moods], dtype=np.float32)

    #Samples
    #Labels
    #Let assume 0=normal, 1=happy, 2=angry
    labels = np.array(n_normal + n_happy + n_angry)

    svm = cv2.SVM()
    svmparams = dict( kernel_type = cv2.SVM_LINEAR,
                       svm_type = cv2.SVM_C_SVC,
                       C = 2 )

    svm.train(moods, labels, params = svmparams)

    #testresult = np.float32( [svm.predict(s) for s in moods])
    #print(testresult)
    #print(labels)
    moods = None
    labels = None
    key = 0

    for frame in camera.capture_continuous(rawCapture,
            format="bgr", use_video_port=True):
        # grab the raw NumPy array representing the image, then initialize the timestamp
        # and occupied/unoccupied text
        image = frame.array

        # clear the stream in preparation for the next frame
        rawCapture.truncate()
        rawCapture.seek(0)


        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

        push = GPIO.input(17)
        if ((not prev_input) and push or 1):
            print("Push")
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags = cv2.cv.CV_HAAR_SCALE_IMAGE
            )
            for (x, y, w, h) in faces:
                img = cv2.resize(gray[y:y+h, x:x+w], (80, 60));
                img = [c  for r in img for c in r]
                img = np.array(img, dtype=np.float32)
                mood = svm.predict(img)
                setRect(image, (x, y, w, h), mood)
                cv2.putText(image, getMood(mood), (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
                setLed(mood, gr, gg, gb)

        prev_input = push

        cv2.imshow('face', image)
        key = cv2.waitKey(20) & 0xFF


    cv2.destroyAllWindows()

    return 0


def getMood(n):
    if n == 1:
        return "happy"
    elif n == 2:
        return "angry"

    return "normal"

def setLed(mood, r, g, b):
    c = [
        (0, 0, 100),
        (0, 100, 0),
        (100, 0, 0)
    ]

    r.start(c[int(mood)][0])
    g.start(c[int(mood)][1])
    b.start(c[int(mood)][2])

def setRect(image, pos, mood):
    c = [
        (250, 0, 0),
        (0, 250, 0),
        (0, 0, 250)
    ]
    cv2.rectangle(image, (pos[0], pos[1]), (pos[0]+pos[2], pos[1]+pos[2]),
        c[int(mood)], 2)

if __name__ == '__main__':
    print('')
    r = main()
    print('')
    sys.exit(r)
