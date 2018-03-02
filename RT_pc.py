#! /usr/bin/python2.7
# -*- coding: utf-8 -*-
#
#
#

try:
    import sys, time, os
    import numpy as np
    import cv2
except ImportError as e:
    print("Failed loading modules : {0}".format(e))
    sys.exit(2)


SCREEN_RES = (640, 480)
SAMP_SIZE = (160, 120)
FPS = 32

dir_save = "save"
dir_moods = ["normal", "happy", "angry", "surprise"]
id_moods = [0, 1, 2, 3]


def main():
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")


#Get samples
    #Moods directory


    moods = []
    n_moods = []
    for i in range(len(dir_moods)):
        #Get the files in dirs
        tmp = []
        for f in os.listdir(dir_moods[i]):
            pic_path = os.path.join(dir_moods[i], f)
            if os.path.isfile(pic_path):
                tmp.append(pic_path)

        #How many pics for each moods
        n_moods.append(len(tmp))
        #Convert img for training (1row = 1grayscale 1D image)
        for pic in tmp:
            img = cv2.imread(pic, 0)
            img = cv2.resize(img, SAMP_SIZE)
            img = np.array([c  for r in img for c in r] , dtype=np.float32)
            moods.append(img)

    #Get labels
    labels = []
    for i in id_moods:
        labels.extend([i]*n_moods[i])

    moods = np.array(moods, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)

    svm = cv2.ml.SVM_create()

    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setC(2)
    svm.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 100, 1.e-06))
    svm.train(moods, labels, cv2.ml.ROW_SAMPLE)

    #Testing predict
    print("Test the algorithm : ")
    test = np.float32( [svm.predict(s) for s in moods])
    for i in range(len(labels)):
        if test[i] != labels[i]:
            sys.stdout.write("Found %.2f instead of %.2f (id:%d)"
                % (test[i], labels[i], i))
    print('\n')

    #For memory clean ?
    moods = None
    labels = None

    key = -1


    capture = cv2.VideoCapture(0);
    if(capture.isOpened() == True):
        print("Default camera opened !")
    else:
        print("Error opening camera")
        exit(1)

    time.sleep(0.3)

    capture.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, SCREEN_RES[0])
    capture.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, SCREEN_RES[1])

    cv2.namedWindow('face', cv2.WINDOW_NORMAL)

    time.sleep(0.3)

    takePic = False
    id_img = 0

    while(True):
        ret, image = capture.read()
        if(ret == False):
            print("Error while reading frame")
            continue

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(25, 25),
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )
        if key == ord("r"):
            takePic = True

        for (x, y, w, h) in faces:
            if takePic:
                savePic(image[y:y+h, x:x+w], i)
                i += 1
            img = cv2.resize(gray[y:y+h, x:x+w], SAMP_SIZE);
            img = [c  for r in img for c in r]
            img = np.array(img, dtype=np.float32)
            mood = svm.predict(img)
            setRect(image, (x, y, w, h), mood)
            setText(image, (x, y), mood)

        takePic = False

        cv2.imshow('face', image)
        key = cv2.waitKey(10) & 0xFF


    capture.release()
    cv2.destroyAllWindows()

    return 0


def getMood(n):
    if n == 0:
        return "normal"
    if n == 1:
        return "happy"
    elif n == 2:
        return "angry"
    elif n == 3:
        return "surprise"

    return "unknown"

def setRect(image, pos, mood):
    #bgr format !
    c = [
        (250, 0, 0),
        (0, 250, 0),
        (0, 0, 250),
        (0, 250, 250)
    ]
    cv2.rectangle(image, (pos[0], pos[1]), (pos[0]+pos[2], pos[1]+pos[2]),
        c[int(mood)], 2)

def setText(image, pos, mood):
    mood = getMood(mood)
    cv2.putText(image, mood, (pos[0], pos[1]-3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0))


def savePic(image, num):
    curr_time = str(time.time())
    pic_path = os.path.join(dir_save, "img_"+curr_time+".jpg")

    #Search for a free filename
    i = 1
    while(os.path.isfile(pic_path)):
        pic_path = os.path.join(dir_save, "img_"+curr_time+"_"+str(i)+".jpg")

    cv2.imwrite(pic_path, image)

    return True

if __name__ == '__main__':
    print('')
    r = main()
    print('')
    sys.exit(r)
