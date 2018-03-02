#! /usr/bin/python2.7
# -*- coding: utf-8 -*-
#

try:
    import sys, time, io
    import RPi.GPIO as GPIO
except ImportError as e:
    print("Failed loading modules : {0}".format(e))
    sys.exit(2)


def main():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(27,GPIO.IN)
    GPIO.setup(22,GPIO.IN)
    GPIO.setup(23,GPIO.IN)

    #r=, g=, b=


    return 0



if __name__ == '__main__':
    print('')
    r = main()
    print('')
    sys.exit(r)
