import cv2
import time
import logging
import numpy as np
from argparse import ArgumentParser
from collections import deque 


parser = ArgumentParser()
parser.add_argument('--device', default=0, help='choose a device')
parser.add_argument('--thr', default=80, help="focus measures that fall below this value will be considered 'blurry'")
args = parser.parse_args()


def draw(frame, fm, thr, width, height, avg_blurr):
    frame = frame[:, :, ::-1]

    text = "Not Blurry"

    if fm < thr:
        text = "Blurry"
    
    cv2.putText(frame, "{}: {:.2f}".format(text, fm, height, width), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 205, 50), 2)
    cv2.putText(frame, "{}x{}".format(height, width), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 205, 50), 2)
    cv2.putText(frame, "average_blurr: {:.4f}".format(avg_blurr), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 205, 50), 2)
    cv2.imshow('image', frame)
    cv2.waitKey(1)


def compute_avg_blurr(last_avg, fm, fm_first):
    return last_avg + 0.25 * (fm - fm_first)


def init_capture_device(source):
    sleep_sec = 0.5
    while True: 
        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            break
        logging.error("camera is not opened, sleeping {}s and try again!".format(sleep_sec))
        cap.release()
        time.sleep(sleep_sec)

    # cap.set(3, 640)
    # cap.set(4, 480)
    return cap

def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()



def main():
    source = int(args.device)
    thr = args.thr
    blurr_list = []
    queue = deque(blurr_list)
    cap = init_capture_device(source)
    idx = 0
    last_blurr = 0

    while True:
        ret, frame = cap.read()
        frame = frame[:, :, ::-1]
        fm = variance_of_laplacian(frame)
        
        
        last_avg = last_blurr/4
        last_blurr = last_blurr + fm
        queue.append(fm)
        
        
        if idx < 4:
            avg_blurr = 0
        else:
            fm_first = queue.popleft()
            last_blurr = last_blurr-fm_first
            avg_blurr = compute_avg_blurr(last_avg, fm, fm_first)    
            
        width, height = frame.shape[1], frame.shape[0]
        draw(frame, fm, thr, width, height, avg_blurr)
        idx += 1

    

if __name__ == '__main__':
    main()