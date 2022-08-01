"""
    open videos in test set

    enter
    |__'q': exit the program
    |__'f': move to the next video
    |__'b': move to the previous video
    |__'p': pause video
"""

import cv2
from os import listdir, rename, path
import argparse


def get_videos_path(rel_path, video_file):
    arr = []
    print(path.join(rel_path, video_file))
    with open(path.join(rel_path, video_file)) as file:
        for filename in file:
            arr.append(path.join(rel_path, filename.split(" ")[1].split("\n")[0]))
    return arr

def open_video(videos):
    idx, lenvid = 0, len(videos)
    cap = cv2.VideoCapture(videos[idx])
    ret, frame = cap.read()
    while(1):
        ret, frame = cap.read()
        cv2.imshow('frame',frame)
        key = cv2.waitKey(1)
        if key == ord('q') or ret == False:
            cap.release()
            cv2.destroyAllWindows()
            break
        elif key == ord('p'):
            cv2.waitKey(0)
        elif key == ord('f'):
            print(idx, idx % lenvid)
            idx = idx + 1
            cap = cv2.VideoCapture(videos[idx % lenvid])
        elif key == ord('b'):
            idx = idx - 1 if idx != 0 else  lenvid - 1
            cap = cv2.VideoCapture(videos[idx % lenvid])
        cv2.imshow('frame',frame)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filevideo', default='video_id.txt', help = "file contains videos' name in test set")
    parser.add_argument('--rel_path', default='..\\..\\..\\dataset\\aic22_track4_testA_video\\TestA\\', help = "relative path of videos")
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    videos = get_videos_path(opt.rel_path, opt.filevideo)
    open_video(videos)
    
