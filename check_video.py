import argparse
from pathlib import Path

if __name__ == "__main__":
    # print("Hello world!!!")
    bg_roi =[[492, 247, 758, 600],
            [582, 292, 786, 621],   
            [599, 303, 765, 606],
            [589, 304, 775, 602],
            [602, 316, 777, 594]
            ]
    parser  = argparse.ArgumentParser()
    parser.add_argument('--huongham', type=str, default='../dataset/aic22_track4_testA_video/TestA/testA_1.mp4', help='source')

    args = parser.parse_args()
    value = getattr(args, 'huongham')

    lst = value.split()[0].split("_", 5)
    tmp = lst[-1].split(".")[0]
    video_idx = int(tmp)
    print(bg_roi[video_idx - 1])

