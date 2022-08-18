bg_roi=[[492, 247, 758, 600],
        [582, 292, 786, 621],   
        [599, 303, 765, 606],
        [589, 304, 775, 602],
        [602, 316, 777, 594]
        ]
video_idx = 0
def checkhihi():
    print(video_idx)
def main():
    print("Hello")
if __name__ == "__main__":
    video_idx = 4
    x_roi, y_roi, w_roi, h_roi = bg_roi[int(video_idx) - 1]
    print(video_idx, bg_roi[int(video_idx) - 1])
    checkhihi()