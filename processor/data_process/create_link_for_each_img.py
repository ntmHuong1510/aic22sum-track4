"""
    This file read the whole list of image and corresponding segment of the dataset and split to each class in total 116 ones
"""
from os import listdir, path, rename


if __name__ == "__main__":
    file_read = open("../docs/image_segment.txt", "r")
    for filename in file_read:
        class_id = int(filename.split(":")[0])
        print(class_id)
        file_write = open("../link_img_seg_per_class/" + str(class_id).zfill(5) + ".txt", "a")
        file_write.write(filename.split(":")[1])