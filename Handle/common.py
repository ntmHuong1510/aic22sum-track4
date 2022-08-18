from os.path import abspath, exists, isfile
from typing import NewType, Tuple
from datetime import datetime
from re import sub


OBJ_DATA = abspath("../processor/yolov5/data/aic-track4.yaml")


Xyxy = NewType("Xyxy", Tuple[int, int, int, int])
Xywh = NewType("Xywh", Tuple[int, int, int, int])
Vector = NewType("Vector", Tuple[int, int])