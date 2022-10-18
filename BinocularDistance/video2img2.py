"""
-------------------------------------
# -*- coding: utf-8 -*-
# @Time    : 2020/10/1 15:44:12
# @Author  : Giyn
# @Email   : giyn.jy@gmail.com
# @File    : video_processing.py
# @Software: PyCharm
-------------------------------------
"""

import cv2
import logging
import os

# log information settings
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s: %(message)s')


def save_image(num, image):
    """Save the images.

    Args:
        num: serial number
        image: image resource

    Returns:
        None
    """
    image_path = save_path + '/' + video_name + '{}.png'.format(str(num))
    cv2.imwrite(image_path, image)

video_name = 'output3'
file_path = './video/' + video_name + '.avi'
save_path = './video/' + video_name
if not os.path.exists(save_path):
    os.makedirs(save_path)
    print(save_path)

vc = cv2.VideoCapture(file_path)  # import video files

# determine whether to open normally
if vc.isOpened():
    ret, frame = vc.read()
else:
    ret = False

count = 0  # count the number of pictures
frame_interval = 30  # video frame count interval frequency
frame_interval_count = 0

# loop read video frame
while ret:
    ret, frame = vc.read()
    # store operation every time f frame
    if ret == True:
        save_image(count, frame)
        logging.info("num：" + str(count) + ", frame: " +
                     str(frame_interval_count))
        count += 1
    else:
        break
    frame_interval_count += 1

print
vc.release()
