import cv2
import numpy as np
from PIL import Image


def merge_img(largeImg, alpha, smallImg, beta, gamma=0.0, regionTopLeftPos=(0, 0)):
    srcW, srcH = largeImg.shape[1::-1]
    refW, refH = smallImg.shape[1::-1]
    x, y = regionTopLeftPos
    if (refW > srcW) or (refH > srcH):
        # raise ValueError("img2's size must less than or equal to img1")
        raise ValueError(
            f"img2's size {smallImg.shape[1::-1]} must less than or equal to img1's size {largeImg.shape[1::-1]}")
    else:
        if (x + refW) > srcW:
            x = srcW - refW
        if (y + refH) > srcH:
            y = srcH - refH
        destImg = np.array(largeImg)
        tmpSrcImg = destImg[y:y + refH, x:x + refW]
        tmpImg = cv2.addWeighted(tmpSrcImg, alpha, smallImg, beta, gamma)
        destImg[y:y + refH, x:x + refW] = tmpImg
        return destImg


if __name__ == "__main__":
    path = "./output3/output3362.png"

    img = cv2.resize(cv2.imread(path), (1280, 480))
    logo = cv2.imread("warningflag.png")
    logo = cv2.resize(logo, (150, 150))
    logo_letter = cv2.imread("warning.png")

    # h_logo_letter, w_logo_letter, _ = logo_letter.shape
    # print(logo_letter.shape)
    # print(img.shape)
    # img[:h_logo_letter,:w_logo_letter] = logo_letter

    img = merge_img(img, 0.8, logo, 0.2, 0, (100, 100))

    cv2.imshow("print", img)
    cv2.imshow("logo", logo)
    cv2.imshow("logo_letter", logo_letter)
    # img[:, :]

    cv2.waitKey(0)
    cv2.destroyAllWindows()
