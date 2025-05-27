import cv2

def resizer(img):
    return cv2.resize(img, (64,64),1,1,interpolation=cv2.INTER_LINEAR)
