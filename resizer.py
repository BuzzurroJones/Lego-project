import cv2
import torch


# resize to 64x64, normalize, convert to CHW from HWC, return a torch tensor
def resizer(img):
    resized = cv2.resize(img, (64,64),interpolation=cv2.INTER_LINEAR)

    resized = resized.astype('float32')/255

    resized = resized.transpose((2,0,1))

    tensor = torch.from_numpy(resized)
    return tensor
