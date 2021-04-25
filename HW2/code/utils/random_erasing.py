import random
import math
import torch


class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=1, sl=0.02, sh=0.4, r1=0.3, mean=None):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        while True:
            channel, h, w = img.shape
            area = h * w

            gt_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(gt_area * aspect_ratio)))
            w = int(round(math.sqrt(gt_area / aspect_ratio)))

            if w < w and h < h:
                x = random.randint(0, h - h)
                y = random.randint(0, w - w)
                if channel == 3:
                    img[:, x:x + h, y:y + w] = torch.empty((3, h, w), dtype=torch.float32).normal_()
                else:
                    img[0, x:x + h, y:y + w] = torch.empty((1, h, w), dtype=torch.float32).normal_()
                return img
