
import torch
import torchvision.transforms.functional as F
import torchvision.transforms as T
import random

class GroupTransform:
    def __init__(self, size=(224, 224), is_train=True):
        self.size = size
        self.is_train = is_train

    def __call__(self, img_group):
        if self.is_train:
            do_flip = random.random() > 0.5
            brightness = random.uniform(0.7, 1.3)
            contrast = random.uniform(0.7, 1.3)

            do_perspective = random.random() > 0.5
            if do_perspective:
                persp_params = T.RandomPerspective.get_params(
                    width=img_group[0].size[0],
                    height=img_group[0].size[1],
                    distortion_scale=0.3
                )

            w, h = img_group[0].size
            th, tw = int(h * random.uniform(0.8, 1.0)), int(w * random.uniform(0.8, 1.0))
            i = random.randint(0, h - th)
            j = random.randint(0, w - tw)
        else:
            do_flip = do_perspective = False
            i, j, th, tw = 0, 0, img_group[0].size[1], img_group[0].size[0]
            brightness = contrast = 1.0

        transformed_group = []
        for img in img_group:
            img = F.resized_crop(img, i, j, th, tw, self.size)

            if self.is_train:
                if do_flip:
                    img = F.hflip(img)
                if do_perspective:
                    img = F.perspective(img, *persp_params, interpolation=T.InterpolationMode.BILINEAR)
                img = F.adjust_brightness(img, brightness)
                img = F.adjust_contrast(img, contrast)

            img = F.to_tensor(img)
            img = F.normalize(img, [0.485, 0.456, 0.406],
                                   [0.229, 0.224, 0.225])

            transformed_group.append(img)

        return torch.stack(transformed_group)
