import os
import cv2

from albumentations import (
    Resize, HorizontalFlip, RandomRotate90,
    RandomBrightnessContrast, Compose,
    ShiftScaleRotate, GaussianBlur, CLAHE,
    HueSaturationValue, CoarseDropout, Solarize)

augmentations = [
    Compose([Resize(640, 640)]),
    Compose([HorizontalFlip(p=1)]),
    Compose([RandomRotate90(p=1)]),
    Compose([RandomBrightnessContrast(p=1)]),
    Compose([ShiftScaleRotate(shift_limit=0.1,scale_limit=0.1,
                              rotate_limit=15, p=1)]),
    Compose([GaussianBlur(blur_limit=7, p=1)]),
    Compose([CLAHE(clip_limit=2, p=1)]),
    Compose([HueSaturationValue(hue_shift_limit=20,
                                sat_shift_limit=30,
                                val_shift_limit=20, p=1)]),
    Compose([CoarseDropout(max_holes=8, max_height=32, max_width=32,
                           fill_value=0, p=1)]),
    Compose([Solarize(threshold=128, p=1)]),
]


def process_images(image_list):
    for image in image_list:
        src = os.path.join(src_folder, image)
        img = cv2.imread(src)
        original_dst = os.path.join(dst_folder, image)
        cv2.imwrite(original_dst, img)

        for i, aug in enumerate(augmentations):
            augmented = aug(image=img)
            aug_img = augmented["image"]
            aug_dst = os.path.join(dst_folder,
                      f"{os.path.splitext(image)[0]}_aug_{i+1}.jpg")
            cv2.imwrite(aug_dst, aug_img)


# Define the folder paths
src_folder = "original_images"
dst_folder = "aug_results"
os.makedirs(dst_folder, exist_ok=True)


if __name__ == "__main__":
    images = [img for img in os.listdir(src_folder) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    process_images(images)
