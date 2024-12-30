import os
import cv2

from albumentations import (
    Resize, HorizontalFlip, RandomRotate90,
    RandomBrightnessContrast, Compose,
    ShiftScaleRotate, GaussianBlur, CLAHE,
    HueSaturationValue, CoarseDropout, Solarize)

augmentations = [
    Compose([Resize(640, 640)]),
    Compose([HorizontalFlip(p=0.5)]),
    Compose([ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5)]),
    Compose([RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5)]),
    Compose([GaussianBlur(blur_limit=3, p=0.3)]),
    Compose([CLAHE(clip_limit=1, p=0.3)]),
    Compose([HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5)]),
]


def process_images(src_folder, dst_folder):
    os.makedirs(dst_folder, exist_ok=True)
    image_list = [img for img in os.listdir(src_folder) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
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

    print(f"Augmentation completed. New images saved in: {dst_folder}")

# Example usage:
# process_images("original_images", "aug_results")
