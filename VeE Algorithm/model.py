import os
import shutil
from ultralytics import YOLO
from sklearn.model_selection import train_test_split


def prepare_dataset(true_folder, false_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_folder, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_folder, split, "labels"), exist_ok=True)

    images = []
    for folder, label in [(true_folder, 0), (false_folder, 1)]:
        for img_file in os.listdir(folder):
            if img_file.lower().endswith('.tiff'):
                images.append((os.path.join(folder, img_file), label))

    train_val, test = train_test_split(images, test_size=0.2, random_state=42)
    train, val = train_test_split(train_val, test_size=0.2, random_state=42)
    splits = {"train": train, "val": val, "test": test}

    for split_name, split_data in splits.items():
        for img_path, label in split_data:
            dest_img_path = os.path.join(output_folder, split_name, "images", os.path.basename(img_path))
            shutil.copy(img_path, dest_img_path)

            label_file = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
            label_path = os.path.join(output_folder, split_name, "labels", label_file)

            if label == 0:  # Positive sample
                annotation = f"{label} 0.5 0.5 0.5 0.5\n"
                with open(label_path, "w") as f:
                    f.write(annotation)
            else:  # Negative sample
                with open(label_path, "w") as f:
                    pass

    print("Dataset preparation completed.")


def train_model(model_name="yolov8n", epochs=50):
    model = YOLO(model_name)  # Load YOLOv8 Nano
    model.train(
        data="dataset.yaml",  # Dataset configuration file
        epochs=epochs,
        imgsz=640,
        batch=16,
        lr0=0.0001,  # Lower learning rate for better optimization
        mosaic=True,  # Enable mosaic augmentation
        project="dataset/results",
        name="run1",
        exist_ok=True
    )
    print("Model training completed.")


def test_model(model_path, dataset_path):
    model = YOLO(model_path)
    test_images_path = os.path.join(dataset_path, "test", "images")
    results = model.predict(source=test_images_path, save=True)
    print("Testing completed. Results saved.")
    return results
