import os
import json
import folium
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from osgeo import gdal


def model_evaluation(results, dataset_path):
    if not results:
        print("No results to evaluate.")
        return {"Performance Summary": {}}

    detailed_results = []

    for result in results:
        image_path = result.path  # Path to the image
        image_name = os.path.basename(image_path)
        label_file = os.path.splitext(image_name)[0] + ".txt"
        label_path = os.path.join(dataset_path, "test", "labels", label_file)
        is_true = os.path.exists(label_path) and os.path.getsize(label_path) > 0

        boxes = result.boxes
        if boxes is not None and len(boxes.data) > 0:
            for box in boxes.data:
                x_center, y_center, width, height, confidence, class_id = box.tolist()

                # Convert pixel coordinates to geographic coordinates
                ds = gdal.Open(image_path)
                geo_transform = ds.GetGeoTransform()
                lon = geo_transform[0] + x_center * geo_transform[1]
                lat = geo_transform[3] + y_center * geo_transform[5]

                detailed_results.append({
                    "name": image_name,
                    "mine_detected": "yes" if class_id == 0 else "no",
                    "mine_present": "yes" if is_true else "no",
                    "x": x_center,
                    "y": y_center,
                    "lat": lat,
                    "lon": lon,
                    "prob": confidence*100,
                    "conf": confidence
                })
        else:
            detailed_results.append({
                "name": image_name,
                "mine_detected": "no",
                "mine_present": "yes" if is_true else "no",
                "x": None,
                "y": None,
                "lat": None,
                "lon": None,
                "prob": None,
                "conf": None
            })

    with open("result.json", "w") as f:
        json.dump(detailed_results, f, indent=4)

    print("Evaluation Metrics and Details saved to result.json")
    return detailed_results


def analyze_results(result_file):
    # Load results
    with open(result_file, "r") as f:
        results = json.load(f)

    y_true = []
    y_pred = []
    probs = []

    for item in results:
        y_true.append(1 if item["mine_present"] == "yes" else 0)
        y_pred.append(1 if item["mine_detected"] == "yes" else 0)
        if item["mine_detected"] == "yes":
            probs.append(item["prob"])

    # Calculate metrics
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    # Print metrics
    print("Model Performance Metrics:")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")

    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(cm)

    # Additional statistics for detection probabilities
    if probs:
        print("\nDetection Probability Statistics:")
        print(f"Mean Probability: {sum(probs) / len(probs):.2f}")
        print(f"Max Probability: {max(probs):.2f}")
        print(f"Min Probability: {min(probs):.2f}")


def visualize_detected_mines(result_file, map_output):
    # Load results from result.json
    with open(result_file, "r") as f:
        results = json.load(f)

    # Initialize a map centered around Paris
    paris_coords = [48.8566, 2.3522]
    detection_map = folium.Map(location=paris_coords, zoom_start=12)

    # Add markers for detected mines
    for item in results:
        if item["mine_detected"] == "yes" and item["lat"] is not None and item["lon"] is not None:
            # Marker for detected mine
            popup_text = f"""
            <strong>Image:</strong> {item['name']}<br>
            <strong>Probability:</strong> {item['prob']:.2f}<br>
            <strong>Confidence:</strong> {item['conf']:.2f}
            """
            folium.Marker(
                location=[item["lat"], item["lon"]],
                popup=popup_text,
                icon=folium.Icon(color="red", icon="exclamation-circle"),
            ).add_to(detection_map)

    # Save the map to an HTML file
    detection_map.save(map_output)
    print(f"\nMap with detected mines saved to {map_output}")


