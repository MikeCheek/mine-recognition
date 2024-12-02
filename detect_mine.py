from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk, ImageDraw, ImageFont


# Load the trained YOLO model
trained_model_path = "results/train_run/weights/best.pt"  # Path to the trained model
model = YOLO(trained_model_path)


# Function to resize an image while maintaining its aspect ratio
def resize_image(image, max_width, max_height):
    width, height = image.size
    aspect_ratio = width / height

    if width > max_width:
        width = max_width
        height = int(width / aspect_ratio)

    if height > max_height:
        height = max_height
        width = int(height * aspect_ratio)

    return image.resize((width, height), Image.Resampling.LANCZOS)


# Function to perform object detection and display results
def detect_objects():
    # Clear previous images and results
    panel_original.config(image="")
    panel_original.image = None
    panel_processed.config(image="")
    panel_processed.image = None
    results_label.config(text="")

    # Open file dialog to select an image
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")]
    )

    if not file_path:
        return  # Exit if no file selected

    # Load the original image for display
    original_image = cv2.imread(file_path)
    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    original_image_pil = Image.fromarray(original_image_rgb)

    # Resize the original image to fit the GUI
    original_image_pil_resized = resize_image(original_image_pil, max_width=400, max_height=300)

    # Perform object detection
    results = model.predict(source=file_path, save=False, conf=0.25)  # Adjust confidence threshold
    detection_results = []
    detected = False

    # Create a copy of the original image for displaying bounding boxes
    processed_image_pil = original_image_pil.copy()
    draw = ImageDraw.Draw(processed_image_pil)

    if results[0].boxes.data is not None and len(results[0].boxes.data) > 0:
        for detection in results[0].boxes.data:
            x1, y1, x2, y2, confidence, class_id = detection.tolist()
            detection_results.append(
                f"MINE DETECTED - Confidence: {confidence:.2f}, Box: ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})"
            )
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, y1 - 10), f"Mine Detected: {confidence:.2f}", fill="red")
            detected = True

    if not detected:
        font = ImageFont.truetype("arial.ttf", 30)
        text = "NO MINE DETECTED"
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        image_width, image_height = processed_image_pil.size
        text_position = ((image_width - text_width) // 2, (image_height - text_height) // 2)
        draw.text(text_position, text, fill="red", font=font)

    processed_image_pil_resized = resize_image(processed_image_pil, max_width=400, max_height=300)

    # Convert images to ImageTk format for displaying in the GUI
    original_image_tk = ImageTk.PhotoImage(original_image_pil_resized)
    processed_image_tk = ImageTk.PhotoImage(processed_image_pil_resized)

    # Update the GUI with the original and processed images
    panel_original.config(image=original_image_tk)
    panel_original.image = original_image_tk
    panel_processed.config(image=processed_image_tk)
    panel_processed.image = processed_image_tk

    # Display detection results below the processed image
    if detection_results:
        results_label.config(text="\n".join(detection_results))
    else:
        results_label.config(text="NO MINE DETECTED")


# Create the GUI window
window = tk.Tk()
window.title("Mine Detection System")
window.geometry("1000x600")
window.configure(bg="#1e1e1e")  # Dark gray background

# Add a header with branding
header_frame = tk.Frame(window, bg="#292929", height=50)
header_frame.pack(fill=tk.X)
header_label = tk.Label(
    header_frame, text="MINE DETECTION SYSTEM", bg="#292929", fg="white", font=("Helvetica", 20, "bold")
)
header_label.pack(pady=10)

# Frame for original and processed images
image_frame = tk.Frame(window, bg="#1e1e1e")
image_frame.pack(pady=20)

panel_original_title = tk.Label(image_frame, text="Original Image", bg="#1e1e1e", fg="white", font=("Helvetica", 12))
panel_original_title.grid(row=0, column=0, padx=20)

panel_processed_title = tk.Label(image_frame, text="Processed Image", bg="#1e1e1e", fg="white", font=("Helvetica", 12))
panel_processed_title.grid(row=0, column=1, padx=20)

panel_original = tk.Label(image_frame, bg="#1e1e1e")
panel_original.grid(row=1, column=0, padx=20)

panel_processed = tk.Label(image_frame, bg="#1e1e1e")
panel_processed.grid(row=1, column=1, padx=20)

# Detection results
results_label = tk.Label(window, text="", bg="#1e1e1e", fg="#00FF00", font=("Helvetica", 14), justify=tk.LEFT)
results_label.pack(pady=20)

# Add a button to trigger object detection
button_frame = tk.Frame(window, bg="#1e1e1e")
button_frame.pack(pady=10)
detect_button = tk.Button(
    button_frame, text="Select Image and Detect", command=detect_objects, font=("Helvetica", 14), bg="#292929", fg="white"
)
detect_button.pack()

# Footer with branding
footer_frame = tk.Frame(window, bg="#292929", height=50)
footer_frame.pack(fill=tk.X, side=tk.BOTTOM)
footer_label = tk.Label(
    footer_frame, text="Advanced Detection Solutions | Confidential", bg="#292929", fg="white", font=("Helvetica", 10)
)
footer_label.pack(pady=5)

# Run the GUI event loop
window.mainloop()
