import torchvision.models as models
import torch.nn as nn
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import segmentation_models_pytorch as smp
import numpy as np
import cv2
import torch.nn.functional as F
import io
import zipfile
import os
from ultralytics import YOLO


# Define models
# UNet model definition
class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()

        def CBR(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        self.enc1 = CBR(3, 64)
        self.enc2 = CBR(64, 128)
        self.enc3 = CBR(128, 256)
        self.enc4 = CBR(256, 512)
        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = CBR(512, 1024)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = CBR(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = CBR(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = CBR(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = CBR(128, 64)
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        bottleneck = self.bottleneck(self.pool(enc4))
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)
        return self.final(dec1)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        return out


class LinkNet(nn.Module):
    def __init__(self, num_classes=4):
        super(LinkNet, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.in_block = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.decoder4 = BasicBlock(512, 256)
        self.decoder3 = BasicBlock(256, 128)
        self.decoder2 = BasicBlock(128, 64)
        self.decoder1 = BasicBlock(64, 64)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x = self.in_block(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder
        d4 = F.interpolate(self.decoder4(e4), scale_factor=2, mode='bilinear', align_corners=False) + e3
        d3 = F.interpolate(self.decoder3(d4), scale_factor=2, mode='bilinear', align_corners=False) + e2
        d2 = F.interpolate(self.decoder2(d3), scale_factor=2, mode='bilinear', align_corners=False) + e1
        d1 = F.interpolate(self.decoder1(d2), scale_factor=2, mode='bilinear', align_corners=False)

        out = self.final_conv(d1)
        return out


# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 4


# Define the path to your zip file
model_zip_path = 'model_weights.zip'

# Extract the zip file if it exists
if os.path.exists(model_zip_path):
    with zipfile.ZipFile(model_zip_path, 'r') as zip_ref:
        zip_ref.extractall("model_directory")  # Extract to a specific directory
else:
    raise FileNotFoundError(f"{model_zip_path} does not exist.")

# Load pretrained models
unet_path = os.path.join("model_directory", "unet_model(multi).pth")
fpn_path = os.path.join("model_directory", "FPN_model(multi)_out.pth")
link_path = os.path.join("model_directory", "linknet_multiclass(new)_out.pth")

# Function to check and load model
def load_model(model_path, model_class, num_classes, device):
    if os.path.exists(model_path):
        print(f"Model file found at: {model_path}")
        model = model_class(num_classes=num_classes)  # Instantiate the model
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device).eval()  # Move to device and set to eval mode
        return model
    else:
        raise FileNotFoundError(f"{model_path} does not exist after extraction.")

# Check and load each model
try:
    unet_model = load_model(unet_path, UNet, num_classes, device)
    fpn_model = load_model(fpn_path, smp.FPN, num_classes, device)
    linknet_model = load_model(link_path, LinkNet, num_classes, device)

    # Store the models in a dictionary for later use
    models = {
        "UNet": unet_model,
        "FPN": fpn_model,
        "LinkNet": linknet_model
    }

except FileNotFoundError as e:
    print(e)


# Image preprocessing function
def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image


# Convert prediction mask to a color-coded image
def class_to_color(mask):
    color_map = {
        0: [0, 0, 0],  # Background
        1: [0, 255, 0],  # Sidewalk (Green)
        2: [255, 0, 0],  # Road (Red)
        3: [0, 0, 255]  # Crosswalk (Blue)
    }

    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    for class_idx, color in color_map.items():
        color_mask[mask == class_idx] = color

    return color_mask


def add_transparent_legend():
    """
    Generates a transparent legend as an image with labels and colors
    """
    # Create an image for the legend with transparent background (RGBA mode)
    legend_image = Image.new("RGBA", (150, 150), (0, 0, 0, 0))  # Transparent background
    draw = ImageDraw.Draw(legend_image)

    labels = [
        ("Sidewalk", (0, 255, 0)),  # Green
        ("Crosswalk", (0, 0, 255)),  # Blue
        ("Road", (255, 0, 0)),  # Red
    ]

    # Font for the legend text
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except IOError:
        font = ImageFont.load_default()

    for idx, (label, color) in enumerate(labels):
        draw.rectangle(
            [10, 10 + idx * 40, 30, 30 + idx * 40],
            fill=color,
        )
        draw.text(
            (40, 10 + idx * 40),
            label,
            font=font,
            fill=(255, 255, 255),  # Black text for labels
        )

    return legend_image


def combine_images(original_image, mask):
    mask_color = class_to_color(mask)
    mask_image = Image.fromarray(mask_color)
    combined = Image.blend(original_image.resize(mask_image.size), mask_image, alpha=0.5)
    return combined


def dynamic_radius(image_shape):
    """
    Dynamically calculate the radius based on the image dimensions.
    Example: 1% of the smaller dimension (height or width).
    """
    height, width = image_shape[:2]  # Extract height and width of the image
    return int(min(width, height) * 0.01)  # Use 1% of the smaller dimension


# def find_contour_intersection(sidewalk_mask, crosswalk_mask, radius=None):
#     # Dynamically adjust radius if not provided
#     if radius is None:
#         radius = dynamic_radius(sidewalk_mask.shape)
#
#     # Use Canny edge detection to find edges in the sidewalk and crosswalk masks
#     sidewalk_edges = cv2.Canny(sidewalk_mask.astype(np.uint8) * 255, 100, 200)
#     crosswalk_edges = cv2.Canny(crosswalk_mask.astype(np.uint8) * 255, 100, 200)
#
#     # Find contours from the edge-detected images
#     contours_sidewalk, _ = cv2.findContours(sidewalk_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     contours_crosswalk, _ = cv2.findContours(crosswalk_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     intersections = []
#
#     # Loop through contours to find intersections
#     for s_contour in contours_sidewalk:
#         for c_contour in contours_crosswalk:
#             for s_point in s_contour:
#                 for c_point in c_contour:
#                     s_x, s_y = s_point[0]
#                     c_x, c_y = c_point[0]
#                     # Check if distance between points is smaller than the radius
#                     if np.linalg.norm(np.array([s_x - c_x, s_y - c_y])) < radius:
#                         intersections.append((s_x, s_y))
#
#     if intersections:
#         # Compute the centroid of the intersection points
#         centroid_x = np.mean([pt[0] for pt in intersections])
#         centroid_y = np.mean([pt[1] for pt in intersections])
#         return intersections, (int(centroid_x), int(centroid_y))
#
#     return [], None

def find_contour_intersection(sidewalk_mask, crosswalk_mask, radius=None):
    # Dynamically adjust radius if not provided
    if radius is None:
        radius = dynamic_radius(sidewalk_mask.shape)

    # Use Canny edge detection to find edges in the sidewalk and crosswalk masks
    sidewalk_edges = cv2.Canny(sidewalk_mask.astype(np.uint8) * 255, 100, 200)
    crosswalk_edges = cv2.Canny(crosswalk_mask.astype(np.uint8) * 255, 100, 200)

    # Find contours from the edge-detected images
    contours_sidewalk, _ = cv2.findContours(sidewalk_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_crosswalk, _ = cv2.findContours(crosswalk_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    intersections = []

    # Loop through contours to find a single representative intersection
    for s_contour in contours_sidewalk:
        for c_contour in contours_crosswalk:
            for s_point in s_contour:
                for c_point in c_contour:
                    s_x, s_y = s_point[0]
                    c_x, c_y = c_point[0]
                    # Check if distance between points is smaller than the radius
                    if np.linalg.norm(np.array([s_x - c_x, s_y - c_y])) < radius:
                        intersections.append((s_x, s_y))
                        break  # Stop searching once one intersection is found per contour pair
                if intersections:
                    break
            if intersections:
                break

    return intersections  # Return the list of single intersections


def save_combined_image(image, mask, save_path):
    # Save the image and mask combined (superimposed) as a PNG file
    combined_image = combine_images(image, mask)
    combined_image.save(save_path)


def export_images(original_image, mask):
    # Combine the original image and the mask
    combined_image = combine_images(original_image, mask)

    # Create an in-memory zip file
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zf:
        # Add original image to the zip
        original_img_byte_arr = io.BytesIO()
        original_image.save(original_img_byte_arr, format="PNG")
        zf.writestr("original_image.png", original_img_byte_arr.getvalue())

        # Add predicted mask (combined image) to the zip
        prediction_img_byte_arr = io.BytesIO()
        combined_image.save(prediction_img_byte_arr, format="PNG")
        zf.writestr("prediction_image.png", prediction_img_byte_arr.getvalue())

    # Make sure the buffer is at the beginning
    zip_buffer.seek(0)

    return zip_buffer


def analyze_mask(pred_mask, class_ids):
    """
    Analyze the predicted mask to check for the presence of different features.
    :param pred_mask: The predicted mask array.
    :param class_ids: A dictionary mapping feature names to class IDs.
    :return: A summary of present and missing features.
    """
    summary = []

    # Iterate over each feature (e.g., sidewalk, crosswalk, road)
    for feature, class_id in class_ids.items():
        if np.any(pred_mask == class_id):
            color = "green" if feature == "Sidewalk" else "blue" if feature == "Crosswalk" else "red"
            summary.append(f"{feature} is present and represented with {color}.")
        else:
            summary.append(f"{feature} is missing.")

    return summary


def display_summary(prediction_mode, pred_mask, class_ids):
    if prediction_mode == "Predict All":
        st.write("The segmentation results indicate the presence of the following features:")
        # Analyze for all features and display a comprehensive summary
        summary = analyze_mask(pred_mask, class_ids)
        for item in summary:
            st.write(item)

    elif prediction_mode == "Sidewalk":
        st.write("Conducting Sidewalk Detection...")
        # Analyze only for the sidewalk
        if np.any(pred_mask == class_ids["Sidewalk"]):
            st.write("Sidewalk successfully detected and visualized in green.")
        else:
            st.write("No sidewalk detected in the segmented area.")

    elif prediction_mode == "Crosswalk":
        st.write("Performing Crosswalk Identification...")
        # Analyze only for the crosswalk
        if np.any(pred_mask == class_ids["Crosswalk"]):
            st.write("Crosswalk successfully identified and highlighted in blue.")
        else:
            st.write("No crosswalk detected within the analyzed area.")

    elif prediction_mode == "Road":
        st.write("Analyzing Road Presence...")
        # Analyze only for the road
        if np.any(pred_mask == class_ids["Road"]):
            st.write("Road successfully detected and segmented in red.")
        else:
            st.write("No road detected within the analyzed area.")


# Define the class IDs for each feature
class_ids = {
    "Sidewalk": 1,
    "Crosswalk": 2,
    "Road": 3,
}


def display_help():
    st.markdown("### User Guidance")
    st.markdown("""
        **Step 1: Upload an Image**
        - Click on the upload button to select an image file (JPG or PNG) from your device.
    
        **Step 2: Select Features to Predict**
        - Choose the features you want the model to predict: 
            - **Predict All**: Get predictions for all features.
            - **Sidewalk**: Get predictions specifically for sidewalks.
            - **Crosswalk**: Get predictions specifically for crosswalks.
            - **Road**: Get predictions specifically for roads.
    
        **Step 3: Choose a Model**
        - Select one of the available models (UNet, FPN, or LinkNet) to perform the segmentation.
    
        **Step 4: Show Suggestions**
        - Click this button to see suggestions based on the model’s predictions. The suggestions will indicate the optimal positions for signals.
    
        **Step 5: Adjust Model Confidence**
        - Use the slider to set the confidence threshold for the model's predictions. Only predictions above this threshold will be displayed.
    
        **Note**: Make sure to upload a suitable image for accurate predictions. The output will show the original image with the predicted masks superimposed.
        """)


def main():
    with st.sidebar:
        if st.button("Overview"):
            st.markdown("""
            ### Summary

            This roadway features segmentation tool utilizes deep learning models (UNet, FPN, LinkNet) 
            to accurately detect and visualize sidewalks, crosswalks, and roads in images.

            ### Purpose

            The primary objective is to assist urban planners, engineers, and researchers 
            in analyzing and optimizing infrastructure for enhanced safety and accessibility.

            ### Contributions

            This project contributes to the development of intelligent transportation systems 
            by providing a user-friendly interface for:

            1. Image segmentation
            2. Feature detection
            3. Suggestion of optimal signal installation points

            ### Future Work

            Expanding the model's capabilities to include:

            1. Real-time video analysis
            2. Integration with geographic information systems (GIS)
            3. Multi-city dataset support
            """, unsafe_allow_html=True)
    # Streamlit selection for task
    task = st.sidebar.selectbox("Choose a task", ("Segmentation", "Object Detection"))

    if task == "Segmentation":

        # Code specific to segmentation task
        # st.title("Segmentation Task")
        # st.title("Roadway Features Segmentation")
        # Initialize session state
        if 'show_help' not in st.session_state:
            st.session_state.show_help = False

        # Button to toggle help
        if st.button("Help"):
            st.session_state.show_help = True

        if st.session_state.show_help:
            display_help()
            # Add a close button
            if st.button("Close"):
                st.session_state.show_help = False

        st.markdown("<h1 style='color: orange;'>Highlighted Road Features</h1>", unsafe_allow_html=True)

        st.sidebar.markdown("<h2 style='color: red;'>Configuration</h2>", unsafe_allow_html=True)
        uploaded_file = st.sidebar.file_uploader("Step 1: Upload image here", type=["jpg", "png", "jpeg"])
        prediction_mode = st.sidebar.selectbox("STEP 2: Features to Predict",
                                               ["Predict All", "Sidewalk", "Crosswalk", "Road"])
        model_name = st.sidebar.selectbox("STEP 3: Choose Model", ["UNet", "FPN", "LinkNet"])
        show_suggestion = st.sidebar.button("STEP 4: Show Suggestions")

        confidence_threshold = st.sidebar.slider("STEP 5: Model Accuracy", min_value=0.0, max_value=1.0, value=0.5,
                                                 step=0.1)



        # Background image path
        background_image_path = "/home/kite/Downloads/StreetDesign.png"

        if uploaded_file is None:
            st.image(background_image_path, use_column_width=True)
        else:
            image = Image.open(uploaded_file).convert("RGB")

            # Hide background image for prediction
            st.empty()  # Clear the background image display

            # Preprocess image
            preprocessed_image = preprocess_image(image)

            # Predict based on model
            model = models[model_name]

            with torch.no_grad():
                output = model(preprocessed_image.to(device))
                softmax_output = F.softmax(output, dim=1)  # Apply softmax to get probabilities
                max_probs, pred_mask = torch.max(softmax_output, dim=1)
                pred_mask = pred_mask.cpu().numpy()[0]
                max_probs = max_probs.cpu().numpy()[0]

                # Filter low-confidence predictions
                pred_mask[max_probs < confidence_threshold] = 0

                # Filter prediction based on selected mode
                if prediction_mode == "Predict All":
                    combined_image = combine_images(image, pred_mask)
                else:
                    filtered_mask = np.zeros_like(pred_mask)
                    if prediction_mode == "Sidewalk":
                        filtered_mask[pred_mask == 1] = 1
                    elif prediction_mode == "Crosswalk":
                        filtered_mask[pred_mask == 3] = 3
                    elif prediction_mode == "Road":
                        filtered_mask[pred_mask == 2] = 2
                    combined_image = combine_images(image, filtered_mask)

                # Extract specific masks for sidewalk and crosswalk for intersection calculation
                sidewalk_mask = (pred_mask == 1).astype(np.uint8)
                crosswalk_mask = (pred_mask == 3).astype(np.uint8)

                # Find contour intersections
                # Find intersections
                intersection_points = find_contour_intersection(sidewalk_mask, crosswalk_mask)

            if show_suggestion and intersection_points:
                icon_path = "/home/kite/Downloads/traffic-signal.png"
                signal_icon = Image.open(icon_path).convert("RGBA")
                icon_size = 50
                signal_icon = signal_icon.resize((icon_size, icon_size))

                for point in intersection_points:
                    s_x, s_y = point
                    icon_position = (s_x - icon_size // 2, s_y - icon_size // 2)
                    image.paste(signal_icon, icon_position, signal_icon)
                    break  # Only display one intersection per area

                # Add text for the suggestion
                if intersection_points:
                    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
                    font = ImageFont.truetype(font_path, size=15)
                    draw = ImageDraw.Draw(image)
                    first_point = intersection_points[0]
                    text_position = (first_point[0] + icon_size + 10, first_point[1] - 10)
                    draw.text(text_position, "Signal Installation Recommended", fill="red", font=font)

            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original Image (with suggestion)", use_column_width=True)
            with col2:
                st.image(combined_image, caption=f"Segmented {prediction_mode}", use_column_width=True)

            if intersection_points:
                st.write(f"Suggested Installation Points at: {intersection_points}")

            # Create legend image
            legend_image = add_transparent_legend()
            st.image(legend_image, caption=None, use_column_width=False)

            # Button to trigger the summary
            if st.button("Show Summary for Selected Feature"):
                display_summary(prediction_mode, pred_mask, class_ids)

        # Export prediction
        if st.sidebar.button("Export Prediction"):
            zip_buffer = export_images(image, pred_mask)
            st.download_button(
                label="Download Prediction",
                data=zip_buffer,
                file_name="prediction.zip",
                mime="application/zip"
            )

    elif task == "Object Detection":
        # Function to load an image and convert it for OpenCV processing
        def load_image(image_file):
            img = Image.open(image_file)
            return np.array(img)

        # Function to run YOLOv8 model for object detection
        def detect_with_yolov8(image, model_path):
            model = YOLO(model_path)
            results = model.predict(image, device='cpu')
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    label = result.names[int(box.cls)]
                    confidence = box.conf[0] * 100  # Convert to percentage
                    # Draw bounding box and label
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image, f'{label}: {confidence:.2f}%', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            return image

        # Function to run Faster RCNN for object detection (placeholder)
        def detect_with_faster_rcnn(image):
            # Implement the Faster RCNN detection logic here
            # This is a placeholder function
            return image

        # Streamlit UI
        st.title("Object Detection App")
        st.sidebar.title("Settings")

        # Sidebar options for object detection
        uploaded_file = st.sidebar.file_uploader("Upload an Image", type=['png', 'jpg', 'jpeg'])
        model_choice = st.sidebar.selectbox("Choose Model", ["YOLOv8", "Faster RCNN"])

        if uploaded_file is not None:
            # Load and display original image
            original_image = load_image(uploaded_file)
            st.image(original_image, caption='Uploaded Image', use_column_width=True)

            if st.sidebar.button("Detect"):
                # Perform object detection
                if model_choice == "YOLOv8":
                    # YOLOv8 detection
                    yolov8_model_path = '/home/kite/Downloads/yolov8_model (1).pt'
                    detected_image = detect_with_yolov8(original_image.copy(), yolov8_model_path)
                elif model_choice == "Faster RCNN":
                    # Faster RCNN detection
                    detected_image = detect_with_faster_rcnn(original_image.copy())

                # Display images side by side
                col1, col2 = st.columns(2)

                with col1:
                    st.image(original_image, caption='Original Image', use_column_width=True)

                with col2:
                    st.image(detected_image, caption='Detected Image', use_column_width=True)


# Add the other elif options for segmentation if needed.

if __name__ == "__main__":
    main()
