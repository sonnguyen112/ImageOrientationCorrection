import cv2
import numpy as np
from utils import DataGenerator, rotate, generate_rotated_image, rotations, crop_largest_rectangle, angle_error
import json
import base64
from openai import OpenAI
from PIL import Image
from keras.models import load_model

# Initialize OpenAI client
client = OpenAI()

# Load pre-trained model
model_path = r'models\rotnet_indoor_resnet50.keras'
model = load_model(model_path, custom_objects={'angle_error': angle_error})

def combine_images(img1, img2, padding=10, padding_color=(255, 255, 255)):
    """
    Combine two images side-by-side with optional padding in between.

    Args:
        img1 (ndarray): First image (BGR format).
        img2 (ndarray): Second image (BGR format).
        padding (int): Padding between images.
        padding_color (tuple): RGB color for padding.

    Returns:
        ndarray: Combined image as a NumPy array.
    """
    # Convert cv2 images to PIL images
    img1 = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    img2 = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

    # Get dimensions for each image
    img1_width, img1_height = img1.size
    img2_width, img2_height = img2.size

    # Calculate combined dimensions
    combined_width = img1_width + img2_width + padding
    combined_height = max(img1_height, img2_height)

    # Create a new blank image with the combined dimensions
    combined_img = Image.new("RGB", (combined_width, combined_height), padding_color)

    # Paste images side-by-side
    combined_img.paste(img1, (0, 0))
    combined_img.paste(img2, (img1_width + padding, 0))

    return np.array(combined_img)

def encode_image(img):
    """
    Encode an image to base64 format.

    Args:
        img (ndarray): Image to encode.

    Returns:
        str: Base64 encoded image string.
    """
    return base64.b64encode(cv2.imencode('.jpg', img)[1]).decode('utf-8')

def choose_image_by_gpt(rotate_1, rotate_2):
    """
    Use GPT to determine which of two rotated images is upright.

    Args:
        rotate_1 (ndarray): First rotated image.
        rotate_2 (ndarray): Second rotated image.

    Returns:
        dict: JSON response indicating the upright image.
    """
    # Encode images to base64
    base64_image_1 = encode_image(rotate_1)
    base64_image_2 = encode_image(rotate_2)

    # Example JSON responses
    example_json_0 = {"upright_img": 0}
    example_json_1 = {"upright_img": 1}

    # Construct prompt for GPT
    prompt = f'Given two images of indoor scenes, determine which one is upright. ' \
             f'If the first image is upright, return JSON like {json.dumps(example_json_0)}; ' \
             f'if the second image is upright, return JSON like {json.dumps(example_json_1)}.'

    # Send request to GPT
    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "Provide output in valid JSON. The data schema should be like this: " + json.dumps(example_json_0)},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url":  f"data:image/jpeg;base64,{base64_image_1}", "detail": "high"}},
                {"type": "image_url", "image_url": {"url":  f"data:image/jpeg;base64,{base64_image_2}", "detail": "high"}}
            ]},
        ]
    )

    # Parse and return response
    finish_reason = response.choices[0].finish_reason
    if finish_reason == "stop":
        response = json.loads(response.choices[0].message.content)
        return response
    else:
        return None

def correct_image_by_line(img, crop=True):
    """
    Correct image orientation by detecting lines.

    Args:
        img (PIL.Image): Input image.
        crop (bool): Whether to crop the image to the largest rectangle.

    Returns:
        PIL.Image: Corrected image.
    """
    # Convert PIL image to OpenCV format
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    cv2.imwrite("steps/edges.jpg", edges)

    # Detect lines using Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 10, minLineLength=100, maxLineGap=10)
    if lines is None:
        return None

    # Calculate angles of detected lines
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
        angle = int(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        angles.append(angle)

    # Split angles into negative and positive
    negative_angles = [x for x in angles if x < 0]
    positive_angles = [x for x in angles if x > 0]
    mean_negative = np.mean(negative_angles) if negative_angles else 0
    mean_positive = np.mean(positive_angles) if positive_angles else 0

    # Generate rotated images based on mean angles
    rotate_1 = generate_rotated_image(img, mean_negative)
    rotate_2 = generate_rotated_image(img, mean_positive)

    # Crop to largest rectangle if needed
    size = (img.shape[0], img.shape[1])
    rotate_1_crop = crop_largest_rectangle(rotate_1, mean_negative, *size)
    rotate_2_crop = crop_largest_rectangle(rotate_2, mean_positive, *size)

    # Use GPT to choose best orientation
    choose_img = choose_image_by_gpt(rotate_1_crop, rotate_2_crop)
    if crop:
        rotate_1 = rotate_1_crop
        rotate_2 = rotate_2_crop

    # Return image based on GPT choice
    if choose_img is None:
        return None
    else:
        if choose_img["upright_img"] == 0:
            return Image.fromarray(cv2.cvtColor(rotate_1, cv2.COLOR_BGR2RGB))
        else:
            return Image.fromarray(cv2.cvtColor(rotate_2, cv2.COLOR_BGR2RGB))

def correct_img_by_rotnet(img_path, crop=True):
    """
    Correct image orientation using a pre-trained RotNet model.

    Args:
        img_path (str): Path to the input image.
        crop (bool): Whether to crop the image to the largest rectangle.

    Returns:
        PIL.Image: Corrected image.
    """
    # Load image using OpenCV
    image = cv2.imread(img_path)

    # Predict rotation angle using RotNet model
    predictions = model.predict_generator(
        DataGenerator(
            np.expand_dims(image, axis=0),
            input_shape=(224, 224, 3),
            batch_size=1,
            one_hot=True,
            preprocess_func=None,
            rotate=False,
            crop_largest_rect=True,
            crop_center=True
        ),
    )

    # Rotate the image to correct orientation
    predicted_angle = rotations[int(np.argmax(predictions, axis=1))]
    rotated_image = rotate(image, -predicted_angle)

    # Crop to largest rectangle if needed
    if crop:
        size = (image.shape[0], image.shape[1])
        rotated_image = crop_largest_rectangle(rotated_image, -predicted_angle, *size)

    # Convert image to RGB and return as PIL Image
    rotated_image = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rotated_image)

