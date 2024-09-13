import cv2
import numpy as np

def resize_image(image, scale_percent=50):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dimensions = (width, height)
    return cv2.resize(image, dimensions, interpolation=cv2.INTER_AREA)

def detect_pattern(image):
    # Resize image to speed up processing
    image = resize_image(image, scale_percent=50)
    
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding to isolate the pattern
    adaptive_thresh = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Apply GaussianBlur to reduce noise
    blurred_image = cv2.GaussianBlur(adaptive_thresh, (3, 3), 0)  # Smaller kernel size
    
    # Find contours from the edges
    contours, _ = cv2.findContours(blurred_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # Use RETR_LIST
    
    # Create a blank mask and draw contours on it
    pattern_mask = np.zeros_like(gray_image, dtype=np.uint8)
    cv2.drawContours(pattern_mask, contours, -1, (255), thickness=cv2.FILLED)
    
    return pattern_mask

def transform_color(hsv_image, blindness_type):
    if blindness_type == 'protanopia':
        hsv_image[..., 0] = np.where((hsv_image[..., 0] < 10) | (hsv_image[..., 0] > 160), 30, hsv_image[..., 0])
    elif blindness_type == 'deuteranopia':
        hsv_image[..., 0] = np.where((35 < hsv_image[..., 0]) & (hsv_image[..., 0] < 85), 160, hsv_image[..., 0])
    elif blindness_type == 'tritanopia':
        hsv_image[..., 0] = np.where((100 < hsv_image[..., 0]) & (hsv_image[..., 0] < 140), 60, hsv_image[..., 0])
    return hsv_image

def detect_and_transform_color(image, pattern_mask, blindness_type):
    # Check if the pattern mask is empty
    if np.count_nonzero(pattern_mask) == 0:
        raise ValueError("Pattern mask is empty. No pattern detected.")
    
    # Convert the pattern mask to a 3-channel mask
    pattern_mask_3ch = cv2.merge([pattern_mask] * 3)
    
    # Apply the pattern mask to the original image to extract the pattern area
    pattern_area = cv2.bitwise_and(image, pattern_mask_3ch)
    
    # Convert the pattern area to HSV color space for easier color manipulation
    hsv_image = cv2.cvtColor(pattern_area, cv2.COLOR_BGR2HSV)
    
    # Apply the color transformation to the HSV image
    hsv_image = hsv_image.astype(np.float32)
    transformed_hsv = transform_color(hsv_image, blindness_type)
    
    # Ensure the values are within the valid range
    transformed_hsv = np.clip(transformed_hsv, 0, 255).astype(np.uint8)
    
    # Convert back to BGR color space
    transformed_pattern_area = cv2.cvtColor(transformed_hsv, cv2.COLOR_HSV2BGR)
    
    # Create the final image by combining the transformed pattern area with the original image
    background = cv2.bitwise_and(image, cv2.bitwise_not(pattern_mask_3ch))
    result_image = cv2.add(background, transformed_pattern_area)
    
    return result_image
