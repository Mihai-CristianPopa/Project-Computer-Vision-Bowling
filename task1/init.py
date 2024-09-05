import os
import cv2
import numpy as np

def extract_pin_positions_from_templates(template_dir):
    """
    Extracts pin positions from full configuration template images.

    Args:
        template_dir: Directory containing template images for different tracks.

    Returns:
        A dictionary where keys are camera IDs and values are lists of pin positions (x, y coordinates).
    """

    pin_positions_by_camera = {}

    # Iterate through template images
    for filename in os.listdir(template_dir):
        if filename.startswith("lane") and filename.endswith(".jpg"):
            lane_id = filename.split(".")[0]  # Extract lane ID from filename (e.g., "lane1")

            # Read template image
            img_path = os.path.join(template_dir, filename)
            img = cv2.imread(img_path)

            # Convert to grayscale (if not already)
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img

            # Apply preprocessing
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)

            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Filter contours
            pin_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]

            # Compute pin positions
            pin_positions = []
            for cnt in pin_contours:
                M = cv2.moments(cnt)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                pin_positions.append((cx, cy))

            # Sort pin positions (optional)
            pin_positions.sort()

            pin_positions_by_camera[lane_id] = pin_positions  # Use lane_id as key

    return pin_positions_by_camera

def visualize_pin_positions(template_dir, pin_positions_by_camera):
    """
    Visualizes the extracted pin positions on the template images for validation.

    Args:
        template_dir: Directory containing template images.
        pin_positions_by_camera: Dictionary of pin positions for each camera.
    """

    for filename in os.listdir(template_dir):
        if filename.startswith("lane") and filename.endswith(".jpg"):
            lane_id = filename.split(".")[0]
            img_path = os.path.join(template_dir, filename)
            img = cv2.imread(img_path)

            pin_positions = pin_positions_by_camera[lane_id]

            for i, (x, y) in enumerate(pin_positions):
                cv2.circle(img, (x, y), 5, (0, 0, 255), -1)  # Draw red circle at pin position
                cv2.putText(img, str(i + 1), (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Display pin number

            cv2.imshow(f"Pin Positions - {lane_id}", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

def classify_pin_positions(image_path, template_dir, pin_positions, threshold=0.8):
    """
    Classifies pin positions in a bowling image using template matching,
    considering multiple template tracks.

    Args:
        image_path: Path to the input image.
        template_dir: Directory containing template images for different tracks.
        pin_positions: List of pin positions (x, y coordinates).
        threshold: Similarity threshold for template matching.

    Returns:
        List of pin states (0 for empty, 1 for occupied).
    """

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    best_match_score = 0
    best_match_states = None

    # Iterate over template tracks
    for template_file in os.listdir(template_dir):
        template_path = os.path.join(template_dir, template_file)
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

        pin_states = []
        for (x, y) in pin_positions:
            roi = img[y-20:y+20, x-20:x+20] 
            res = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            if max_val >= threshold:
                pin_states.append(1) 
            else:
                pin_states.append(0) 

        # Keep track of the best matching template
        total_match_score = sum(pin_states)  # More pins detected is better
        if total_match_score > best_match_score:
            best_match_score = total_match_score
            best_match_states = pin_states

    return best_match_states

# Example usage
image_path = "F:/Master/An1/sem2/cv/Project-Computer-Vision-Bowling/data/train/Task1/01.jpg"
template_dir = "F:/Master/An1/sem2/cv/Project-Computer-Vision-Bowling/data/train/Task1/full-configuration-templates"  # Directory containing template tracks

pin_positions_by_camera = extract_pin_positions_from_templates(template_dir)
# Call the visualization function after sorting the pin positions
visualize_pin_positions(template_dir, pin_positions_by_camera)

# pin_positions = [(100, 200), (150, 250), ...] 

# pin_states = classify_pin_positions(image_path, template_dir, pin_positions)

# Write results to file
# with open('pin_states.txt', 'w') as f:
#     for i, state in enumerate(pin_states):
#         f.write(f"{i+1} {state}\n")