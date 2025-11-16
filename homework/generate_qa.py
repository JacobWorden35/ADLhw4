import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

# Define object type mapping
OBJECT_TYPES = {
    1: "Kart",
    2: "Track Boundary",
    3: "Track Element",
    4: "Special Element 1",
    5: "Special Element 2",
    6: "Special Element 3",
}

# Define colors for different object types (RGB format)
COLORS = {
    1: (0, 255, 0),  # Green for karts
    2: (255, 0, 0),  # Blue for track boundaries
    3: (0, 0, 255),  # Red for track elements
    4: (255, 255, 0),  # Cyan for special elements
    5: (255, 0, 255),  # Magenta for special elements
    6: (0, 255, 255),  # Yellow for special elements
}

# Original image dimensions for the bounding box coordinates
ORIGINAL_WIDTH = 600
ORIGINAL_HEIGHT = 400


def extract_frame_info(image_path: str) -> tuple[int, int]:
    """
    Extract frame ID and view index from image filename.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (frame_id, view_index)
    """
    filename = Path(image_path).name
    # Format is typically: XXXXX_YY_im.png where XXXXX is frame_id and YY is view_index
    parts = filename.split("_")
    if len(parts) >= 2:
        frame_id = int(parts[0], 16)  # Convert hex to decimal
        view_index = int(parts[1])
        return frame_id, view_index
    return 0, 0  # Default values if parsing fails


def draw_detections(
    image_path: str, info_path: str, font_scale: float = 0.5, thickness: int = 1, min_box_size: int = 5
) -> np.ndarray:
    """
    Draw detection bounding boxes and labels on the image.

    Args:
        image_path: Path to the image file
        info_path: Path to the corresponding info.json file
        font_scale: Scale of the font for labels
        thickness: Thickness of the bounding box lines
        min_box_size: Minimum size for bounding boxes to be drawn

    Returns:
        The annotated image as a numpy array
    """
    # Read the image using PIL
    pil_image = Image.open(image_path)
    if pil_image is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Get image dimensions
    img_width, img_height = pil_image.size

    # Create a drawing context
    draw = ImageDraw.Draw(pil_image)

    # Read the info.json file
    with open(info_path) as f:
        info = json.load(f)

    # Extract frame ID and view index from image filename
    _, view_index = extract_frame_info(image_path)

    # Get the correct detection frame based on view index
    if view_index < len(info["detections"]):
        frame_detections = info["detections"][view_index]
    else:
        print(f"Warning: View index {view_index} out of range for detections")
        return np.array(pil_image)

    # Calculate scaling factors
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    # Draw each detection
    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)

        if class_id != 1:
            continue

        # Scale coordinates to fit the current image size
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)

        # Skip if bounding box is too small
        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue

        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue

        # Get color for this object type
        if track_id == 0:
            color = (255, 0, 0)
        else:
            color = COLORS.get(class_id, (255, 255, 255))

        # Draw bounding box using PIL
        draw.rectangle([(x1_scaled, y1_scaled), (x2_scaled, y2_scaled)], outline=color, width=thickness)

    # Convert PIL image to numpy array for matplotlib
    return np.array(pil_image)


def extract_kart_objects(
    info_path: str, view_index: int, img_width: int = 150, img_height: int = 100, min_box_size: int = 5
) -> list:
    """
    Extract kart objects from the info.json file, including their center points and identify the center kart.
    Filters out karts that are out of sight (outside the image boundaries).

    Args:
        info_path: Path to the corresponding info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)

    Returns:
        List of kart objects, each containing:
        - instance_id: The track ID of the kart
        - kart_name: The name of the kart
        - center: (x, y) coordinates of the kart's center
        - is_center_kart: Boolean indicating if this is the kart closest to image center
    """
    # Read the info.json file
    with open(info_path) as f:
        info = json.load(f)
    
    kart_names = info["karts"]
    
    # Get detections for this view
    if view_index >= len(info["detections"]):
        return []
    
    frame_detections = info["detections"][view_index]
    
    # Calculate scaling factors
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT
    
    kart_objects = []
    
    # Extract kart detections
    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)
        
        # Only process karts
        if class_id != 1:
            continue
        
        # Scale coordinates
        x1_scaled = x1 * scale_x
        y1_scaled = y1 * scale_y
        x2_scaled = x2 * scale_x
        y2_scaled = y2 * scale_y
        
        # Skip if bounding box is too small
        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue
        
        # Skip if completely outside image boundaries
        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue
        
        # Calculate center
        center_x = (x1_scaled + x2_scaled) / 2
        center_y = (y1_scaled + y2_scaled) / 2
        
        # Get kart name
        kart_name = kart_names[track_id]
        
        kart_objects.append({
            "instance_id": track_id,
            "kart_name": kart_name,
            "center": (center_x, center_y),
            "is_center_kart": False  # Will be updated below
        })
    
    # Find the kart closest to image center
    if kart_objects:
        img_center = (img_width / 2, img_height / 2)
        min_dist = float('inf')
        center_kart_idx = 0
        
        for idx, kart in enumerate(kart_objects):
            dist = ((kart["center"][0] - img_center[0]) ** 2 + 
                   (kart["center"][1] - img_center[1]) ** 2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                center_kart_idx = idx
        
        kart_objects[center_kart_idx]["is_center_kart"] = True
    
    return kart_objects


def extract_track_info(info_path: str) -> str:
    """
    Extract track information from the info.json file.

    Args:
        info_path: Path to the info.json file

    Returns:
        Track name as a string
    """
    with open(info_path) as f:
        info = json.load(f)
    
    return info["track"]


def generate_qa_pairs(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate question-answer pairs for a given view.

    Args:
        info_path: Path to the info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)

    Returns:
        List of dictionaries, each containing a question and answer
    """
    qa_pairs = []
    
    # Get track info
    track_name = extract_track_info(info_path)
    
    # Get kart objects
    kart_objects = extract_kart_objects(info_path, view_index, img_width, img_height)
    
    # Get ego car name (kart at view_index position)
    with open(info_path) as f:
        info = json.load(f)
    kart_names = info["karts"]
    
    if view_index >= len(kart_names):
        return qa_pairs
    
    ego_car_name = kart_names[view_index]
    
    # Find ego car in kart_objects
    ego_kart = None
    other_karts = []
    for kart in kart_objects:
        if kart["kart_name"] == ego_car_name:
            ego_kart = kart
        else:
            other_karts.append(kart)
    
    # 1. Ego car question
    qa_pairs.append({
        "question": "What kart is the ego car?",
        "answer": ego_car_name
    })
    
    # 2. Total karts question (including ego car)
    total_karts = len(kart_objects)
    qa_pairs.append({
        "question": "How many karts are there in the scenario?",
        "answer": str(total_karts)
    })
    
    # 3. Track information question
    qa_pairs.append({
        "question": "What track is this?",
        "answer": track_name
    })
    
    # If ego car not detected or no other karts, return what we have
    if ego_kart is None or len(other_karts) == 0:
        return qa_pairs
    
    ego_center = ego_kart["center"]
    
    # Counters for position-based counting
    count_left = 0
    count_right = 0
    count_front = 0
    count_behind = 0
    
    # 4. Relative position questions for each other kart
    for kart in other_karts:
        kart_name = kart["kart_name"]
        kart_center = kart["center"]
        
        # Determine left/right (X axis)
        # Lower X = left, Higher X = right
        is_left = kart_center[0] < ego_center[0]
        is_right = kart_center[0] > ego_center[0]
        
        # Determine front/behind (Y axis)
        # In image coordinates, Y increases downward
        # Lower Y (toward top) = in front
        # Higher Y (toward bottom) = behind
        is_front = kart_center[1] < ego_center[1]
        is_behind = kart_center[1] > ego_center[1]
        
        # Update counters
        if is_left:
            count_left += 1
        if is_right:
            count_right += 1
        if is_front:
            count_front += 1
        if is_behind:
            count_behind += 1
        
        # Determine threshold for "significantly" left/right/front/back
        # Use a threshold to avoid classifying karts that are nearly aligned
        x_threshold = 10  # pixels
        y_threshold = 10  # pixels
        
        x_diff = abs(kart_center[0] - ego_center[0])
        y_diff = abs(kart_center[1] - ego_center[1])
        
        # Generate position questions
        # Left/Right question
        if x_diff > y_diff and x_diff > x_threshold:
            lr_answer = "left" if is_left else "right"
            qa_pairs.append({
                "question": f"Is {kart_name} to the left or right of the ego car?",
                "answer": lr_answer
            })
        
        # Front/Behind question
        if y_diff > x_diff and y_diff > y_threshold:
            fb_answer = "front" if is_front else "back"
            qa_pairs.append({
                "question": f"Is {kart_name} in front of or behind the ego car?",
                "answer": fb_answer
            })
        
        # Combined position question (where is X relative to ego)
        if x_diff > x_threshold or y_diff > y_threshold:
            position_parts = []
            if y_diff > y_threshold:
                position_parts.append("front" if is_front else "back")
            if x_diff > x_threshold:
                position_parts.append("left" if is_left else "right")
            
            if position_parts:
                position_answer = " and ".join(position_parts)
                qa_pairs.append({
                    "question": f"Where is {kart_name} relative to the ego car?",
                    "answer": position_answer
                })
    
    # 5. Counting questions
    qa_pairs.append({
        "question": "How many karts are to the left of the ego car?",
        "answer": str(count_left)
    })
    
    qa_pairs.append({
        "question": "How many karts are to the right of the ego car?",
        "answer": str(count_right)
    })
    
    qa_pairs.append({
        "question": "How many karts are in front of the ego car?",
        "answer": str(count_front)
    })
    
    qa_pairs.append({
        "question": "How many karts are behind the ego car?",
        "answer": str(count_behind)
    })
    
    return qa_pairs


def check_qa_pairs(info_file: str, view_index: int):
    """
    Check QA pairs for a specific info file and view index.

    Args:
        info_file: Path to the info.json file
        view_index: Index of the view to analyze
    """
    # Find corresponding image file
    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    # Visualize detections
    annotated_image = draw_detections(str(image_file), info_file)

    # Display the image
    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()

    # Generate QA pairs
    qa_pairs = generate_qa_pairs(info_file, view_index)

    # Print QA pairs
    print("\nQuestion-Answer Pairs:")
    print("-" * 50)
    for qa in qa_pairs:
        print(f"Q: {qa['question']}")
        print(f"A: {qa['answer']}")
        print("-" * 50)


def generate_all_qa_pairs(data_dir: str = "../data/train", output_file: str = None):
    """
    Generate QA pairs for all images in a directory and save to JSON.
    
    Args:
        data_dir: Directory containing the data
        output_file: Output JSON file path (optional)
    """
    from pathlib import Path
    import json
    
    data_path = Path(data_dir)
    all_qa_pairs = []
    
    # Find all info files
    info_files = sorted(data_path.glob("*_info.json"))
    
    print(f"Found {len(info_files)} info files")
    
    for info_file in info_files:
        base_name = info_file.stem.replace("_info", "")
        
        # Process all 10 views for this frame
        for view_index in range(10):
            # Check if image exists
            image_file = data_path / f"{base_name}_{view_index:02d}_im.jpg"
            if not image_file.exists():
                continue
            
            # Generate QA pairs
            qa_pairs = generate_qa_pairs(str(info_file), view_index)
            
            # Add image file path to each QA pair
            for qa in qa_pairs:
                qa["image_file"] = f"train/{image_file.name}"
                all_qa_pairs.append(qa)
    
    print(f"Generated {len(all_qa_pairs)} QA pairs")
    
    # Save to file
    if output_file is None:
        output_file = data_path / "generated_qa_pairs.json"
    else:
        output_file = Path(output_file)
    
    with open(output_file, 'w') as f:
        json.dump(all_qa_pairs, f, indent=2)
    
    print(f"Saved to {output_file}")
    
    return all_qa_pairs


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python -m homework.generate_qa check --info_file ../data/valid/00000_info.json --view_index 0

Generate all QA pairs for training:
   python -m homework.generate_qa generate_all --data_dir ../data/train --output_file ../data/train/all_qa_pairs.json
"""


def main():
    fire.Fire({
        "check": check_qa_pairs,
        "generate_all": generate_all_qa_pairs
    })


if __name__ == "__main__":
    main()