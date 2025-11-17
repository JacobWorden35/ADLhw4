from pathlib import Path

import fire
from matplotlib import pyplot as plt

from .generate_qa import draw_detections, extract_frame_info, extract_kart_objects, extract_track_info


def generate_caption(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate caption for a specific view.
    """
    captions = []
    
    # Get track info
    track_name = extract_track_info(info_path)
    
    # Get kart objects
    kart_objects = extract_kart_objects(info_path, view_index, img_width, img_height)
    
    # Get ego car name
    import json
    with open(info_path) as f:
        info = json.load(f)
    kart_names = info["karts"]
    
    if view_index >= len(kart_names):
        return captions
    
    ego_car_name = kart_names[view_index]
    
    # Find ego car in kart_objects
    ego_kart = None
    other_karts = []
    for kart in kart_objects:
        if kart["kart_name"] == ego_car_name:
            ego_kart = kart
        else:
            other_karts.append(kart)
    
    # 1. Ego car caption
    captions.append(f"{ego_car_name} is the ego car.")
    
    # 2. Counting caption
    total_karts = len(kart_objects)
    captions.append(f"There are {total_karts} karts in the scene.")
    
    # 3. Track name caption
    captions.append(f"The track is {track_name}.")
    
    # If ego car not detected or no other karts, return what we have
    if ego_kart is None or len(other_karts) == 0:
        return captions
    
    ego_center = ego_kart["center"]
    
    # 4. Relative position captions for each other kart
    for kart in other_karts:
        kart_name = kart["kart_name"]
        kart_center = kart["center"]
        
        # Determine left/right
        is_left = kart_center[0] < ego_center[0]
        is_right = kart_center[0] > ego_center[0]
        
        # Determine front/behind
        is_front = kart_center[1] < ego_center[1]
        is_behind = kart_center[1] > ego_center[1]
        
        # Thresholds
        x_threshold = 10
        y_threshold = 10
        
        x_diff = abs(kart_center[0] - ego_center[0])
        y_diff = abs(kart_center[1] - ego_center[1])
        
        # Generate position captions
        if x_diff > x_threshold or y_diff > y_threshold:
            position_parts = []
            if y_diff > y_threshold:
                position_parts.append("front" if is_front else "back")
            if x_diff > x_threshold:
                position_parts.append("left" if is_left else "right")
            
            if position_parts:
                position = " and ".join(position_parts)
                captions.append(f"{kart_name} is {position} of the ego car.")
        
        # Simple directional captions
        if x_diff > y_diff and x_diff > x_threshold:
            direction = "left" if is_left else "right"
            captions.append(f"{kart_name} is {direction} of the ego car.")
        
        if y_diff > x_diff and y_diff > y_threshold:
            direction = "in front" if is_front else "behind"
            captions.append(f"{kart_name} is {direction} of the ego car.")
    
    return captions


def check_caption(info_file: str, view_index: int):
    captions = generate_caption(info_file, view_index)

    print("\nCaption:")
    print("-" * 50)
    for i, caption in enumerate(captions):
        print(f"{i + 1}. {caption}")
        print("-" * 50)

    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    annotated_image = draw_detections(str(image_file), info_file)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()


def generate_all_captions(data_dir: str = "../data/train", output_file: str = None):
    """
    Generate captions for all images in a directory and save to JSON.
    
    Args:
        data_dir: Directory containing the data
        output_file: Output JSON file path (optional)
    """
    from pathlib import Path
    import json
    
    data_path = Path(data_dir)
    all_captions = []
    
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
            
            # Generate captions
            captions = generate_caption(str(info_file), view_index)
            
            # Add each caption as a separate entry
            for caption in captions:
                all_captions.append({
                    "image_file": f"train/{image_file.name}",
                    "caption": caption
                })
    
    print(f"Generated {len(all_captions)} captions")
    
    # Save to file
    if output_file is None:
        output_file = data_path / "generated_captions.json"
    else:
        output_file = Path(output_file)
    
    with open(output_file, 'w') as f:
        json.dump(all_captions, f, indent=2)
    
    print(f"Saved to {output_file}")
    
    return all_captions


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python -m homework.generate_captions check --info_file ../data/valid/00000_info.json --view_index 0

Generate all captions:
   python -m homework.generate_captions generate_all --data_dir ../data/train --output_file ../data/train/all_captions.json
"""


def main():
    fire.Fire({
        "check": check_caption,
        "generate_all": generate_all_captions
    })


if __name__ == "__main__":
    main()