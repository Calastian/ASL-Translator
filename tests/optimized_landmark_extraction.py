import os
import cv2
import mediapipe as mp
import numpy as np
import pickle
import json
from tqdm import tqdm
import time

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic

# GPU optimization settings
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Allow GPU memory growth

def extract_landmarks_from_video(video_path, holistic_model):
    # Extract landmarks from a single video file.
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
    
    landmarks_data = {
        'video_path': video_path,
        'frames': []
    }
    
    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        
        if not success:
            break
        
        # Resize frame for faster processing (optional - reduces accuracy slightly)
        # height, width = frame.shape[:2]
        # if width > 640:  # Only resize if larger than 640px width
        #     scale = 640 / width
        #     new_width = int(width * scale)
        #     new_height = int(height * scale)
        #     frame = cv2.resize(frame, (new_width, new_height))
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        
        # Process frame
        results = holistic_model.process(rgb_frame)
        
        # Extract landmarks
        frame_data = {
            'frame_number': frame_count,
            'pose_landmarks': None,
            'pose_world_landmarks': None,
            'left_hand_landmarks': None,
            'right_hand_landmarks': None,
            'face_landmarks': None
        }
        
        # Convert landmarks to serializable format
        if results.pose_landmarks:
            frame_data['pose_landmarks'] = [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]
        
        if results.pose_world_landmarks:
            frame_data['pose_world_landmarks'] = [[lm.x, lm.y, lm.z] for lm in results.pose_world_landmarks.landmark]
            
        if results.left_hand_landmarks:
            frame_data['left_hand_landmarks'] = [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]
            
        if results.right_hand_landmarks:
            frame_data['right_hand_landmarks'] = [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]
            
        if results.face_landmarks:
            frame_data['face_landmarks'] = [[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark]
        
        landmarks_data['frames'].append(frame_data)
        frame_count += 1
    
    cap.release()
    return landmarks_data

def process_all_videos(root_dir, output_dir):
    """Process all videos in the directory."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of all video files
    video_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
                video_files.append(os.path.join(dirpath, filename))
    
    print(f"Found {len(video_files)} video files to process")
    
    # Initialize MediaPipe Holistic with GPU acceleration
    with mp_holistic.Holistic(
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5,
        model_complexity=1,  # Use lighter model for speed (0=lite, 1=full, 2=heavy)
        enable_segmentation=False,  # Disable segmentation for speed
        refine_face_landmarks=False,  # Disable face refinement for speed
        static_image_mode=False  # Dynamic mode for video processing
    ) as holistic:
        
        processed_count = 0
        failed_count = 0
        start_time = time.time()
        
        for video_path in tqdm(video_files, desc="Processing videos"):
            try:
                # Extract landmarks
                landmarks_data = extract_landmarks_from_video(video_path, holistic)
                
                if landmarks_data:
                    # Save landmarks data
                    video_name = os.path.splitext(os.path.basename(video_path))[0]
                    output_path = os.path.join(output_dir, f"{video_name}_landmarks.json")
                    
                    with open(output_path, 'w') as f:
                        json.dump(landmarks_data, f, indent=2)
                    
                    processed_count += 1
                else:
                    failed_count += 1
                    print(f"Failed to process: {video_path}")
                    
            except Exception as e:
                failed_count += 1
                print(f"Error processing {video_path}: {str(e)}")
            
            # Print progress every 100 videos
            if (processed_count + failed_count) % 100 == 0:
                elapsed_time = time.time() - start_time
                avg_time_per_video = elapsed_time / (processed_count + failed_count)
                remaining_videos = len(video_files) - (processed_count + failed_count)
                estimated_time_remaining = avg_time_per_video * remaining_videos
                
                print(f"Progress: {processed_count + failed_count}/{len(video_files)}")
                print(f"Successful: {processed_count}, Failed: {failed_count}")
                print(f"Avg time per video: {avg_time_per_video:.2f}s")
                print(f"Estimated time remaining: {estimated_time_remaining/3600:.2f} hours")
                print("-" * 50)
    
    total_time = time.time() - start_time
    print(f"\nProcessing complete!")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Videos processed successfully: {processed_count}")
    print(f"Videos failed: {failed_count}")
    print(f"Average time per video: {total_time/len(video_files):.2f}s")

if __name__ == "__main__":
    # Configuration
    root_dir = '../ASL_Citizen/videos'  # Your video directory
    output_dir = '../ASL_Citizen/landmarks'  # Where to save landmark files
    
    # Process all videos
    process_all_videos(root_dir, output_dir)
