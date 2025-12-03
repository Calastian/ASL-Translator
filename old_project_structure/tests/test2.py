import os
import cv2
import mediapipe as mp
import numpy as np
import json
from tqdm import tqdm
import time
from scipy import interpolate

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic

# GPU optimization settings
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Allow GPU memory growth

# Constants for landmark handling
MISSING_VALUE = -1.0  # Use -1.0 for missing landmarks (out of normalized range [0,1])
POSE_LANDMARKS_COUNT = 33
HAND_LANDMARKS_COUNT = 21

def create_missing_landmarks(landmark_count):
    """Create missing landmark array with consistent missing values."""
    return [[MISSING_VALUE, MISSING_VALUE, MISSING_VALUE] for _ in range(landmark_count)]

def interpolate_landmarks(landmarks_sequence, landmark_type):
    """
    Interpolate missing landmarks in a sequence using linear interpolation.
    
    Args:
        landmarks_sequence: List of landmark frames
        landmark_type: 'pose', 'left_hand', or 'right_hand'
    """
    if not landmarks_sequence:
        return landmarks_sequence
    
    # Convert to numpy array for easier manipulation
    sequence_array = np.array(landmarks_sequence)
    
    # Find valid (non-missing) frames
    valid_mask = ~np.all(sequence_array == MISSING_VALUE, axis=(1, 2))
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) < 2:
        # Not enough valid frames for interpolation
        return landmarks_sequence
    
    # Interpolate missing frames between valid ones
    for i in range(len(sequence_array)):
        if not valid_mask[i]:  # This frame is missing
            # Find nearest valid frames
            prev_valid = valid_indices[valid_indices < i]
            next_valid = valid_indices[valid_indices > i]
            
            if len(prev_valid) > 0 and len(next_valid) > 0:
                # Interpolate between previous and next valid frames
                prev_idx = prev_valid[-1]
                next_idx = next_valid[0]
                
                # Linear interpolation
                alpha = (i - prev_idx) / (next_idx - prev_idx)
                interpolated = (1 - alpha) * sequence_array[prev_idx] + alpha * sequence_array[next_idx]
                sequence_array[i] = interpolated
            
            elif len(prev_valid) > 0:
                # Use last known position (forward fill)
                sequence_array[i] = sequence_array[prev_valid[-1]]
            
            elif len(next_valid) > 0:
                # Use next known position (backward fill)
                sequence_array[i] = sequence_array[next_valid[0]]
    
    return sequence_array.tolist()

def smooth_landmarks(landmarks_sequence, window_size=3):
    """
    Apply temporal smoothing to reduce jitter in landmark detection.
    
    Args:
        landmarks_sequence: List of landmark frames
        window_size: Size of smoothing window
    """
    if len(landmarks_sequence) < window_size:
        return landmarks_sequence
    
    sequence_array = np.array(landmarks_sequence)
    smoothed = np.copy(sequence_array)
    
    # Apply moving average filter
    for i in range(window_size // 2, len(sequence_array) - window_size // 2):
        start_idx = i - window_size // 2
        end_idx = i + window_size // 2 + 1
        
        # Only smooth non-missing values
        window = sequence_array[start_idx:end_idx]
        valid_mask = window != MISSING_VALUE
        
        if np.any(valid_mask):
            # Average only valid values
            for j in range(window.shape[1]):  # For each landmark
                for k in range(window.shape[2]):  # For each coordinate (x, y, z)
                    valid_values = window[:, j, k][valid_mask[:, j, k]]
                    if len(valid_values) > 0:
                        smoothed[i, j, k] = np.mean(valid_values)
    
    return smoothed.tolist()


def extract_label(video_filename):
    # Remove extension
    name, _ = os.path.splitext(video_filename)
    # Split by the first dash
    parts = name.split('-', 1)
    if len(parts) > 1:
        label = parts[1].strip()
    else:
        label = name.strip()
    return label


def extract_landmarks_from_video(video_path, holistic_model, apply_processing=True):
    """Extract landmarks from a single video file with intelligent missing data handling."""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    # Extract label and trimmed video name
    label = extract_label(os.path.basename(video_path))
    
    landmarks_data = {
        'video_path': video_path,
        'Label': label,
        'fps': fps,
        'total_frames': total_frames,
        'duration_seconds': duration,
        'frames': []
    }
    
    # Temporary storage for sequences (for post-processing)
    pose_sequence = []
    pose_world_sequence = []
    left_hand_sequence = []
    right_hand_sequence = []
    
    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        
        if not success:
            break
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        
        # Process frame
        results = holistic_model.process(rgb_frame)
        
        # Extract landmarks with missing value handling
        pose_landmarks = (
            [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]
            if results.pose_landmarks else create_missing_landmarks(POSE_LANDMARKS_COUNT)
        )
        
        pose_world_landmarks = (
            [[lm.x, lm.y, lm.z] for lm in results.pose_world_landmarks.landmark]
            if results.pose_world_landmarks else create_missing_landmarks(POSE_LANDMARKS_COUNT)
        )
        
        left_hand_landmarks = (
            [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]
            if results.left_hand_landmarks else create_missing_landmarks(HAND_LANDMARKS_COUNT)
        )
        
        right_hand_landmarks = (
            [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]
            if results.right_hand_landmarks else create_missing_landmarks(HAND_LANDMARKS_COUNT)
        )
        
        # Store in sequences for post-processing
        pose_sequence.append(pose_landmarks)
        pose_world_sequence.append(pose_world_landmarks)
        left_hand_sequence.append(left_hand_landmarks)
        right_hand_sequence.append(right_hand_landmarks)
        
        frame_count += 1
    
    cap.release()
    
    # Post-process sequences if requested
    if apply_processing and frame_count > 1:
        print(f"Processing {frame_count} frames for {os.path.basename(video_path)}")
        
        # Apply interpolation to fill gaps
        pose_sequence = interpolate_landmarks(pose_sequence, 'pose')
        left_hand_sequence = interpolate_landmarks(left_hand_sequence, 'left_hand')
        right_hand_sequence = interpolate_landmarks(right_hand_sequence, 'right_hand')
        
        # Apply smoothing to reduce jitter
        pose_sequence = smooth_landmarks(pose_sequence)
        left_hand_sequence = smooth_landmarks(left_hand_sequence)
        right_hand_sequence = smooth_landmarks(right_hand_sequence)
    
    # Create final frame data
    for i in range(frame_count):
        frame_data = {
            'frame_number': i,
            'timestamp': i / fps if fps > 0 else i,
            'pose_landmarks_count': len(pose_sequence[i]),
            'pose_world_landmarks_count': len(pose_world_sequence[i]),
            'left_hand_landmarks_count': len(left_hand_sequence[i]),
            'right_hand_landmarks_count': len(right_hand_sequence[i]),
            'pose_landmarks': pose_sequence[i],
            'pose_world_landmarks': pose_world_sequence[i],
            'left_hand_landmarks': left_hand_sequence[i],
            'right_hand_landmarks': right_hand_sequence[i],
            # Add metadata about landmark availability
            'has_pose': not np.all(np.array(pose_sequence[i]) == MISSING_VALUE),
            'has_left_hand': not np.all(np.array(left_hand_sequence[i]) == MISSING_VALUE),
            'has_right_hand': not np.all(np.array(right_hand_sequence[i]) == MISSING_VALUE)
        }
        landmarks_data['frames'].append(frame_data)
    
    return landmarks_data

def analyze_landmark_coverage(landmarks_data):
    """Analyze what percentage of frames have each type of landmark."""
    total_frames = len(landmarks_data['frames'])
    
    pose_coverage = sum(1 for frame in landmarks_data['frames'] if frame['has_pose'])
    left_hand_coverage = sum(1 for frame in landmarks_data['frames'] if frame['has_left_hand'])
    right_hand_coverage = sum(1 for frame in landmarks_data['frames'] if frame['has_right_hand'])
    
    return {
        'pose_coverage': pose_coverage / total_frames * 100,
        'left_hand_coverage': left_hand_coverage / total_frames * 100,
        'right_hand_coverage': right_hand_coverage / total_frames * 100
    }

def process_all_videos(root_dir, output_dir):
    """Process all videos in the directory with intelligent landmark handling."""
    
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
        
        # Statistics tracking
        total_coverage_stats = {
            'pose_coverage': [],
            'left_hand_coverage': [],
            'right_hand_coverage': []
        }
        
        for video_path in tqdm(video_files, desc="Processing videos"):
            try:
                # Extract landmarks
                landmarks_data = extract_landmarks_from_video(video_path, holistic, apply_processing=True)
                
                if landmarks_data:
                    # Analyze coverage
                    coverage_stats = analyze_landmark_coverage(landmarks_data)
                    
                    # Add coverage stats to the data
                    landmarks_data['coverage_stats'] = coverage_stats
                    
                    # Track overall statistics
                    total_coverage_stats['pose_coverage'].append(coverage_stats['pose_coverage'])
                    total_coverage_stats['left_hand_coverage'].append(coverage_stats['left_hand_coverage'])
                    total_coverage_stats['right_hand_coverage'].append(coverage_stats['right_hand_coverage'])
                    
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
                
                if total_coverage_stats['pose_coverage']:
                    print(f"Avg coverage - Pose: {np.mean(total_coverage_stats['pose_coverage']):.1f}%, "
                          f"Left hand: {np.mean(total_coverage_stats['left_hand_coverage']):.1f}%, "
                          f"Right hand: {np.mean(total_coverage_stats['right_hand_coverage']):.1f}%")
                print("-" * 50)
    
    # Final statistics
    total_time = time.time() - start_time
    print(f"\nProcessing complete!")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Videos processed successfully: {processed_count}")
    print(f"Videos failed: {failed_count}")
    print(f"Average time per video: {total_time/len(video_files):.2f}s")
    
    if total_coverage_stats['pose_coverage']:
        print(f"\nOverall Coverage Statistics:")
        print(f"Pose landmarks: {np.mean(total_coverage_stats['pose_coverage']):.1f}% ± {np.std(total_coverage_stats['pose_coverage']):.1f}%")
        print(f"Left hand landmarks: {np.mean(total_coverage_stats['left_hand_coverage']):.1f}% ± {np.std(total_coverage_stats['left_hand_coverage']):.1f}%")
        print(f"Right hand landmarks: {np.mean(total_coverage_stats['right_hand_coverage']):.1f}% ± {np.std(total_coverage_stats['right_hand_coverage']):.1f}%")

if __name__ == "__main__":
    # Configuration
    root_dir = '../ASL_Citizen/test_videos' 
    output_dir = '../ASL_Citizen/landmarks_2'  
    
    # Process all videos
    process_all_videos(root_dir, output_dir)
