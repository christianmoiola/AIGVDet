import os
import cv2
from tqdm import tqdm

def create_folders():
    base_path = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(base_path, exist_ok=True)
    subfolders = [
        'train/train_set_1/0_real',
        'train/train_set_1/1_fake',
        'val/val_set_1/0_real',
        'val/val_set_1/1_fake'
    ]
    
    for subfolder in subfolders:
        path = os.path.join(base_path, subfolder)
        os.makedirs(path, exist_ok=True)

def fill_train_fake():
    base_LTX_video = os.path.join(os.path.dirname(__file__), 'LTXVideo')
    save_path = os.path.join(os.path.dirname(__file__), 'data', 'train', 'train_set_1', '1_fake')
    
    categories = ['landscapes', 'objects', 'vehicles']
    video_count = 0  # Track video index
    
    for cat in categories:
        cat_path = os.path.join(base_LTX_video, cat)
        
        if not os.path.exists(cat_path):
            print(f"Skipping {cat_path}, directory not found.")
            continue
        
        for video in os.listdir(cat_path):
            if not video.endswith(".mp4"):  # Ensure it's an MP4 file
                continue
            
            video_path = os.path.join(cat_path, video)
            video_folder = os.path.join(save_path, f"video_{video_count:04d}")
            os.makedirs(video_folder, exist_ok=True)  # Create video folder
            
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total number of frames
            frame_index = 0  # Track frame number
            
            print(f"Extracting frames from {video} ({total_frames} frames)...")
            
            with tqdm(total=total_frames, desc=f"Processing {video}", unit="frame") as pbar:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break  # Stop when no more frames
                    
                    frame_filename = os.path.join(video_folder, f"{frame_index:04d}.png")
                    cv2.imwrite(frame_filename, frame)  # Save frame as PNG
                    
                    frame_index += 1
                    pbar.update(1)  # Update progress bar
            
            cap.release()  # Release video file
            video_count += 1  # Increment video counter

def fill_val_fake():
    base_LTX_video = os.path.join(os.path.dirname(__file__), 'LTXVideo')
    save_path = os.path.join(os.path.dirname(__file__), 'data', 'val', 'val_set_1', '1_fake')
    
    categories = ['animals', 'people']
    video_count = 0  # Track video index
    
    for cat in categories:
        cat_path = os.path.join(base_LTX_video, cat)
        
        if not os.path.exists(cat_path):
            print(f"Skipping {cat_path}, directory not found.")
            continue
        
        for video in os.listdir(cat_path):
            if not video.endswith(".mp4"):  # Ensure it's an MP4 file
                continue
            
            video_path = os.path.join(cat_path, video)
            video_folder = os.path.join(save_path, f"video_{video_count:04d}")
            os.makedirs(video_folder, exist_ok=True)  # Create video folder
            
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total number of frames
            frame_index = 0  # Track frame number
            
            print(f"Extracting frames from {video} ({total_frames} frames)...")
            
            with tqdm(total=total_frames, desc=f"Processing {video}", unit="frame") as pbar:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break  # Stop when no more frames
                    
                    frame_filename = os.path.join(video_folder, f"{frame_index:04d}.png")
                    cv2.imwrite(frame_filename, frame)  # Save frame as PNG
                    
                    frame_index += 1
                    pbar.update(1)  # Update progress bar
            
            cap.release()  # Release video file
            video_count += 1  # Increment video counter

def fill_train_real():
    base_LTX_video = os.path.join(os.path.dirname(__file__), 'Real')
    save_path = os.path.join(os.path.dirname(__file__), 'data', 'train', 'train_set_1', '0_real')
    
    categories = ['landscapes', 'objects', 'vehicles']
    video_count = 0  # Track video index
    
    for cat in categories:
        cat_path = os.path.join(base_LTX_video, cat)
        
        if not os.path.exists(cat_path):
            print(f"Skipping {cat_path}, directory not found.")
            continue
        
        for video in os.listdir(cat_path):
            video_path = os.path.join(cat_path, video, 'origin')
            
            if not os.path.exists(video_path):
                print(f"Skipping {video_path}, directory not found.")
                continue
            
            video_folder = os.path.join(save_path, f"video_{video_count:04d}")
            os.makedirs(video_folder, exist_ok=True)  # Create video folder
            
            frame_files = os.listdir(video_path)
            total_frames = len(frame_files)  # Get total number of frames
            frame_index = 0  # Track frame number
            
            print(f"Extracting frames from {video} ({total_frames} frames)...")
            
            with tqdm(total=total_frames, desc=f"Processing {video}", unit="frame") as pbar:
                for frame_file in frame_files:
                    frame_path = os.path.join(video_path, frame_file)
                    frame = cv2.imread(frame_path)
                    
                    if frame is None:
                        print(f"Skipping {frame_path}, unable to read image.")
                        continue
                    
                    frame_filename = os.path.join(video_folder, f"{frame_index:04d}.png")
                    cv2.imwrite(frame_filename, frame)  # Save frame as PNG
                    
                    frame_index += 1
                    pbar.update(1)  # Update progress bar
            
            video_count += 1  # Increment video counter

def fill_val_real():
    base_LTX_video = os.path.join(os.path.dirname(__file__), 'Real')
    save_path = os.path.join(os.path.dirname(__file__), 'data', 'val', 'val_set_1', '0_real')
    
    categories = ['animals', 'people']
    video_count = 0  # Track video index
    
    for cat in categories:
        cat_path = os.path.join(base_LTX_video, cat)
        
        if not os.path.exists(cat_path):
            print(f"Skipping {cat_path}, directory not found.")
            continue
        
        for video in os.listdir(cat_path):
            video_path = os.path.join(cat_path, video, 'origin')
            
            if not os.path.exists(video_path):
                print(f"Skipping {video_path}, directory not found.")
                continue
            
            video_folder = os.path.join(save_path, f"video_{video_count:04d}")
            os.makedirs(video_folder, exist_ok=True)  # Create video folder
            
            frame_files = os.listdir(video_path)
            total_frames = len(frame_files)  # Get total number of frames
            frame_index = 0  # Track frame number
            
            print(f"Extracting frames from {video} ({total_frames} frames)...")
            
            with tqdm(total=total_frames, desc=f"Processing {video}", unit="frame") as pbar:
                for frame_file in frame_files:
                    frame_path = os.path.join(video_path, frame_file)
                    frame = cv2.imread(frame_path)
                    
                    if frame is None:
                        print(f"Skipping {frame_path}, unable to read image.")
                        continue
                    
                    frame_filename = os.path.join(video_folder, f"{frame_index:04d}.png")
                    cv2.imwrite(frame_filename, frame)  # Save frame as PNG
                    
                    frame_index += 1
                    pbar.update(1)  # Update progress bar
            
            video_count += 1  # Increment video counter


if __name__ == '__main__':
    # create_folders()
    # fill_train_fake()
    #fill_train_real()
    fill_val_real()
