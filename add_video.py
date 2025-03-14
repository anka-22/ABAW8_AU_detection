import os
import numpy as np
from PIL import Image


source_dir = '/home/data/AU_test/cropped_aligned'
target_root_dir = '/home/data/compacted_48'


for subdir_name in os.listdir(source_dir):
    subdir_path = os.path.join(source_dir, subdir_name)
    if os.path.isdir(subdir_path):

        images_list = []

        file_names = sorted(os.listdir(subdir_path))
        for file_name in file_names:
            if file_name.endswith('.jpg'):  
                file_path = os.path.join(subdir_path, file_name)
  
                try:
                    with Image.open(file_path) as img:
                        img_array = np.array(img)
                        images_list.append(img_array)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    continue
        

        frame_npy_path = os.path.join(target_root_dir, subdir_name, "frame.npy")
        if os.path.exists(frame_npy_path):
            frame_data = np.load(frame_npy_path)
            num_frames = frame_data.shape[0]  


            if len(images_list) > num_frames:
                images_list = images_list[:num_frames]  
            elif len(images_list) < num_frames:

                while len(images_list) < num_frames:
                    images_list.append(images_list[-1])
        
            if images_list:

                images_array = np.stack(images_list)

                target_subdir = os.path.join(target_root_dir, subdir_name)
                if not os.path.exists(target_subdir):
                    os.makedirs(target_subdir)

                np.save(os.path.join(target_subdir, "video.npy"), images_array)
                print(f"Saved {os.path.join(target_subdir, 'video.npy')} with shape {images_array.shape}")
            else:
                print(f"No images found in {subdir_path}")
        else:
            print(f"Frame data file not found: {frame_npy_path}")
