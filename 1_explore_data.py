import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt

# 1. SETUP PATHS
# Make sure this matches your folder name!
DATA_DIR = 'archive' 
TRAIN_PATH = os.path.join(DATA_DIR, 'Train')

# 2. CHECK FOLDERS
if not os.path.exists(TRAIN_PATH):
    print(f"❌ Error: Could not find '{TRAIN_PATH}'.")
    print("Please make sure you extracted the 'archive.zip' file correctly.")
    exit()
else:
    print(f"✅ Found dataset at: {TRAIN_PATH}")

# 3. COUNT CLASSES
folders = os.listdir(TRAIN_PATH)
print(f"Total Traffic Sign Classes: {len(folders)}")

# 4. VISUALIZE RANDOM IMAGES
# We will pick one image from the first 5 classes to show you what they look like
plt.figure(figsize=(15, 5))

for i in range(5):
    path = os.path.join(TRAIN_PATH, str(i)) # Look in folder '0', '1', '2'...
    images = os.listdir(path)
    
    # Pick the first image in that folder
    img_path = os.path.join(path, images[0])
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Fix colors for matplotlib
    
    # Plot it
    plt.subplot(1, 5, i+1)
    plt.imshow(img)
    plt.title(f"Class {i}")
    plt.axis('off')

print("Opening visualization window...")
plt.show()