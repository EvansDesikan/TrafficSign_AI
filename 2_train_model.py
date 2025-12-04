import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from PIL import Image
import os
from tqdm import tqdm  # <--- NEW LIBRARY

# 1. SETUP & CONFIGURATION
data = []
labels = []
classes = 43
cur_path = os.getcwd()
DATA_DIR = 'archive' 

print("⏳ Loading images... This might take a minute.")

# 2. LOAD ALL IMAGES
# We wrap 'range(classes)' with tqdm to create the bar
for i in tqdm(range(classes), desc="Processing Classes"):
    path = os.path.join(cur_path, DATA_DIR, 'Train', str(i))
    try:
        images = os.listdir(path)
    except FileNotFoundError:
        print(f"Skipping class {i} (folder not found)")
        continue

    for a in images:
        try:
            # Open image and resize to 30x30
            image = Image.open(os.path.join(path, a))
            image = image.resize((30,30))
            image = np.array(image)
            
            data.append(image)
            labels.append(i)
        except:
            print("Error loading image")

# Convert lists to numpy arrays
data = np.array(data)
labels = np.array(labels)

print(f"✅ Loaded {data.shape[0]} images.")

# 3. PREPARE DATA
print("⚙️ Preparing data split...")
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)

# 4. BUILD THE CNN MODEL
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(43, activation='softmax'))

# 5. COMPILE & TRAIN
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print("🚀 Starting Training... (Look at the Epoch progress below)")
history = model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))

# 6. SAVE
model.save("traffic_classifier.h5")
print("🎉 Model saved as 'traffic_classifier.h5'")

# Plot
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()