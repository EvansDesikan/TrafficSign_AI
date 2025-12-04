import numpy as np
import cv2
import tensorflow as tf
import os
import matplotlib.pyplot as plt

# 1. LOAD THE TRAINED MODEL
print("⏳ Loading model...")
model = tf.keras.models.load_model('traffic_classifier.h5')
print("✅ Model loaded successfully.")

# 2. DEFINE THE LABELS (The 43 German Traffic Signs)
classes = { 
    0:'Speed limit (20km/h)', 1:'Speed limit (30km/h)', 2:'Speed limit (50km/h)', 
    3:'Speed limit (60km/h)', 4:'Speed limit (70km/h)', 5:'Speed limit (80km/h)', 
    6:'End of speed limit (80km/h)', 7:'Speed limit (100km/h)', 8:'Speed limit (120km/h)', 
    9:'No passing', 10:'No passing veh over 3.5 tons', 11:'Right-of-way at intersection', 
    12:'Priority road', 13:'Yield', 14:'Stop', 15:'No vehicles', 
    16:'Veh > 3.5 tons prohibited', 17:'No entry', 18:'General caution', 
    19:'Dangerous curve left', 20:'Dangerous curve right', 21:'Double curve', 
    22:'Bumpy road', 23:'Slippery road', 24:'Road narrows on the right', 
    25:'Road work', 26:'Traffic signals', 27:'Pedestrians', 28:'Children crossing', 
    29:'Bicycles crossing', 30:'Beware of ice/snow', 31:'Wild animals crossing', 
    32:'End speed + passing limits', 33:'Turn right ahead', 34:'Turn left ahead', 
    35:'Ahead only', 36:'Go straight or right', 37:'Go straight or left', 
    38:'Keep right', 39:'Keep left', 40:'Roundabout mandatory', 
    41:'End of no passing', 42:'End no passing veh > 3.5 tons' 
}

# 3. FUNCTION TO TEST AN IMAGE
def test_on_image(img_path):
    # Check if file exists
    if not os.path.exists(img_path):
        print(f"❌ Error: Image '{img_path}' not found.")
        return

    # Load and preprocess the image
    image = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert for display
    
    # Resize to the same size we used for training (30x30)
    resize_image = cv2.resize(image, (30, 30))
    resize_image = np.array(resize_image)
    resize_image = np.expand_dims(resize_image, axis=0) # Add the batch dimension

    # Predict
    prediction = model.predict(resize_image)
    class_id = np.argmax(prediction)
    class_name = classes[class_id]
    confidence = np.max(prediction) * 100

    # Show result
    print(f"\n🔮 Prediction: {class_name}")
    print(f"📊 Confidence: {confidence:.2f}%")

    # Plot the image with the label
    plt.imshow(image_rgb)
    plt.title(f"AI sees: {class_name}\n({confidence:.1f}%)")
    plt.axis('off')
    plt.show()

# --- RUN THE TEST ---
# Let's test it on a random image from the 'Test' folder
# You can change '00005.png' to any number you see in that folder!
test_image_path = 'archive/Test/00005.png' 

test_on_image(test_image_path)