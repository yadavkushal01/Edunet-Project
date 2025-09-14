import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array,ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import joblib
import fix_labels as fix


# Parameters
img_size = (128, 128)
batch_size = 32
epochs = 20
data_dir = "C:\\msys64\\home\\kushal yadav\\Project_Edunet_internship\\project_opencv\\dataset\\images"

# Fixing labels in .txt file
fix.fix_label("C:\\msys64\\home\\kushal yadav\\Project_Edunet_internship\\project_opencv\\dataset\\labels","train.txt")
fix.fix_label("C:\\msys64\\home\\kushal yadav\\Project_Edunet_internship\\project_opencv\\dataset\\labels","val.txt")
fix.fix_label("C:\\msys64\\home\\kushal yadav\\Project_Edunet_internship\\project_opencv\\dataset\\labels","test.txt")

# Map class labels to folder name
class_map = {
    0: "cardboard",
    1: "glass",
    2: "metal",
    3: "paper",
    4: "plastic",
    5: "trash"
}

def load_data(txt_file, img_size=(128, 128)):
    images, labels = [], []
    with open(txt_file, "r") as f:
        for line in f:
            img_name, _ = line.strip().split()

            # separating char and digit
            class_name = ''.join([c for c in img_name if not c.isdigit()]).replace(".jpg", "")
            
            # check for class_map values
            if class_name not in class_map.values():
                print(f"[Error] Unknown class {class_name} for {img_name}")
                continue

            # find label from class_map
            label = list(class_map.keys())[list(class_map.values()).index(class_name)]
            img_path = os.path.join(data_dir, class_name, img_name)

            if os.path.exists(img_path):
                img = load_img(img_path, target_size=img_size)
                img = img_to_array(img) / 255.0
                images.append(img)
                labels.append(label)
            else:
                print(f"[Warning] File not found: {img_path}")
    return np.array(images), np.array(labels)


# Load train, val, test .txt files
X_train, y_train = load_data("C:\\msys64\\home\\kushal yadav\\Project_Edunet_internship\\project_opencv\\dataset\\labels\\train_fix.txt")
X_val, y_val     = load_data("C:\\msys64\\home\\kushal yadav\\Project_Edunet_internship\\project_opencv\\dataset\\labels\\val_fix.txt")
X_test, y_test   = load_data("C:\\msys64\\home\\kushal yadav\\Project_Edunet_internship\\project_opencv\\dataset\\labels\\test_fix.txt")

num_classes = len(class_map)  # cardboard, glass, paper, metal, plastic, trash

# Converting labels to one-hot encoding
y_train = to_categorical(y_train, num_classes=num_classes)
y_val   = to_categorical(y_val, num_classes=num_classes)
y_test  = to_categorical(y_test, num_classes=num_classes)

# Data augmentation
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Loading MobileNetV2 base model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128,128,3))
base_model.trainable = False  # Freeze pretrained layers

# Adding top layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Training model with augmentation
history = model.fit(
    train_datagen.flow(X_train, y_train, batch_size=batch_size),
    epochs=epochs,
    validation_data=(X_val, y_val),
    callbacks=[lr_reduce, early_stop]
)

# test model
test_loss, test_acc = model.evaluate(X_test, y_test)
# print(f"Test Accuracy: {test_acc*100:.2f}%")

model.save("garbage_classifier.h5")


# # predict image
# class_names = ["cardboard", "glass", "paper", "metal", "plastic", "trash"]

# img_path = "sample_image.jpeg"
# img = load_img(img_path, target_size=img_size)
# x = img_to_array(img) / 255.0
# x = np.expand_dims(x, axis=0)

# pred = model.predict(x)
# pred_class = np.argmax(pred)
# # print(f" Prediction: {class_names[pred_class]}")


# # Dump model using joblib
# joblib.dump(model, 'waste_model.joblib')
# print("Model saved as waste_model.joblib")