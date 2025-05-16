import cv2
import numpy as np
from PIL import Image
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
path = 'dataset'

def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    face_samples, ids = [], []

    for image_path in image_paths:
        gray_img = Image.open(image_path).convert('L')
        img_np = np.array(gray_img, 'uint8')
        id = int(os.path.split(image_path)[-1].split('.')[1])
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")\
            .detectMultiScale(img_np)

        for (x, y, w, h) in faces:
            face_samples.append(img_np[y:y+h, x:x+w])
            ids.append(id)

    return face_samples, ids

faces, ids = get_images_and_labels(path)
recognizer.train(faces, np.array(ids))
recognizer.save('trainer.yml')
print("[INFO] Model trained.")
