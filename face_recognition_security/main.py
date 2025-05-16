import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
from datetime import datetime
import webbrowser

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def log_access(user_id, status):
    with open("access_log.txt", "a") as file:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file.write(f"{timestamp} - ID: {user_id} - Status: {status}\n")

# def secret_action():
#     # You can replace this with any custom action
#     webbrowser.open("https://chat.openai.com")

class FaceRecognitionApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Face Recognition Security")
        self.video = cv2.VideoCapture(0)

        self.label = Label(window, text="Starting...", font=("Helvetica", 16), fg="green")
        self.label.pack()

        self.canvas = tk.Canvas(window, width=640, height=480)
        self.canvas.pack()

        self.update_frame()
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

    def update_frame(self):
        ret, frame = self.video.read()
        if not ret:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        access_granted = False
        for (x, y, w, h) in faces:
            id_, confidence = recognizer.predict(gray[y:y + h, x:x + w])

            if confidence < 60:
                access_granted = True
                text = f"Access Granted: User {id_}"
                self.label.config(text=text, fg="green")
                log_access(id_, "Granted")
                #secret_action()
            else:
                text = "Access Denied"
                self.label.config(text=text, fg="red")
                log_access("Unknown", "Denied")

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0) if access_granted else (0, 0, 255), 2)

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(img))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
        self.canvas.image = img
        self.window.after(10, self.update_frame)

    def on_closing(self):
        self.video.release()
        self.window.destroy()

# Start GUI
root = tk.Tk()
app = FaceRecognitionApp(root)
root.mainloop()
