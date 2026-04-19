import tensorflow as tf
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk

# Incarca modelul
model = tf.keras.models.load_model('flower_model.keras')
class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
class_names_ro = {
    'daisy': 'Margareta',
    'dandelion': 'Papadie',
    'rose': 'Trandafir',
    'sunflower': 'Floarea soarelui',
    'tulip': 'Lalea'
}

def predict_flower(img_path):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_array = tf.expand_dims(img_resized, 0)
    predictions = model.predict(img_array, verbose=0)
    score = tf.nn.softmax(predictions[0])
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)
    return predicted_class, confidence

def open_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    if not file_path:
        return

    img = Image.open(file_path)
    img.thumbnail((400, 400))
    img_tk = ImageTk.PhotoImage(img)
    image_label.config(image=img_tk)
    image_label.image = img_tk

    predicted_class, confidence = predict_flower(file_path)
    name_ro = class_names_ro[predicted_class]
    color = "#2ecc71" if confidence > 70 else "#e67e22"
    result_label.config(
        text=f"{name_ro} ({predicted_class}) - {confidence:.1f}%",
        fg=color
    )

# Interfata
root = tk.Tk()
root.title("Flower Detection")
root.geometry("500x620")
root.configure(bg="#1a1a2e")

title = Label(root, text="Flower Detection", font=("Arial", 22, "bold"),
              bg="#1a1a2e", fg="white")
title.pack(pady=20)

image_label = Label(root, bg="#16213e", width=50, height=25)
image_label.pack(pady=10)

btn = Button(root, text="Alege o imagine", font=("Arial", 13),
             bg="#e94560", fg="white", relief="flat", padx=20, pady=10,
             cursor="hand2", command=open_image)
btn.pack(pady=15)

result_label = Label(root, text="Nicio imagine selectata", font=("Arial", 14),
                     bg="#1a1a2e", fg="white")
result_label.pack(pady=10)

root.mainloop()