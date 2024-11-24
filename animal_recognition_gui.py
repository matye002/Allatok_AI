import os
import cv2
import numpy as np
from tkinter import Tk, Label, Button, filedialog, StringVar, Canvas
from PIL import Image, ImageTk
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import mysql.connector

# Kapcsolódás az adatbázishoz
connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="allat"
)
cursor = connection.cursor()

# Kategóriák számának lekérdezése
cursor.execute("SELECT COUNT(*) FROM categories")
num_classes = cursor.fetchone()[0]

# MobileNetV2 alapmodell betöltése
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(128, 128, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)  # Kimeneti réteg a kategóriák számával
animal_recognition_model = Model(inputs=base_model.input, outputs=predictions)

# Modell betöltése (csak súlyok)
animal_recognition_model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
animal_recognition_model.load_weights("animal_recognition_model.weights.h5")


# Állat felismerési függvény
def detect_and_recognize_animal(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            return "Hiba: A kép nem található vagy hibás formátum."
        image_resized = cv2.resize(image, (128, 128))
        image_array = img_to_array(image_resized)
        image_array = np.expand_dims(image_array, axis=0) / 255.0

        # Előrejelzés
        prediction = animal_recognition_model.predict(image_array)
        predicted_class = int(np.argmax(prediction))

        # Lekérdezés az adatbázisból az állat nevének megjelenítéséhez
        cursor.execute("SELECT name FROM categories WHERE id = %s", (predicted_class,))
        result = cursor.fetchone()
        if result:
            return f"A képen egy {result[0]} látható."
        else:
            return "Hiba: A kategória nem található az adatbázisban."
    except Exception as e:
        return f"Hiba: {str(e)}"


# Tkinter GUI
def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("Képfájlok", "*.jpg *.jpeg *.png")])
    if file_path:
        image_path.set(file_path)
        display_image(file_path)

def recognize_image():
    path = image_path.get()
    if not os.path.isfile(path):
        result.set("Hiba: Nem létező fájl.")
        return
    result.set("Felismerés folyamatban...")
    prediction = detect_and_recognize_animal(path)
    result.set(prediction)

def display_image(image_path):
    image = Image.open(image_path)
    image = image.resize((250, 250), Image.Resampling.LANCZOS)  # Javított opció
    img = ImageTk.PhotoImage(image)
    canvas.delete("all")
    canvas.create_image(0, 0, anchor="nw", image=img)
    canvas.image = img  # Referencia fenntartása a memóriaszivárgás elkerüléséhez


# Tkinter ablak létrehozása
root = Tk()
root.title("Állatfelismerő MI")
root.geometry("800x600")

# Felhasználói elemek
image_path = StringVar()
result = StringVar()

Label(root, text="Állatfelismerő rendszer", font=("Arial", 18)).pack(pady=10)
Label(root, text="Válasszon egy képet:").pack(pady=5)
Button(root, text="Tallózás Képhez", command=browse_file).pack(pady=5)

canvas = Canvas(root, width=250, height=250, bg="gray")
canvas.pack(pady=10)

Button(root, text="Felismerés Indítása", command=recognize_image).pack(pady=10)
Label(root, text="Eredmény:", font=("Arial", 14)).pack(pady=5)
Label(root, textvariable=result, font=("Arial", 12), wraplength=400).pack(pady=10)

# Tkinter fő ciklus indítása
root.mainloop()

# Adatbázis kapcsolat lezárása
cursor.close()
connection.close()
