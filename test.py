import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
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
num_classes = cursor.fetchone()[0]  # A kategóriák száma

# Érvénytelen `category_id` értékek ellenőrzése és törlése
cursor.execute("""
DELETE FROM images
WHERE category_id NOT IN (SELECT id FROM categories)
""")
connection.commit()

# Lekérdezés a képek útvonalaira és kategóriákra
cursor.execute("SELECT file_path, category_id FROM images")
data = cursor.fetchall()

# Képek alapmappa
base_directory = "C:/xampp/htdocs/animal_images/"

# Képek és címkék betöltése
images = []
labels = []

for (file_path, category_id) in data:
    full_path = os.path.join(base_directory, file_path)
    print(f"Feldolgozás alatt: {full_path}")
    try:
        image = cv2.imread(full_path)
        if image is None:
            print(f"Hiba: Nem található vagy hibás kép: {full_path}")
            continue  # Hibás képet kihagyjuk
        if category_id >= num_classes:  # Ellenőrzés: érvénytelen category_id
            print(f"Hiba: Érvénytelen category_id ({category_id}), kihagyva.")
            continue
        image = cv2.resize(image, (128, 128))
        image = img_to_array(image)
        images.append(image)
        labels.append(category_id)
    except Exception as e:
        print(f"Hiba a kép betöltésekor: {e}")

# Ellenőrzés a képek és címkék számának szinkronizálására
if len(images) != len(labels):
    print(f"Figyelmeztetés: Az images ({len(images)}) és labels ({len(labels)}) listák mérete eltér!")
    min_length = min(len(images), len(labels))
    images = images[:min_length]
    labels = labels[:min_length]

# Ha nincs kép, álljunk meg
if not images or not labels:
    print("Egyetlen kép sem került betöltésre! Ellenőrizd az elérési utakat és a képfájlokat.")
    exit()

# Adatok előkészítése
images = np.array(images, dtype="float") / 255.0
labels = to_categorical(labels, num_classes=num_classes)

# Adatok szétválasztása betanítási és validációs halmazra
train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# Adatbővítés beállítása a betanító halmazra
from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
train_generator = datagen.flow(train_images, train_labels, batch_size=32)
val_generator = ImageDataGenerator().flow(val_images, val_labels, batch_size=32)

# MobileNetV2 alapmodell betöltése és testreszabása
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(128, 128, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)  # Kimeneti réteg a kategóriák számával
animal_recognition_model = Model(inputs=base_model.input, outputs=predictions)

# Csak a saját rétegeket tanítjuk
for layer in base_model.layers:
    layer.trainable = False

# Modell konfigurálása
animal_recognition_model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Modell betanítása
print("Betanítás indítása...")
animal_recognition_model.fit(train_generator, epochs=10, validation_data=val_generator)

# Modell súlyok mentése
animal_recognition_model.save_weights("animal_recognition_model.weights.h5")
print("Betanítás befejeződött, súlyok elmentve.")


# Állat detekció és felismerés függvény
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
            animal_name = result[0]
            return f"A képen egy {animal_name} látható."
        else:
            return "Hiba: A kategória nem található az adatbázisban."
    except Exception as e:
        return f"Hiba: {str(e)}"

# Példa használat
test_image = "C:/Users/zolta/Desktop/semmi.jpg"
print(detect_and_recognize_animal(test_image))

# Adatbázis kapcsolat lezárása
cursor.close()
connection.close()
