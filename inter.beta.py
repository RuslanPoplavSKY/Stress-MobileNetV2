import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Завантаження моделі
model = tf.keras.models.load_model('model_mobilenet.keras')

# Класи
class_labels = ['Без стресу', 'Стрес']

# Функція для передобробки зображення
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Основна логіка інтерфейсу
def select_image():
    global selected_image_path
    selected_image_path = filedialog.askopenfilename(
        initialdir=r"C:\Users\rusla\OneDrive\Desktop\something\facesData\test",
        title="Оберіть зображення",
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    
    if selected_image_path:
        img = Image.open(selected_image_path)
        img = img.resize((300, 300))
        img = ImageTk.PhotoImage(img)
        panel.config(image=img)
        panel.image = img
        result_label.config(text="")
        chart_button.pack_forget()  # Сховати кнопку поки нема результату
        predict_button.pack(pady=10)

def predict_stress():
    img_array = preprocess_image(selected_image_path)
    predictions = model.predict(img_array)[0]
    predicted_class = class_labels[np.argmax(predictions)]
    confidence = np.max(predictions)

    result_label.config(
        text=f"Результат: {predicted_class} ({confidence*100:.2f}%)", 
        fg="lime" if predicted_class == "Без стресу" else "red"
    )
    chart_button.pack(pady=10)  # Показати кнопку для графіка
    global last_predictions
    last_predictions = predictions

def show_chart():
    plt.figure(figsize=(6, 4))
    plt.bar(class_labels, last_predictions, color=["green", "red"])
    plt.title("Ймовірності класифікації")
    plt.ylabel("Ймовірність")
    plt.ylim([0, 1])
    for i, val in enumerate(last_predictions):
        plt.text(i, val + 0.02, f"{val*100:.2f}%", ha='center')
    plt.tight_layout()
    plt.show()

# Ініціалізація GUI
root = tk.Tk()
root.title("Виявлення стресу за фото")
root.geometry("400x550")

select_button = tk.Button(root, text="Обрати зображення", command=select_image)
select_button.pack(pady=10)

panel = tk.Label(root)
panel.pack()

predict_button = tk.Button(root, text="Виявити стрес", command=predict_stress)

result_label = tk.Label(root, text="", font=("Helvetica", 14))
result_label.pack(pady=10)

chart_button = tk.Button(root, text="Показати діаграму", command=show_chart)

root.mainloop()