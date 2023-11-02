import cv2
import tkinter as tk
from PIL import Image, ImageTk

class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        # Inicializa la cámara
        self.cap = cv2.VideoCapture(0)

        # Inicializa el widget de lienzo para mostrar la imagen de la cámara
        self.canvas = tk.Canvas(window, width=self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()

        # Inicializa el botón para desactivar la cámara
        self.btn_stop = tk.Button(window, text="Detener cámara", command=self.stop_camera)
        self.btn_stop.pack(pady=10)

        # Actualiza el lienzo continuamente con el flujo de la cámara
        self.update()

        # Maneja el cierre de la ventana
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

    def update(self):
        # Lee un fotograma de la cámara
        ret, frame = self.cap.read()

        # Convierte el fotograma de BGR a RGB
        if ret:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convierte la imagen de OpenCV a formato compatible con Tkinter
            image = Image.fromarray(image)
            photo = ImageTk.PhotoImage(image=image)

            # Actualiza el lienzo con la nueva imagen
            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.canvas.photo = photo

            # Llama a la función update después de un breve intervalo
            self.window.after(10, self.update)

    def stop_camera(self):
        # Detiene la cámara
        self.cap.release()

    def on_closing(self):
        # Detiene la cámara antes de cerrar la ventana
        self.stop_camera()
        self.window.destroy()

# Crea la ventana principal de la aplicación
root = tk.Tk()
app = App(root, "Cámara con Tkinter y OpenCV")

# Inicia el bucle principal de Tkinter
root.mainloop()