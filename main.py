# pip install opencv-python numpy
# C:/Users/david/AppData/Local/Microsoft/WindowsApps/python3.13.exe -m pip install opencv-python numpy
import cv2
import numpy as np
import os

# --- MEJORA: Usar un detector de caras basado en Redes Neuronales (DNN) ---
# Este método es más preciso y robusto que los Haar Cascades.
# Necesitarás descargar los archivos del modelo:
# 1. Prototxt (definición de la arquitectura): deploy.prototxt.txt
# 2. CaffeModel (pesos pre-entrenados): res10_300x300_ssd_iter_140000.caffemodel
# Puedes encontrarlos en el repositorio de OpenCV o buscarlos en internet.
# Por conveniencia, crea una carpeta 'model' y ponlos ahí.

# Rutas a los archivos del modelo
proto_path = os.path.join('model', 'deploy.prototxt.txt')
model_path = os.path.join('model', 'res10_300x300_ssd_iter_140000.caffemodel')

# Comprobar si los archivos del modelo existen
if not os.path.exists(proto_path) or not os.path.exists(model_path):
    print("Error: No se encontraron los archivos del modelo.")
    print(f"Asegúrate de tener '{proto_path}' y '{model_path}'.")
    exit()

# Cargar el clasificador de caras DNN pre-entrenado
net = cv2.dnn.readNetFromCaffe(proto_path, model_path)

# Iniciar camara por defecto (0 = cámara por defecto)
cap = cv2.VideoCapture(0)

# Establecer resolución personalizada a la ventana de la camara
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Presiona 'q' para salir...")

while True:
    ret, frame = cap.read()  # Lee un frame de la camara

    if not ret:
        print("No se pudo acceder a la cámara.")
        break

    # Obtener las dimensiones del frame y crear un blob
    (h, w) = frame.shape[:2]
    # Se redimensiona a 300x300 y se normaliza para el modelo
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    # Pasar el blob a través de la red y obtener las detecciones
    net.setInput(blob)
    detections = net.forward()

    faces_count = 0
    # Iterar sobre las detecciones
    for i in range(0, detections.shape[2]):
        # Extraer la confianza (probabilidad) de la detección
        confidence = detections[0, 0, i, 2]

        # Filtrar detecciones débiles asegurando que la confianza sea
        # mayor que un umbral mínimo (ej. 50%)
        if confidence > 0.5:
            faces_count += 1
            # Calcular las coordenadas (x, y) del cuadro delimitador
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # --- MEJORA: DIBUJAR TEXTO CON LA CONFIANZA ---
            # Crear el texto que se mostrará (ej: "99.85%")
            text = f"{confidence * 100:.2f}%"
            # Coordenada Y para el texto (un poco arriba del rectángulo)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            # Dibujar el rectángulo alrededor de la cara
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            # Escribir el texto de confianza
            cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(frame, f"Caras detectadas: {faces_count}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Mostrar el resultado
    cv2.imshow("Deteccion de Caras (DNN)", frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar ventanas
cap.release()
cv2.destroyAllWindows()
