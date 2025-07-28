# C:/Users/david/AppData/Local/Microsoft/WindowsApps/python3.13.exe -m pip install opencv-python numpy

import cv2
import numpy as np
import os
from collections import deque

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

# Historial de caras
# Crear un deque para almacenar las imágenes de las caras nuevas max 10 caras
faces_history = deque(maxlen=10)
# Lista para guardar las coordenadas de las caras del fotograma anterior
previous_faces_boxes = []
# Distancia (en píxeles) para considerar una cara como "nueva"
NEW_FACE_DISTANCE_THRESHOLD = 100
# Tamaño de las miniaturas en el historial
HISTORY_THUMBNAIL_SIZE = (80, 80)

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
    current_faces_boxes = []
    # Iterar sobre las detecciones
    for i in range(0, detections.shape[2]):
        # Extraer la confianza (probabilidad) de la detección
        confidence = detections[0, 0, i, 2]

        # Filtrar detecciones débiles asegurando que la confianza sea mayor a 50%
        if confidence > 0.5:
            faces_count += 1
            # Calcular las coordenadas (x, y) del cuadro delimitador
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            current_faces_boxes.append((startX, startY, endX, endY))

            # Lógica para detectar si es una cara "nueva" 
            is_new_face = True
            current_center = ((startX + endX) // 2, (startY + endY) // 2)

            for prev_box in previous_faces_boxes:
                prev_center = ((prev_box[0] + prev_box[2]) // 2, (prev_box[1] + prev_box[3]) // 2)
                # Calcular la distancia
                distance = np.sqrt((current_center[0] - prev_center[0])**2 + (current_center[1] - prev_center[1])**2)
                if distance < NEW_FACE_DISTANCE_THRESHOLD:
                    is_new_face = False
                    break

            if is_new_face:
                # Recortar la cara del fotograma
                face_roi = frame[startY:endY, startX:endX]
                if face_roi.size > 0:
                    # Redimensionar a miniatura y añadir al historial
                    thumbnail = cv2.resize(face_roi, HISTORY_THUMBNAIL_SIZE)
                    faces_history.appendleft(thumbnail)

            # Crear el texto que se mostrará %
            text = f"{confidence * 100:.2f}%"
            # Coordenada Y para el texto (un poco arriba del rectángulo)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            # Dibujar el rectángulo alrededor de la cara
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            # Escribir el texto de confianza
            cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Actualizar las caras del fotograma anterior para la siguiente iteración
    previous_faces_boxes = current_faces_boxes

    # Mostrar el conteo actual
    cv2.putText(frame, f"Caras detectadas: {faces_count}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Mostrar el historial de imágenes de caras nuevas
    cv2.putText(frame, "Caras Nuevas:", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    for i, thumbnail in enumerate(faces_history):
        y_pos = 95 + i * (HISTORY_THUMBNAIL_SIZE[1] + 10)  # 10px de padding
        # Comprobar si la miniatura cabe en el frame verticalmente
        if y_pos + HISTORY_THUMBNAIL_SIZE[1] > h:
            break  # Dejar de dibujar si no hay más espacio
        frame[y_pos:y_pos + HISTORY_THUMBNAIL_SIZE[1], 10:10 + HISTORY_THUMBNAIL_SIZE[0]] = thumbnail
        cv2.rectangle(frame, (10, y_pos), (10 + HISTORY_THUMBNAIL_SIZE[0], y_pos + HISTORY_THUMBNAIL_SIZE[1]), (255, 255, 0), 1)

    # Mostrar el resultado
    cv2.imshow("Deteccion de Caras (DNN)", frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar ventanas
cap.release()
cv2.destroyAllWindows()
