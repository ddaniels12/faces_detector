# pip install opencv-python
# C:/Users/david/AppData/Local/Microsoft/WindowsApps/python3.13.exe -m pip install opencv-python
import cv2

# Clasificador de caras preentrenado
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convertir a blanco y negro / escala de grises

    # Detectar caras
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Dibujar rectangulos alrededor de las caras detectadas
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.putText(frame, f"Caras detectadas: {len(faces)}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)


    # Mostrar el resultado
    cv2.imshow("Deteccion de Caras", frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar ventanas
cap.release()
cv2.destroyAllWindows()
