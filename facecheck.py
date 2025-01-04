import cv2
import os
import numpy as np

# Set up directories
path = input("Enter the path of the dataset: ")
dataset_path = path  # Directory to save the images

def load_known_faces():
    """
    Load all face images and their labels from the dataset directory.
    """
    known_faces = []
    known_names = []

    for person_name in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person_name)
        if os.path.isdir(person_folder):
            for image_file in os.listdir(person_folder):
                image_path = os.path.join(person_folder, image_file)
                face_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                known_faces.append(face_image)
                known_names.append(person_name)

    return known_faces, known_names

def mse(imageA, imageB):
    """
    Compute the Mean Squared Error between two images.
    """
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def is_known_face(face, known_faces, known_names, threshold=2000):
    """
    Check if the detected face matches any face in the dataset.
    """
    face_resized = cv2.resize(face, (200, 200))
    for i, known_face in enumerate(known_faces):
        if known_face.shape == face_resized.shape:
            error = mse(known_face, face_resized)
            if error < threshold:  # Lower error indicates a better match
                return True, known_names[i]
    return False, None

# Load known faces
known_faces, known_names = load_known_faces()

# Initialize face detector
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Initialize video capture
video_capture = cv2.VideoCapture(0)
print("Press 'q' to quit.")

while True:
    result, frame = video_capture.read()
    if not result:
        print("Failed to capture video frame.")
        break

    # Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_classifier.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

    for (x, y, w, h) in faces:
        # Extract face region
        face = gray_frame[y:y + h, x:x + w]

        # Check if the face matches any known faces
        match, name = is_known_face(face, known_faces, known_names)

        if match:
            label = f"Known: {name}"
            color = (0, 255, 0)  # Green for known faces
            print(f"Known face detected: {name}")
        else:
            label = "Unknown"
            color = (0, 0, 255)  # Red for unknown faces
            print("Unknown face detected!")

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Display the frame
    cv2.imshow("Face Check", frame)

    # Break on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()