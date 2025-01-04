import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
def scan_image():
    imagePath = input("Enter the path of the image: ")
    print("Path of the image is: ", imagePath)
    img = cv2.imread(imagePath)
    img.shape
    print("Shape of the image is: ", img.shape)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_image.shape
    print("Shape of the image is: ", gray_image.shape)
    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    face = face_classifier.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )
    print("Number of faces found in the image: ", len(face))
    for (x, y, w, h) in face:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(20,10))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()
def constface():
    
    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    video_capture = cv2.VideoCapture(0)
    def detect_bounding_box(vid):
        gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
        for (x, y, w, h) in faces:
            cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
        return faces, vid
    while True:

        result, video_frame = video_capture.read()  # read frames from the video
        if result is False:
            break  # terminate the loop if the frame is not read successfully

        faces = detect_bounding_box(
            video_frame
        )  # apply the function we created to the video frame

        cv2.imshow(
            "My Face Detection Project", video_frame
        )  # display the processed frame in a window named "My Face Detection Project"

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()

def createdataset():
 

    # Set up directories
    dataset_path = "face_dataset"  # Directory to save the images
    person_name = input("Enter the name of the person: ")
    save_path = os.path.join(dataset_path, person_name)
    
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Initialize face detector
    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    
    # Initialize video capture
    video_capture = cv2.VideoCapture(0)
    print("Press 'q' to quit and save data")
    
    image_count = 0
    max_images = 500  # Maximum number of images to record
    
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
            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
            # Extract face region
            face = gray_frame[y:y + h, x:x + w]
    
            # Resize to a standard size (e.g., 200x200)
            face_resized = cv2.resize(face, (200, 200))
    
            # Save face image
            image_path = os.path.join(save_path, f"{person_name}_{image_count}.jpg")
            cv2.imwrite(image_path, face_resized)
            image_count += 1
    
        # Display the frame
        cv2.imshow("Face Data Collection", frame)
    
        # Break on 'q' or when max images are saved
        if cv2.waitKey(1) & 0xFF == ord('q') or image_count >= max_images:
            break

    print(f"Collected {image_count} face images for {person_name}.")
    video_capture.release()
    cv2.destroyAllWindows()
# def facecheck():
#     
#     path = input("Enter the path of the dataset: ")
#     dataset_path = input("Enter the path of the dataset: ")  # Directory to save the images
# 
# def load_known_faces():
#     """
#     Load all face images and their labels from the dataset directory.
#     """
#     known_faces = []
#     known_names = []
# 
#     for person_name in os.listdir(dataset_path):
#         person_folder = os.path.join(dataset_path, person_name)
#         if os.path.isdir(person_folder):
#             for image_file in os.listdir(person_folder):
#                 image_path = os.path.join(person_folder, image_file)
#                 face_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#                 known_faces.append(face_image)
#                 known_names.append(person_name)
# 
#     return known_faces, known_names
# 
# def mse(imageA, imageB):
#     """
#     Compute the Mean Squared Error between two images.
#     """
#     err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
#     err /= float(imageA.shape[0] * imageA.shape[1])
#     return err
# 
# def is_known_face(face, known_faces, known_names, threshold=2000):
#     """
#     Check if the detected face matches any face in the dataset.
#     """
#     face_resized = cv2.resize(face, (200, 200))
#     for i, known_face in enumerate(known_faces):
#         if known_face.shape == face_resized.shape:
#             error = mse(known_face, face_resized)
#             if error < threshold:  # Lower error indicates a better match
#                 return True, known_names[i]
#     return False, None
# 
# # Load known faces
# known_faces, known_names = load_known_faces()
# 
# # Initialize face detector
# face_classifier = cv2.CascadeClassifier(
#     cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
# )
# 
# # Initialize video capture
# video_capture = cv2.VideoCapture(0)
# print("Press 'q' to quit.")
# 
# while True:
#     result, frame = video_capture.read()
#     if not result:
#         print("Failed to capture video frame.")
#         break
# 
#     # Convert to grayscale
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# 
#     # Detect faces
#     faces = face_classifier.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
# 
#     for (x, y, w, h) in faces:
#         # Extract face region
#         face = gray_frame[y:y + h, x:x + w]
# 
#         # Check if the face matches any known faces
#         match, name = is_known_face(face, known_faces, known_names)
# 
#         if match:
#             label = f"Known: {name}"
#             color = (0, 255, 0)  # Green for known faces
#         else:
#             label = "Unknown"
#             color = (0, 0, 255)  # Red for unknown faces
# 
#         # Draw a rectangle around the face
#         cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
#         cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
# 
#     # Display the frame
#     cv2.imshow("Face Check", frame)
# 
#     # Break on 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# 
# video_capture.release()
# cv2.destroyAllWindows()
# 
choice = input("Press 1 for scanning image: " "\nPress 2 for detecting face: " "\nPress 3 for creating dataset: ")
if choice == '1':
     scan_image()
elif choice == '2':
    constface()
elif choice == '3':
    createdataset()
