import cv2
import os
import json

# Set up directories
dataset_path = "face_dataset"
person_name = input("Enter the name of the person: ")
save_path = os.path.join(dataset_path, person_name)
mesh_path = os.path.join(save_path, "mesh_data")
obj_path = os.path.join(save_path, "3d_models")

# Create directories if they don't exist
os.makedirs(save_path, exist_ok=True)
os.makedirs(mesh_path, exist_ok=True)
os.makedirs(obj_path, exist_ok=True)

# Initialize face detector
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Predefined points to simulate basic landmarks (2D ratios)
landmark_ratios = [
    (0.3, 0.3), (0.7, 0.3),  # Eyes
    (0.5, 0.5),              # Nose
    (0.4, 0.7), (0.6, 0.7)   # Mouth corners
]

# Define triangle faces based on landmark indices
triangle_indices = [
    (0, 2, 1),  # Triangle connecting left eye, nose, and right eye
    (2, 3, 4),  # Triangle connecting nose and mouth corners
    (0, 3, 2),  # Triangle connecting left eye, left mouth, and nose
    (1, 4, 2)   # Triangle connecting right eye, right mouth, and nose
]

# Initialize video capture
video_capture = cv2.VideoCapture(0)
print("Press 'q' to quit and save data")

image_count = 0
max_images = 5  # Maximum number of images to record

def save_mesh_as_obj(landmarks, output_path):
    """
    Save 2D landmarks as a 3D mesh model in .obj format compatible with Blender.
    """
    with open(output_path, "w") as obj_file:
        obj_file.write("# 3D Mesh OBJ file generated from 2D landmarks\n")
        
        # Write vertices
        for x, y in landmarks:
            z = 0  # Assign depth (z-coordinate)
            obj_file.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        
        # Write faces (triangles)
        obj_file.write("\n# Faces\n")
        for t in triangle_indices:
            # OBJ file indices are 1-based
            obj_file.write(f"f {t[0] + 1} {t[1] + 1} {t[2] + 1}\n")

while True:
    result, frame = video_capture.read()
    if not result:
        print("Failed to capture video frame.")
        break

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_classifier.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Simulate facial landmarks
        landmarks = []
        for (lx, ly) in landmark_ratios:
            landmark_x = int(x + lx * w)
            landmark_y = int(y + ly * h)
            landmarks.append((landmark_x, landmark_y))
            cv2.circle(frame, (landmark_x, landmark_y), 2, (0, 255, 0), -1)

        # Save face and mesh data
        if image_count < max_images:
            # Save the cropped face image
            face_region = gray_frame[y:y + h, x:x + w]
            if face_region.size > 0:  # Ensure the cropped region is valid
                resized_face = cv2.resize(face_region, (200, 200))
                image_path = os.path.join(save_path, f"{person_name}_{image_count}.jpg")
                cv2.imwrite(image_path, resized_face)

                # Save the mesh data
                mesh_file_path = os.path.join(mesh_path, f"{person_name}_{image_count}.json")
                mesh_data = {
                    "image_path": image_path,
                    "landmarks": landmarks
                }
                with open(mesh_file_path, "w") as mesh_file:
                    json.dump(mesh_data, mesh_file, indent=4)

                # Save the 3D mesh as an OBJ file
                obj_file_path = os.path.join(obj_path, f"{person_name}_{image_count}.obj")
                save_mesh_as_obj(landmarks, obj_file_path)

                image_count += 1

    # Display the frame
    cv2.imshow("Face Dataset Creation with 3D Mesh", frame)

    # Break on 'q' or when max images are saved
    if cv2.waitKey(1) & 0xFF == ord('q') or image_count >= max_images:
        break

print(f"Collected {image_count} face images, mesh data, and 3D models for {person_name}.")
video_capture.release()
cv2.destroyAllWindows()
