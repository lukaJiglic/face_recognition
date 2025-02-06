import cv2
import os
import numpy as np
from datetime import datetime
from live_pca import *
from pca_module import *

def interactive_camera_capture(output_dir="new_user_data",
                               recommended_count=20,
                               face_size=(84, 96),
                               cascade_path="haarcascade_frontalface_default.xml"):
    """
    Open a window to capture images for PCA, but detect the face and crop it,
    then resize to `face_size` (84x96).
    
    Press keys:
      - 'c' : capture and store the face
      - 'd' : delete last captured face
      - 'b' : build PCA
      - 't' : test PCA live
      - 'q' : quit
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load the Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_path)
    if face_cascade.empty():
        print("[ERROR] Could not load Haar Cascade. Check path.")
        return

    # Start video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    captured_faces = []

    print("[INFO] Press 'c' to capture, 'd' to delete last, 'b' to build PCA, 't' to test PCA, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame from camera.")
            break

        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Face detection
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.3, 
            minNeighbors=5, 
            minSize=(50, 50)
        )

        # Draw bounding boxes around faces (for user feedback)
        for (x, y, w, h) in faces:
            cv2.rectangle(
                frame, 
                (x, y), 
                (x + w, y + h), 
                (0, 255, 0), 
                2
            )

        overlay_text_1 = f"Faces Captured: {len(captured_faces)}/{recommended_count}"
        overlay_text_2 = "[c] Capture | [d] Delete last | [b] Build PCA | [t] Test PCA | [q] Quit"
        cv2.putText(frame, overlay_text_1, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, overlay_text_2, (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Interactive Capture", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            # Quit the loop
            break
        elif key == ord('c'):
            # If a face is detected, take the first (or largest) and crop it
            if len(faces) == 0:
                print("[INFO] No face detected. Try again.")
            else:
                # Here we could pick the largest face or just the first one
                (x, y, w, h) = faces[0]  # simplest approach: take the first
                face_crop = gray[y:y+h, x:x+w]
                # Resize to face_size
                face_crop = cv2.resize(face_crop, face_size)
                captured_faces.append(face_crop)
                print(f"[INFO] Captured face {len(captured_faces)}")
        elif key == ord('d'):
            # Delete last face if available
            if captured_faces:
                captured_faces.pop()
                print(f"[INFO] Deleted last captured face. Count is now {len(captured_faces)}.")
            else:
                print("[INFO] No faces to delete.")
        elif key == ord('b'):
            # Build PCA with captured faces
            if len(captured_faces) < 2:
                print("[INFO] Not enough images to build PCA. Capture more.")
            else:
                print("[INFO] Building PCA with captured faces...")
                build_pca_for_new_user(captured_faces, output_dir)
                print("[INFO] PCA build complete. You may now press 't' to test or 'q' to quit.")
        elif key == ord('t'):
            # Test PCA with camera (you can implement a real-time recognition test)
            print("[INFO] Starting PCA test with the camera feed...")
            test_pca_live(output_dir, face_size=face_size)

    cap.release()
    cv2.destroyAllWindows()

    # If user never pressed 'b' but we want to save images anyway:
    if captured_faces:
        print("[INFO] Saving captured faces to disk (without PCA build).")
        for i, face in enumerate(captured_faces):
            filename = os.path.join(output_dir, f"captured_{i}.png")
            cv2.imwrite(filename, face)
        print("[INFO] Faces saved. Goodbye!")
