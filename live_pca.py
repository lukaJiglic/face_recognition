import pickle
import cv2
import numpy as np
import os
from pca_module import *

def load_all_pca_models_hybrid(pca_dir):
    """
    A unified function that handles two scenarios:
    1. Multi-user folder (has subfolders). For each subfolder:
       - If pca_model.pkl is found, load it.
       - Else if there are >=2 .png images, build PCA.
    2. Single-user folder (no subfolders):
       - If pca_model.pkl is found, load it.
       - Else if there are >=2 .png images, build PCA.

    Returns:
      pca_models : dict
        { 'UserNameOrFolder' : {
             'mean_face': ...,
             'eigenfaces': ...,
             'eigenvalues': ...
          }, 
          ...
        }
    """
    pca_models = {}

    entries = os.listdir(pca_dir)
    subfolders = [
        f for f in entries 
        if os.path.isdir(os.path.join(pca_dir, f))
    ]

    # --- CASE 1: If we find subfolders, treat pca_dir as a multi-user directory
    if subfolders:
        for sub_name in subfolders:
            sub_path = os.path.join(pca_dir, sub_name)
            pkl_path = os.path.join(sub_path, "pca_model.pkl")

            if os.path.isfile(pkl_path):
                # Load existing PCA model
                try:
                    with open(pkl_path, "rb") as f:
                        pca_models[sub_name] = pickle.load(f)
                    print(f"[INFO] Loaded PCA model for '{sub_name}' from {pkl_path}.")
                except Exception as e:
                    print(f"[WARNING] Could not load '{pkl_path}': {e}")
                    continue
            else:
                # No pca_model.pkl => see if we can build from PNG images
                X, n_images = gather_images_in_folder(sub_path, target_size=(84, 96))
                if X is None or n_images < 2:
                    print(f"[WARNING] Subfolder '{sub_name}' has <2 PNG images. Skipping.")
                    continue
                print(f"[INFO] Building PCA for '{sub_name}' on the fly.")
                mean_face, eigenfaces, eigenvalues = build_dual_pca(X)
                pca_models[sub_name] = {
                    "mean_face": mean_face,
                    "eigenfaces": eigenfaces,
                    "eigenvalues": eigenvalues
                }
                # Optionally save the new pkl
                try:
                    with open(pkl_path, "wb") as f:
                        pickle.dump(pca_models[sub_name], f)
                    print(f"[INFO] Saved new PCA model to '{pkl_path}'.")
                except Exception as e:
                    print(f"[WARNING] Could not save pca_model.pkl for '{sub_name}': {e}")

    # --- CASE 2: If NO subfolders found, treat pca_dir as a single-user folder
    else:
        pkl_path = os.path.join(pca_dir, "pca_model.pkl")
        folder_name = os.path.basename(os.path.abspath(pca_dir))

        if os.path.isfile(pkl_path):
            # Load existing PCA
            try:
                with open(pkl_path, "rb") as f:
                    pca_models[folder_name] = pickle.load(f)
                print(f"[INFO] Loaded PCA model from '{pkl_path}' for user '{folder_name}'.")
            except Exception as e:
                print(f"[WARNING] Could not load '{pkl_path}': {e}")
        else:
            # No .pkl => see if we can build from PNG images
            X, n_images = gather_images_in_folder(pca_dir, target_size=(84, 96))
            if X is None or n_images < 2:
                print(f"[WARNING] Folder '{pca_dir}' has <2 PNG images and no pkl. Nothing to load/build.")
                return pca_models

            print(f"[INFO] Building PCA for single-user folder '{folder_name}' on the fly.")
            mean_face, eigenfaces, eigenvalues = build_dual_pca(X)
            pca_models[folder_name] = {
                "mean_face": mean_face,
                "eigenfaces": eigenfaces,
                "eigenvalues": eigenvalues
            }

            # Optionally save new pkl
            try:
                with open(pkl_path, "wb") as f:
                    pickle.dump(pca_models[folder_name], f)
                print(f"[INFO] Saved PCA model to '{pkl_path}'.")
            except Exception as e:
                print(f"[WARNING] Could not save pca_model.pkl for '{folder_name}': {e}")

    return pca_models


def load_all_pca_models_including_parent(pca_dir):
    """
    1) Look at the parent directory of `pca_dir`.
    2) For each subfolder in that parent directory (including e.g. `old_user/`, `data/`, etc.),
       load or build PCA models. (Using load_all_pca_models_hybrid)
       - We skip if the subfolder is exactly pca_dir itself to avoid double-loading.
    3) Then also load/build PCA models from `pca_dir` itself.
    4) Combine everything into a single dict and return.
    """
    
    # Dictionary to hold all PCA models from parent + this folder
    combined_pca_models = {}

    # Get the absolute path and parent directory
    abs_pca_dir = os.path.abspath(pca_dir)
    parent_dir = os.path.dirname(abs_pca_dir)
    
    # 1) Iterate all subfolders in the parent directory
    parent_subfolders = [
        d for d in os.listdir(parent_dir)
        if os.path.isdir(os.path.join(parent_dir, d))
    ]

    for subfolder in parent_subfolders:
        subfolder_path = os.path.join(parent_dir, subfolder)
        # Skip if it's exactly the same as pca_dir (we'll handle pca_dir below)
        if os.path.abspath(subfolder_path) == abs_pca_dir:
            continue

        # Use the hybrid loader on each sibling folder
        sub_models = load_all_pca_models_hybrid(subfolder_path)
        # Merge them into combined_pca_models
        for key, model_dict in sub_models.items():
            # Potentially rename or just trust keys won't collide
            combined_pca_models[key] = model_dict

    # 2) Now also handle pca_dir itself
    this_folder_models = load_all_pca_models_hybrid(abs_pca_dir)
    for key, model_dict in this_folder_models.items():
        combined_pca_models[key] = model_dict

    return combined_pca_models



def test_pca_live(pca_dir="new_user_data",
                  face_size=(84, 96),
                  cascade_path="haarcascade_frontalface_default.xml",
                  variance_threshold=0.95,
                  unknown_threshold=15000):
    """
    Demonstration of using *all* PCA models on a live camera feed.
    For each detected face, we:
       1) Crop and resize to face_size
       2) For each user-PCA, compute reconstruction error
       3) Pick the best user or label as 'Unknown' if error > unknown_threshold
    """
    # Load all PCA models found in pca_dir
    pca_models = load_all_pca_models_including_parent(pca_dir)
    if not pca_models:
        print(f"[ERROR] No PCA models found in {pca_dir}. Build them first.")
        return

    # Load Haar Cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_path)
    if face_cascade.empty():
        print("[ERROR] Could not load Haar Cascade.")
        return

    # Start camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera for testing.")
        return

    print("[INFO] Press 'q' to quit PCA test.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(50, 50)
        )

        # For each face, we do PCA classification
        for (x, y, w, h) in faces:
            face_crop = gray[y:y+h, x:x+w]
            face_crop = cv2.resize(face_crop, face_size)
            test_vec = face_crop.flatten().astype(np.float64)

            # Now classify using the approach from test_pca_reconstruction
            best_user = None
            best_error = float('inf')

            for user_name, model in pca_models.items():
                mean_face = model["mean_face"]
                eigenfaces = model["eigenfaces"]
                eigenvalues = model["eigenvalues"]

                # 1) Center test
                centered_test = test_vec - mean_face

                # 2) Project
                coords = eigenfaces.T.dot(centered_test)

                # 3) Find number of PCs that preserve `variance_threshold`
                total_var = np.sum(eigenvalues)
                cumvar = np.cumsum(eigenvalues)
                k = np.searchsorted(cumvar, variance_threshold * total_var) + 1

                # 4) Reconstruct
                recon_95 = eigenfaces[:, :k].dot(coords[:k]) + mean_face

                # 5) Compute error
                error = np.linalg.norm(test_vec - recon_95)

                if error < best_error:
                    best_error = error
                    best_user = user_name

            # Decide if it's "Unknown" or best_user
            label_text = f"{best_user} ({best_error:.1f})"
            if best_error > unknown_threshold:
                label_text = f"Unknown ({best_error:.1f})"

            # Draw bounding box & label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame,
                        label_text,
                        (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0,255,0),
                        2)

        cv2.imshow("PCA Live Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
