import os
import cv2
import numpy as np

def gather_images_in_folder(folder_path, target_size=(100, 100)):
    """
    Loads all .png images from `folder_path`, converts to grayscale,
    resizes to target_size, and returns a data matrix X for that folder.

    Returns:
    --------
    X : np.ndarray of shape (d, n)
        Each column is a flattened image from this folder.
    n_images : int
        Number of images loaded (i.e., n).
    """
    all_vectors = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.png'):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Warning: Could not read {img_path}. Skipping.")
                continue
            if target_size is not None:
                img = cv2.resize(img, target_size)
            
            vec = img.flatten().astype(np.float64)
            all_vectors.append(vec)
    
    if len(all_vectors) == 0:
        return None, 0  # No images found or problem reading
    
    X = np.array(all_vectors, dtype=np.float64)  # shape (n, d)
    X = X.T  # shape (d, n)
    return X, X.shape[1]


def build_dual_pca(X):
    """
    Builds a PCA model using the dual method (X^T X).

    Returns:
    --------
    mean_face : np.ndarray of shape (d,)
    eigenfaces : np.ndarray of shape (d, n)
        We keep ALL the eigenfaces (no truncation here).
    eigenvalues : np.ndarray of shape (n,)
        The corresponding eigenvalues for each eigenface.
    """
    d, n = X.shape

    # 1. Compute mean
    mean_face = np.mean(X, axis=1, keepdims=True)  # (d, 1)
    X_centered = X - mean_face

    # 2. Dual approach => X^T X => shape (n, n)
    XtX = np.dot(X_centered.T, X_centered)  # (n, n)

    # 3. Eigen-decomposition
    eigvals, eigvecs = np.linalg.eig(XtX)
    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs)

    # 4. Sort in descending order
    idx_sorted = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx_sorted]
    eigvecs = eigvecs[:, idx_sorted]

    # 5. Build eigenfaces in original d-dim
    eigenfaces = np.dot(X_centered, eigvecs)  # shape (d, n)

    # Normalize each eigenface
    for i in range(n):
        norm = np.linalg.norm(eigenfaces[:, i])
        if norm > 1e-12:
            eigenfaces[:, i] /= norm

    # Squeeze mean_face to shape (d,)
    mean_face = mean_face.squeeze()

    return mean_face, eigenfaces, eigvals


def build_all_pca(data_dir, target_size=(100, 100)):
    """
    For each subfolder in `data_dir`, build a PCA model (keeping ALL eigenfaces).
    Returns a dictionary: { person_name : (mean_face, eigenfaces, eigenvals) }

    This effectively trains "one PCA per folder/person".

    Parameters:
    -----------
    data_dir : str
        Top-level directory containing subfolders, each for one person.
    target_size : (width, height)
        Size to which images are resized.

    Returns:
    --------
    pca_models : dict
        Dictionary of the form:
            {
              'PersonName': {
                  'mean_face': mean_face,
                  'eigenfaces': eigenfaces,
                  'eigenvalues': eigenvals
              },
              ...
            }
    person_list : list of str
        The names of the subfolders (persons) for which PCA was built.
    """
    pca_models = {}
    person_list = []

    for person_name in os.listdir(data_dir):
        person_folder = os.path.join(data_dir, person_name)
        if not os.path.isdir(person_folder):
            continue

        X, n_images = gather_images_in_folder(person_folder, target_size=target_size)
        if X is None or n_images == 0:
            print(f"No images for {person_name}, skipping PCA.")
            continue

        mean_face, eigenfaces, eigvals = build_dual_pca(X)
        pca_models[person_name] = {
            'mean_face': mean_face,
            'eigenfaces': eigenfaces,
            'eigenvalues': eigvals
        }
        person_list.append(person_name)

    return pca_models, person_list


def build_pca_for_new_user(captured_images, output_dir):
    """
    Build PCA for a single new user from the captured images.
    This is a skeleton. Replace the steps with your actual PCA pipeline from build_all_pca.
    """

    # Convert images to a 2D array [num_images x (height*width)]
    # Possibly resize to match your existing PCA pipeline’s target_size
    # For a robust approach, do your typical preprocessing steps
    data_matrix = []
    for img in captured_images:
        # If you normally do resizing:
        # img = cv2.resize(img, (84, 96))  # example
        vec = img.flatten().astype(np.float64)
        data_matrix.append(vec)
    data_matrix = np.array(data_matrix).T  # shape (d, n)

    # Compute mean face
    mean_face = np.mean(data_matrix, axis=1, keepdims=True)  # shape (d,1)

    # Center the data
    centered_data = data_matrix - mean_face

    # Compute covariance in a smaller dimension:
    # Usually we do SVD or use the trick with centered_data.T * centered_data
    # For example, with SVD:
    U, S, Vt = np.linalg.svd(centered_data, full_matrices=False)

    eigenfaces = U  # shape (d, n)
    eigenvalues = S**2  # since S is the singular values, S**2 are the eigenvalues

    # Save the PCA model (eigenfaces, eigenvalues, mean_face) to disk or in memory
    # For example as a NumPy pickled dict:
    pca_model = {
        "mean_face": mean_face.ravel(),     # shape (d,)
        "eigenfaces": eigenfaces,           # shape (d, n)
        "eigenvalues": eigenvalues,         # shape (n,)
    }

    # Write to disk
    import pickle
    model_path = os.path.join(output_dir, "pca_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(pca_model, f)

    print(f"[INFO] Saved new user PCA model to {model_path}")
