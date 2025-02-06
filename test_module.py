import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def test_pca_reconstruction(test_dir,
                            pca_models,
                            person_list,
                            target_size=(100, 100),
                            variance_threshold=0.95):
    """
    Classify each .png in `test_dir` by computing the reconstruction 
    error in each person's PCA. Then pick the best match,
    showing each result in a *blocking* matplotlib window 
    that only closes on user dismissal.
    """
    for filename in os.listdir(test_dir):
        if filename.lower().endswith('.png'):
            img_path = os.path.join(test_dir, filename)
            test_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if test_img is None:
                print(f"Could not read {img_path}. Skipping.")
                continue
            
            if target_size is not None:
                test_img = cv2.resize(test_img, target_size)
            
            test_vec = test_img.flatten().astype(np.float64)

            best_person = None
            best_error = float('inf')
            best_reconstruction = None

            # Check each person's PCA
            for person_name in person_list:
                mean_face = pca_models[person_name]['mean_face']
                eigenfaces = pca_models[person_name]['eigenfaces']
                eigenvals = pca_models[person_name]['eigenvalues']

                centered_test = test_vec - mean_face
                coords = eigenfaces.T.dot(centered_test)

                total_var = np.sum(eigenvals)
                cumvar = np.cumsum(eigenvals)
                # find k such that sum(eigvals[:k])/sum(eigvals) >= variance_threshold
                k = np.searchsorted(cumvar, variance_threshold * total_var) + 1

                recon_95 = eigenfaces[:, :k].dot(coords[:k]) + mean_face
                error = np.linalg.norm(test_vec - recon_95)

                if error < best_error:
                    best_error = error
                    best_person = person_name
                    best_reconstruction = recon_95

            print(f"Test Image: {filename}, Predicted: {best_person}, Error: {best_error:.2f}")

            # Show using matplotlib in a blocking manner
            height, width = target_size[1], target_size[0]
            recon_img = best_reconstruction.reshape((height, width))

            fig, axes = plt.subplots(1, 2, figsize=(6, 4))
            # Original
            axes[0].imshow(test_img, cmap='gray')
            axes[0].set_title(f'Original\n{filename}')
            axes[0].axis('off')

            # Reconstruction
            axes[1].imshow(recon_img, cmap='gray')
            axes[1].set_title(f'Reconstruction\nPred: {best_person}')
            axes[1].axis('off')

            plt.show(block=False)
            plt.waitforbuttonpress()
            plt.close(fig)
