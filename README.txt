===========================================
 FACE RECOGNITION DEMO - README
===========================================

This is a face recognition demo using Principal Component Analysis (PCA) 
(often referred to as the “Eigenfaces” technique). The project allows you to:

1. Build PCA models from a dataset of images (stored in a 'data/' folder).
2. Test the PCA models on static images (stored in a 'test/' folder).
3. Capture new images from a live camera to create or overwrite a user folder.
4. Perform real-time face recognition on a live camera feed using the 
   previously captured or existing PCA models.

-------------------------------------------
 CONTENTS
-------------------------------------------
- [1. Project Structure]
- [2. Installation]
- [3. Usage]
  - [3.1 Running the GUI App]
  - [3.2 Using the GUI]
  - [3.3 Additional CLI Scripts (Optional)]
- [4. How It Works]
- [5. Customization]
- [6. Troubleshooting]

-------------------------------------------
 1. PROJECT STRUCTURE
-------------------------------------------
A typical structure for this repository is as follows:

face_recognition/
├─ data/
│   ├─ person1/
│   ├─ person2/
│   └─ ...
├─ test/
│   ├─ test_image_1.png
│   ├─ test_image_2.png
│   └─ ...
├─ user00/
│   ├─ user00_pca_model.pkl
│   ├─ user_image_1.png
│   └─ ...
├─ live_pca.py
├─ pca_module.py
├─ test_module.py
├─ camera_capture.py
├─ face_recognition_app.py
├─ requirements.txt
├─ README.txt (this file)
├─ LICENSE
│   (other user files)
└─ ...

Here's what each folder/file generally contains:
- data/: Subfolders for each person, each containing .png images.
- test/: Additional .png images for static testing (not in data).
- user00/: Predownloaded captures from live test and PCA model.
- face_recognition_app.py: Main GUI-based application script.
- pca_module.py: PCA building functions.
- test_module.py: Functions for static reconstruction/testing.
- camera_capture.py: Interactive camera capture logic.
- live_pca.py: Real-time face recognition using PCA.
- requirements.txt: Python library dependencies.
- README.txt: This document.
- LICENSE: MIT license

-------------------------------------------
 2. INSTALLATION
-------------------------------------------
1) Clone or download the repository:
   
   git clone https://github.com/lukaJiglic/face_recognition.git
   cd face_recognition

2) Install dependencies (preferably in a virtual environment):
   
   pip install -r requirements.txt

   Main libraries:
   - numpy
   - opencv-python
   - matplotlib
   - PyQt5
   - pickle (built into Python, typically)

3) Make sure your camera is connected and accessible by OpenCV 
   (usually index 0 by default).

-------------------------------------------
 3. USAGE
-------------------------------------------

3.1) Running the GUI App
------------------------
From inside the face_recognition folder:

   python face_recognition_app.py

This will open a window titled "Face Recognition Demo."

3.2) Using the GUI
------------------
1. BUILD PCA FROM DATA:
   - The app automatically builds PCA from the 'data/' folder when it starts.
   - If 'data/' is missing, you’ll see a warning message.

2. TEST ON TEST/ IMAGES:
   - Click the button "Test on test/ images."
   - For each .png in 'test/', a reconstruction window appears (side-by-side original and reconstruction). 
     Close each window to move to the next image.
   - Check the console for the predicted person and reconstruction error.

3. OPEN CAMERA (CAPTURE):
   - Click "Open Camera (Capture)" to start the camera feed.
   - You will be asked for a folder name (e.g., "Alice"). 
     - Leave blank or type "new_user_data" to use (and overwrite) the default folder.
   - The capture window allows you to:
     - Press 'c' to capture a face.
     - Press 'd' to delete the last capture.
     - Press 'b' to build a PCA model for your newly captured faces (saving it into your folder).
     - Press 't' to test PCA live immediately after building.
     - Press 'q' to quit the capture window.

4. OPEN CAMERA (LIVE TEST):
   - Click "Open Camera (Live Test)" to recognize faces in real time.
   - You will be asked which folder to load (default: "new_user_data").
   - The camera feed then tries to recognize any face it sees using the PCA models. 
     If the reconstruction error is too high, it labels as "Unknown."
   - Press 'q' in the camera window to quit.

3.3) Additional CLI Scripts (Optional)
--------------------------------------
You can also use a CLI approach with a script like main.py.
To run:

   python main.py

Typically, it:
- Builds PCA from 'data/'
- Tests on 'test/'
- Optionally performs camera capture or live testing.

-------------------------------------------
 4. HOW IT WORKS
-------------------------------------------
1) PCA (Eigenfaces):
   - Each subfolder in 'data/' is one "class" (person).
   - We load all .png images, convert them to grayscale, resize, and flatten into vectors.
   - PCA (via dual method and SVD) finds principal components ("eigenfaces").

2) Reconstruction & Classification:
   - A test image is projected into each person's PCA space, then reconstructed.
   - The reconstruction error (L2 norm) is computed. The smallest error indicates the best match.

3) Live Testing:
   - We use a Haar Cascade to detect faces in a video feed.
   - Each cropped face is projected into PCA models for all known users. 
   - If every reconstruction error is above a threshold, we label "Unknown."

-------------------------------------------
 5. CUSTOMIZATION
-------------------------------------------
- Change image size by modifying 'target_size=(width, height)' in 
  face_recognition_app.py or the relevant modules.
- Adjust PCA variance threshold (e.g. 95%) in test_pca_live or test_pca_reconstruction.
- Tweak the unknown threshold if you find too many false positives or negatives.
- Update cascade settings (scaleFactor, minNeighbors) in camera_capture.py or live_pca.py 
  to improve face detection in different conditions.

-------------------------------------------
 6. TROUBLESHOOTING
-------------------------------------------
- If plots close immediately, try running your code from a standard terminal 
  instead of certain IDEs or Jupyter notebooks. Also ensure you have a 
  GUI backend like 'TkAgg' for matplotlib.
- If camera fails to open, verify your webcam index or that it's not in use by another application.
- If you see "Please build PCA first!", confirm the 'data/' folder exists 
  and that the PCA models were built without errors.
- If real-time recognition is inaccurate, capture more images in varied lighting 
  and with different expressions.

