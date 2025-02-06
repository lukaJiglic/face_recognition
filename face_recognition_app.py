import sys
import os
import shutil
import string
from PyQt5.QtWidgets import (
    QApplication, 
    QWidget, 
    QPushButton, 
    QVBoxLayout, 
    QLabel, 
    QInputDialog,
    QMessageBox
)
from PyQt5.QtCore import Qt
from pca_module import build_all_pca
from test_module import test_pca_reconstruction
from camera_capture import interactive_camera_capture
from live_pca import test_pca_live

class FaceRecognitionApp(QWidget):
    def __init__(self):
        super().__init__()

        # Window title, size
        self.setWindowTitle("Face Recognition Demo")
        self.setFixedSize(300, 250)

        # Create layout
        layout = QVBoxLayout()

        # Add a simple label at the top
        self.label = QLabel("Welcome to Face Recognition")
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

        # --- Button: Test PCA on /test images
        self.test_pca_button = QPushButton("Test on test/ images")
        self.test_pca_button.clicked.connect(self.test_on_test_dir)
        layout.addWidget(self.test_pca_button)

        # --- Button: Camera Capture (interactive)
        self.capture_button = QPushButton("Open Camera (Capture)")
        self.capture_button.clicked.connect(self.run_camera_capture)
        layout.addWidget(self.capture_button)

        # --- Button: Live PCA test
        self.live_test_button = QPushButton("Open Camera (Live Test)")
        self.live_test_button.clicked.connect(self.run_live_test)
        layout.addWidget(self.live_test_button)

        # --- Quit button
        self.quit_button = QPushButton("Quit")
        self.quit_button.clicked.connect(self.close)
        layout.addWidget(self.quit_button)

        self.setLayout(layout)

        # Keep track of built models in memory
        self.pca_models = None
        self.person_list = None

        # Build PCA automatically (no separate button)
        self.build_pca_from_data()

    def build_pca_from_data(self):
        """
        Automatically builds PCA from the 'data' folder when the app starts.
        """
        data_dir = "data"
        if not os.path.isdir(data_dir):
            self.label.setText("data/ folder not found!")
            return
        
        try:
            self.pca_models, self.person_list = build_all_pca(data_dir, target_size=(84, 96))
            self.label.setText("PCA built successfully!")
        except Exception as e:
            self.label.setText(f"Error building PCA: {e}")

    def test_on_test_dir(self):
        """
        Use test_pca_reconstruction on the 'test' folder.
        """
        if not self.pca_models or not self.person_list:
            self.label.setText("Please build PCA first!")
            return
        
        test_dir = "test"
        if not os.path.isdir(test_dir):
            self.label.setText("test/ folder not found!")
            return

        try:
            test_pca_reconstruction(test_dir, self.pca_models, self.person_list, target_size=(84, 96))
            self.label.setText("Test completed. Check console/plots.")
        except Exception as e:
            self.label.setText(f"Error testing PCA: {e}")

    def run_camera_capture(self):
        """
        Prompt the user for a new folder name. If blank or 'new_user_data',
        it will overwrite/create that folder. Otherwise, create or overwrite
        any specified folder (while forbidding certain special characters).
        Then run the interactive camera capture, storing images in that folder.
        """
        user_name, ok_pressed = QInputDialog.getText(
            self, "New User Name", "Enter your name (blank -> new_user_data):"
        )
        if not ok_pressed:
            # User canceled
            self.label.setText("Camera capture canceled.")
            return
        
        user_name = user_name.strip()
        if not user_name:
            # Default to "new_user_data" if blank
            user_name = "new_user_data"

        # Validate folder name: forbid certain special characters
        invalid_chars = set('/\\:*?"<>|')
        if any(ch in invalid_chars for ch in user_name):
            QMessageBox.warning(self, "Invalid Name", 
                                f"'{user_name}' contains invalid characters: / \\ : * ? \" < > |")
            return

        # Overwrite or create the folder
        if os.path.exists(user_name):
            try:
                shutil.rmtree(user_name)  # remove existing content
            except Exception as e:
                QMessageBox.warning(self, "Error", 
                                    f"Could not remove old folder '{user_name}': {e}")
                return

        try:
            os.makedirs(user_name)  # create new folder
        except Exception as e:
            QMessageBox.warning(self, "Error", 
                                f"Could not create folder '{user_name}': {e}")
            return

        # Run interactive camera capture in the chosen folder
        try:
            interactive_camera_capture(
                output_dir=user_name,
                recommended_count=20,
                face_size=(84, 96),
                cascade_path="haarcascade_frontalface_default.xml"
            )
            self.label.setText(f"Camera capture finished for '{user_name}'.")
        except Exception as e:
            self.label.setText(f"Error capturing: {e}")

    def run_live_test(self):
        """
        Let the user pick which folder to test for real-time recognition.
        If blank, default to 'new_user_data'. If the folder doesn't exist,
        show a warning. Otherwise, run test_pca_live on that folder.
        """
        user_folder, ok_pressed = QInputDialog.getText(
            self, "Live Test Folder", "Enter the folder to test (blank -> new_user_data):"
        )
        if not ok_pressed:
            self.label.setText("Live test canceled.")
            return

        user_folder = user_folder.strip()
        if not user_folder:
            user_folder = "new_user_data"

        if not os.path.isdir(user_folder):
            QMessageBox.warning(self, "Folder not found",
                                f"Folder '{user_folder}' does not exist.")
            return

        try:
            test_pca_live(
                pca_dir=user_folder,
                face_size=(84, 96),
                cascade_path="haarcascade_frontalface_default.xml",
                variance_threshold=0.95,
                unknown_threshold=15000
            )
            self.label.setText(f"Live test ended for folder: {user_folder}.")
        except Exception as e:
            self.label.setText(f"Error live testing: {e}")

def main():
    app = QApplication(sys.argv)
    window = FaceRecognitionApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
