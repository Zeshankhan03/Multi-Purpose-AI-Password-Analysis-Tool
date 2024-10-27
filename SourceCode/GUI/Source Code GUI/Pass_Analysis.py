import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QFileDialog, QMessageBox
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
import os
import pickle
import joblib

class PasswordAnalysisApp(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setWindowTitle("AI-Powered Password Analysis")
        self.setGeometry(133, 34, 1100, 700)
        self.setStyleSheet("""
            QMainWindow {background-color: black;}
            QLabel {color: white; font-size: 12pt;}
            QPushButton {
                background-color: dark cyan;
                color: white;
                font-weight: bold;
                border: none;
                padding: 5px 10px;
                font-size: 12pt;
            }
            QPushButton:hover {
                background-color: #008B8B;
            }
        """)
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.layout = QVBoxLayout(self.central_widget)
        self.model = None
        self.setup_ui()

    def setup_ui(self):
        self.create_back_button()
        self.create_heading()
        self.create_file_selection_button()

    def create_back_button(self):
        back_button = QPushButton("Back")
        back_button.setFont(QFont("Helvetica", 12))
        back_button.setStyleSheet("background-color: dark cyan; color: white;")
        back_button.clicked.connect(self.go_back)
        back_button.setFixedSize(100, 40)
        self.layout.addWidget(back_button, alignment=Qt.AlignTop | Qt.AlignRight)

    def go_back(self):
        if self.parent:
            self.parent.show_main_window()
        self.close()

    def create_heading(self):
        heading = QLabel("AI-Powered Password Analysis")
        heading.setFont(QFont("Helvetica", 24, QFont.Bold))
        heading.setStyleSheet("color: white;")
        heading.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(heading)

    def create_file_selection_button(self):
        load_model_button = QPushButton("Load AI Model")
        load_model_button.setFont(QFont("Helvetica", 16))
        load_model_button.setStyleSheet("background-color: dark cyan; color: white;")
        load_model_button.clicked.connect(self.load_ai_model)
        load_model_button.setFixedSize(200, 60)
        self.layout.addWidget(load_model_button, alignment=Qt.AlignCenter)

    def load_ai_model(self):
        model_file, _ = QFileDialog.getOpenFileName(self, "Select AI Model", "", "Model Files (*.pkl)")
        
        if not model_file or not os.path.isfile(model_file):
            QMessageBox.critical(self, "Error", "Please select a valid AI model file.")
            return
        
        file_size = os.path.getsize(model_file) / (1024 * 1024)  # Convert bytes to MB

        try:
            if file_size > 100:  # Use joblib if the file is larger than 100 MB
                self.model = self.load_model_with_joblib(model_file)
                QMessageBox.information(self, "Success", f"Large AI Model loaded successfully with joblib! (Size: {round(file_size, 2)} MB)")
            else:  # Use pickle for smaller models
                self.model = self.load_model_with_pickle(model_file)
                QMessageBox.information(self, "Success", f"Small AI Model loaded successfully with pickle! (Size: {round(file_size, 2)} MB)")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")

    def load_model_with_joblib(self, file_path):
        return joblib.load(file_path)

    def load_model_with_pickle(self, file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PasswordAnalysisApp()
    window.show()
    sys.exit(app.exec_())
