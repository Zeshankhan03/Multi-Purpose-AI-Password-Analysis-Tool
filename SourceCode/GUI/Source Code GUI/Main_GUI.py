import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QLabel
from PyQt5.QtGui import QFont, QColor, QPalette
from PyQt5.QtCore import Qt

from Pass_Analysis import PasswordAnalysisApp
from DICT_ATTACK_GUI_2 import HashcatGUI

class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multipurpose AI Password Analysis Tool")
        self.setGeometry(133, 34, 1100, 700)
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.main_layout = QVBoxLayout(self.central_widget)
        self.create_widgets()

        self.password_analysis_window = None
        self.dict_attack_window = None

    def create_widgets(self):
        # Create top black half
        black_widget = QWidget()
        black_widget.setStyleSheet("background-color: black;")
        black_layout = QVBoxLayout(black_widget)

        # Create heading
        heading = QLabel("Multipurpose AI Password Analysis Tool")
        heading.setAlignment(Qt.AlignCenter)
        heading.setFont(QFont("Helvetica", 30, QFont.Bold))
        heading.setStyleSheet("color: white;")
        black_layout.addWidget(heading)

        self.main_layout.addWidget(black_widget)

        # Create bottom white half
        white_widget = QWidget()
        white_widget.setStyleSheet("background-color: white;")
        white_layout = QVBoxLayout(white_widget)

        # Create button container
        button_container = QWidget()
        button_container.setStyleSheet("background-color: white;")
        button_layout = QHBoxLayout(button_container)

        # Create Dictionary Attack button
        dictionary_button = QPushButton("Dictionary Attack")
        dictionary_button.setFixedSize(300, 100)
        dictionary_button.setFont(QFont("Helvetica", 20))
        dictionary_button.setStyleSheet("""
            QPushButton {
                background-color: black;
                color: white;
                font-weight: bold;
                border: 2px solid black;
            }
            QPushButton:hover {
                background-color: #333333;
            }
        """)
        dictionary_button.clicked.connect(self.open_dictionary_attack)
        button_layout.addWidget(dictionary_button)

        # Add spacing between buttons
        button_layout.addSpacing(40)

        # Create Password Analysis button
        analysis_button = QPushButton("Password Analysis")
        analysis_button.setFixedSize(300, 100)
        analysis_button.setFont(QFont("Helvetica", 20))
        analysis_button.setStyleSheet("""
            QPushButton {
                background-color: black;
                color: white;
                font-weight: bold;
                border: 2px solid black;
            }
            QPushButton:hover {
                background-color: #333333;
            }
        """)
        analysis_button.clicked.connect(self.open_password_analysis)
        button_layout.addWidget(analysis_button)

        white_layout.addWidget(button_container)
        self.main_layout.addWidget(white_widget)

        # Set the stretch factors to make the black and white halves equal
        self.main_layout.setStretch(0, 1)  # Black half
        self.main_layout.setStretch(1, 1)  # White half

    def open_dictionary_attack(self):
        self.hide()
        self.dict_attack_window = HashcatGUI(self)
        self.dict_attack_window.show()

    def open_password_analysis(self):
        self.hide()
        self.password_analysis_window = PasswordAnalysisApp(self)
        self.password_analysis_window.show()

    def show_main_window(self):
        if self.password_analysis_window:
            self.password_analysis_window.close()
            self.password_analysis_window = None
        if self.dict_attack_window:
            self.dict_attack_window.close()
            self.dict_attack_window = None
        self.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainApp()
    main_window.show()
    sys.exit(app.exec_())
