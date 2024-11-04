import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QLabel,
                             QLineEdit, QFileDialog, QComboBox, QCheckBox, QSlider, QTextEdit, QMessageBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import subprocess
import os

class HashcatGUI(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setWindowTitle("Dictionary Attack")
        self.setGeometry(133, 34, 1100, 700)
        self.setStyleSheet("""
            QMainWindow {background-color: black;}
            QLabel {color: white; font-size: 12pt;}
            QLineEdit {background-color: black; color: white; border: 1px solid white;}
            QComboBox {background-color: black; color: white; border: 1px solid white;}
            QCheckBox {color: white;}
            QSlider {background-color: black;}
            QTextEdit {background-color: black; color: white; border: 1px solid white;}
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                border: none;
                padding: 5px 10px;
                font-size: 12pt;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QMessageBox {
                background-color: white;
            }
            QMessageBox QLabel {
                color: black;
                font-size: 12pt;
            }
            QMessageBox QPushButton {
                background-color: #4CAF50;
                color: white;
                min-width: 75px;
                min-height: 24px;
            }
        """)
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.main_layout = QVBoxLayout(self.central_widget)
        self.create_widgets()

    def create_widgets(self):
        # Main layout
        main_content = QVBoxLayout()
        
        # Add title at the top
        title_label = QLabel("AI-Powered Dictionary Attack")
        title_label.setFont(QFont("Helvetica", 16))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: white; margin: 10px 0;")  # Add margin for spacing
        main_content.addWidget(title_label)
        
        # Back button at top right
        back_button_layout = QHBoxLayout()
        back_button_layout.addStretch()  # Push button to right
        self.back_button = QPushButton("Back")
        self.back_button.setFixedSize(100, 40)
        self.back_button.clicked.connect(self.go_back)
        back_button_layout.addWidget(self.back_button)
        main_content.addLayout(back_button_layout)
        
        # Main content layout
        content_layout = QHBoxLayout()
        self.main_layout.addLayout(main_content)
        self.main_layout.addLayout(content_layout)

        # Left side - Options
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # Hash file selection
        left_layout.addWidget(QLabel("Select Hash File:"))
        hash_file_layout = QHBoxLayout()
        self.hash_file_entry = QLineEdit()
        hash_file_layout.addWidget(self.hash_file_entry)
        browse_hash_button = QPushButton("Browse")
        browse_hash_button.clicked.connect(self.browse_hash_file)
        hash_file_layout.addWidget(browse_hash_button)
        left_layout.addLayout(hash_file_layout)

        # Wordlist selection
        left_layout.addWidget(QLabel("Select Wordlist (Dictionary) File:"))
        wordlist_layout = QHBoxLayout()
        self.wordlist_entry = QLineEdit()
        wordlist_layout.addWidget(self.wordlist_entry)
        browse_wordlist_button = QPushButton("Browse")
        browse_wordlist_button.clicked.connect(self.browse_wordlist)
        wordlist_layout.addWidget(browse_wordlist_button)
        left_layout.addLayout(wordlist_layout)

        # Hash type selection
        left_layout.addWidget(QLabel("Select Hash Type:"))
        self.hash_type_combo = QComboBox()
        self.hash_types = {
            "MD5": 0, "SHA1": 100, "SHA256": 1400, "SHA512": 1700, "NTLM": 1000
        }
        self.hash_type_combo.addItems(self.hash_types.keys())
        left_layout.addWidget(self.hash_type_combo)

        # Additional options (moved from right side)
        self.gpu_checkbox = QCheckBox("Use GPU")
        left_layout.addWidget(self.gpu_checkbox)

        workload_layout = QHBoxLayout()
        workload_layout.addWidget(QLabel("Workload Profile (1 to 4):"))
        self.workload_entry = QLineEdit()
        self.workload_entry.setFixedWidth(50)
        workload_layout.addWidget(self.workload_entry)
        workload_layout.addStretch()
        left_layout.addLayout(workload_layout)

        self.temp_option_checkbox = QCheckBox("Use Temperature Abortion Threshold")
        left_layout.addWidget(self.temp_option_checkbox)

        left_layout.addWidget(QLabel("Temperature Abortion Threshold (°C):"))
        self.temp_slider = QSlider(Qt.Horizontal)
        self.temp_slider.setRange(70, 250)
        self.temp_slider.setValue(90)
        left_layout.addWidget(self.temp_slider)
        self.temp_value_label = QLabel(f"Current value: {self.temp_slider.value()}°C")
        left_layout.addWidget(self.temp_value_label)
        self.temp_slider.valueChanged.connect(self.update_temp_value)

        self.kernel_checkbox = QCheckBox("Use Optimized Kernel")
        left_layout.addWidget(self.kernel_checkbox)

        # Run Hashcat button
        run_button = QPushButton("Run Hashcat")
        run_button.clicked.connect(self.run_hashcat)
        left_layout.addWidget(run_button)

        # Add stretch to push everything up
        left_layout.addStretch()

        # Right side - Output
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        right_layout.addWidget(QLabel("Hashcat Output"))
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        right_layout.addWidget(self.output_text)

        # Add widgets to main content layout
        content_layout.addWidget(left_widget, 1)  # 1 part for options
        content_layout.addWidget(right_widget, 2)  # 2 parts for output

    def update_temp_value(self, value):
        self.temp_value_label.setText(f"Current value: {value}°C")

    def browse_hash_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Hash File")
        if file_path:
            self.hash_file_entry.setText(file_path)

    def browse_wordlist(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Wordlist File")
        if file_path:
            self.wordlist_entry.setText(file_path)

    def run_hashcat(self):
        hash_file = self.hash_file_entry.text()
        wordlist = self.wordlist_entry.text()
        selected_hash_type = self.hash_types[self.hash_type_combo.currentText()]
        gpu_enabled = self.gpu_checkbox.isChecked()
        workload_profile = self.workload_entry.text()
        temp_abort_enabled = self.temp_option_checkbox.isChecked()
        temp_abort_value = self.temp_slider.value()
        optimized_kernel = self.kernel_checkbox.isChecked()

        if not os.path.isfile(hash_file):
            msg = QMessageBox(self)  # Add parent
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("Error")
            msg.setText("Invalid hash file.")
            msg.setWindowModality(Qt.WindowModal)  # Make it modal
            msg.setStyleSheet("""
                QMessageBox {
                    background-color: white;
                }
                QMessageBox QLabel {
                    color: black;
                    font-size: 12pt;
                }
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    font-weight: bold;
                    border: none;
                    padding: 5px 10px;
                    font-size: 12pt;
                    min-width: 75px;
                    min-height: 24px;
                }
            """)
            msg.exec_()
            return
        if not os.path.isfile(wordlist):
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("Error")
            msg.setText("Invalid wordlist file.")
            msg.setStyleSheet("""
                QMessageBox {
                    background-color: white;
                }
                QLabel {
                    color: black;
                    font-size: 12pt;
                }
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    font-weight: bold;
                    border: none;
                    padding: 5px 10px;
                    font-size: 12pt;
                    min-width: 75px;
                    min-height: 24px;
                }
            """)
            msg.exec_()
            return

        command = [f"I:\Project AI Password\Multi-Purpose-AI-Password-Analysis-Tool\SourceCode\hashcat-6.2.6.\hashcat.exe", "-m", str(selected_hash_type), hash_file, wordlist]

        if gpu_enabled:
            command.append("--opencl-device-types=1")
        else:
            command.append("--opencl-device-types=2")

        if workload_profile.isdigit() and 1 <= int(workload_profile) <= 4:
            command.extend(["-w", workload_profile])
        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("Error")
            msg.setText("Workload profile must be between 1 and 4.")
            msg.setStyleSheet("""
                QMessageBox {
                    background-color: white;
                }
                QLabel {
                    color: black;
                    font-size: 12pt;
                }
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    font-weight: bold;
                    border: none;
                    padding: 5px 10px;
                    font-size: 12pt;
                    min-width: 75px;
                    min-height: 24px;
                }
            """)
            msg.exec_()
            return

        if optimized_kernel:
            command.append("--optimized-kernel-enable")

        if temp_abort_enabled:
            command.extend(["--gpu-temp-abort", str(temp_abort_value)])

        file_size = os.path.getsize(wordlist)
        self.output_text.append(f"Estimated time: {file_size // 1024} seconds")

        try:
            # Create a new console window for Hashcat
            startupinfo = None
            if os.name == 'nt':  # Windows
                import subprocess
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                creationflags = subprocess.CREATE_NEW_CONSOLE
            else:  # Unix-like systems
                creationflags = 0

            self.output_text.append(f"Starting Hashcat in a new window...")
            process = subprocess.Popen(
                command,
                creationflags=creationflags,
                startupinfo=startupinfo
            )
            
            # Don't wait for the process to complete
            self.output_text.append("Hashcat is running in a separate window. Please check the other window for progress.")
            
        except Exception as e:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("Error")
            msg.setText(f"Failed to execute Hashcat: {e}")
            msg.setStyleSheet("""
                QMessageBox {
                    background-color: white;
                }
                QLabel {
                    color: black;
                    font-size: 12pt;
                }
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    font-weight: bold;
                    border: none;
                    padding: 5px 10px;
                    font-size: 12pt;
                    min-width: 75px;
                    min-height: 24px;
                }
            """)
            msg.exec_()

    def go_back(self):
        if self.parent:
            self.parent.show()
        self.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HashcatGUI()
    window.show()
    sys.exit(app.exec_())
