import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, 
                            QWidget, QLabel, QFileDialog, QMessageBox, QLineEdit, QProgressBar, 
                            QTextEdit, QFrame, QCheckBox, QComboBox, QDialog)
from PyQt5.QtGui import QFont, QColor
from PyQt5.QtCore import Qt
import os
import pickle
import joblib
import re
import json
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler

class PasswordAnalysisApp(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setWindowTitle("AI-Powered Password Analysis")
        self.setGeometry(133, 34, 1100, 700)
        self.setAttribute(Qt.WA_DeleteOnClose, False)
        
        # Update the base styling to match Dictionary Attack GUI
        self.setStyleSheet("""
            QMainWindow {
                background-color: black;
            }
            QLabel {
                color: white;
                font-size: 12pt;
            }
            QLineEdit {
                background-color: black;
                color: white;
                border: 1px solid white;
                padding: 5px;
                font-size: 12pt;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 5px 10px;
                font-size: 12pt;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QTextEdit {
                background-color: black;
                color: white;
                border: 1px solid white;
                font-size: 12pt;
            }
            QProgressBar {
                border: 1px solid white;
                background-color: black;
                text-align: center;
                padding: 2px;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
            }
            QComboBox {
                background-color: black;
                color: white;
                border: 1px solid white;
                padding: 5px;
                font-size: 12pt;
            }
            QCheckBox {
                color: white;
                font-size: 12pt;
                spacing: 5px;
            }
            QCheckBox::indicator {
                width: 13px;
                height: 13px;
                background-color: black;
                border: 1px solid white;
            }
            QCheckBox::indicator:checked {
                background-color: #4CAF50;
            }
        """)
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.layout = QVBoxLayout(self.central_widget)
        self.model = None
        self.setup_ui()
        self.setup_logging()  # Add this line

    def go_back(self):
        if self.parent:
            self.parent.show()
        self.close()

    def create_back_button(self):
        back_button = QPushButton("Back")
        back_button.setFixedSize(100, 40)
        back_button.clicked.connect(self.go_back)
        self.layout.addWidget(back_button, alignment=Qt.AlignTop | Qt.AlignRight)

    def create_heading(self):
        heading = QLabel("AI-Powered Password Analysis")
        heading.setFont(QFont("Helvetica", 16))  # Reduced from 24pt
        heading.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(heading)

    def create_file_selection_button(self, layout):
        load_model_button = QPushButton("Load AI Model")
        load_model_button.setFixedSize(300, 80)
        load_model_button.clicked.connect(self.load_ai_model)
        layout.addWidget(load_model_button)

    def setup_ui(self):
        # Create a layout for the top section
        top_layout = QHBoxLayout()
        
        # Add title with spacing
        heading = QLabel("AI-Powered Password Analysis")
        heading.setFont(QFont("Helvetica", 16))
        heading.setAlignment(Qt.AlignCenter)
        heading.setStyleSheet("margin: 10px 0;")  # Add margin for spacing
        
        # Add back button
        back_button = QPushButton("Back")
        back_button.setFixedSize(100, 40)
        back_button.clicked.connect(self.go_back)
        
        # Add empty label for spacing on left to center the title
        empty_label = QLabel()
        empty_label.setFixedSize(100, 40)
        
        top_layout.addWidget(empty_label)
        top_layout.addWidget(heading, 1)  # 1 for stretch factor
        top_layout.addWidget(back_button)
        
        self.layout.addLayout(top_layout)
        
        # Add spacing after title
        self.layout.addSpacing(20)
        
        # Create main content area
        content_layout = QHBoxLayout()
        
        # Left side - Input and Controls
        left_panel = QVBoxLayout()
        
        # Password Input Methods Group
        input_method_label = QLabel("Password Input Method:")
        input_method_label.setFont(QFont("Helvetica", 14))
        left_panel.addWidget(input_method_label)
        
        # Input method buttons
        input_buttons_layout = QHBoxLayout()
        
        self.file_input_button = QPushButton("Load from File")
        self.file_input_button.setFixedWidth(200)
        self.file_input_button.clicked.connect(self.load_password_file)
        left_panel.addWidget(self.file_input_button)
        
        left_panel.addSpacing(20)
        
        # AI Model loading (Optional)
        ai_label = QLabel("AI Enhancement (Optional):")
        ai_label.setFont(QFont("Helvetica", 14))
        left_panel.addWidget(ai_label)
        
        self.create_file_selection_button(left_panel)
        left_panel.addSpacing(20)
        
        # Password input
        self.password_label = QLabel("Enter Password:")
        self.password_label.setFont(QFont("Helvetica", 12))  # Reduced from 14pt
        left_panel.addWidget(self.password_label)
        
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setFixedWidth(400)
        left_panel.addWidget(self.password_input)
        
        # Properly labeled show/hide password checkbox
        self.show_password = QCheckBox("Show Password")
        self.show_password.setFont(QFont("Helvetica", 12))
        self.show_password.setStyleSheet("""
            QCheckBox {
                color: white;
                spacing: 5px;
            }
            QCheckBox::indicator {
                width: 13px;
                height: 13px;
                background-color: white;
            }
            QCheckBox::indicator:checked {
                background-color: #4CAF50;
            }
        """)
        self.show_password.stateChanged.connect(self.toggle_password_visibility)
        left_panel.addWidget(self.show_password)
        
        # Analyze Password button immediately after password input
        analyze_button = QPushButton("Analyze Password")
        analyze_button.setFixedSize(200, 30)  # Smaller size to match Dictionary Attack
        analyze_button.setFont(QFont("Helvetica", 12))
        analyze_button.clicked.connect(self.analyze_password)
        left_panel.addWidget(analyze_button)
        
        # Password strength section with updated styling
        strength_label = QLabel("Password Strength:")
        strength_label.setFont(QFont("Helvetica", 14))
        left_panel.addWidget(strength_label)
        
        self.strength_bar = QProgressBar()
        self.strength_bar.setFixedHeight(20)  # Thinner progress bar
        self.strength_bar.setFixedWidth(300)  # Match input width
        self.strength_bar.setMinimum(0)
        self.strength_bar.setMaximum(100)
        self.strength_bar.setValue(0)  # Initialize to 0
        self.strength_bar.setTextVisible(True)
        self.strength_bar.setFormat("%p%")
        self.strength_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #4CAF50;
                background-color: white;
                text-align: center;
                padding: 2px;
                height: 30px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
            }
        """)
        left_panel.addWidget(self.strength_bar)
        
        left_panel.addStretch()  # Push everything up
        
        # Right panel - Results and buttons
        right_panel = QVBoxLayout()
        
        results_label = QLabel("Analysis Results:")
        results_label.setFont(QFont("Helvetica", 14))
        right_panel.addWidget(results_label)
        
        self.results_display = QTextEdit()
        self.results_display.setReadOnly(True)
        self.results_display.setMinimumWidth(500)
        right_panel.addWidget(self.results_display)
        
        # Add View Logs and View Outputs buttons at the bottom of right panel
        buttons_layout = QHBoxLayout()
        
        view_logs_button = QPushButton("View Logs")
        view_logs_button.setFixedSize(190, 50)
        view_logs_button.setFont(QFont("Helvetica", 12))
        view_logs_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                border: none;
                padding: 10px;
                text-align: center;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        view_logs_button.clicked.connect(self.show_logs)
        buttons_layout.addWidget(view_logs_button)
        
        buttons_layout.addSpacing(20)  # Space between buttons
        
        view_outputs_button = QPushButton("View Outputs")
        view_outputs_button.setFixedSize(190, 50)
        view_outputs_button.setFont(QFont("Helvetica", 12))
        view_outputs_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                border: none;
                padding: 10px;
                text-align: center;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        view_outputs_button.clicked.connect(self.show_outputs)
        buttons_layout.addWidget(view_outputs_button)
        
        right_panel.addLayout(buttons_layout)
        
        # Add layouts to main content layout
        content_layout.addLayout(left_panel, 1)  # 1 part for left panel
        content_layout.addLayout(right_panel, 2)  # 2 parts for right panel
        
        self.layout.addLayout(content_layout)

    def toggle_password_visibility(self, state):
        if state == Qt.Checked:
            self.password_input.setEchoMode(QLineEdit.Normal)
        else:
            self.password_input.setEchoMode(QLineEdit.Password)

    def show_manual_input(self):
        self.password_input.clear()
        self.password_input.setEnabled(True)
        self.password_label.setText("Enter Password:")

    def load_password_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Password File", "", "Text Files (*.txt)")
        if file_path:
            try:
                with open(file_path, 'r') as file:
                    password = file.read().strip()
                    self.password_input.setText(password)
                    self.password_label.setText("Password loaded from file:")
            except Exception as e:
                self.show_message("Error", f"Failed to load password file: {str(e)}", QMessageBox.Critical)

    def load_ai_model(self):
        model_file, _ = QFileDialog.getOpenFileName(self, "Select AI Model", "", "Model Files (*.pkl)")
        
        if not model_file or not os.path.isfile(model_file):
            self.show_message("Error", "Please select a valid AI model file.", QMessageBox.Critical)
            return
        
        file_size = os.path.getsize(model_file) / (1024 * 1024)  # Convert bytes to MB

        try:
            if file_size > 100:  # Use joblib if the file is larger than 100 MB
                self.model = self.load_model_with_joblib(model_file)
                self.show_message("Success", f"Large AI Model loaded successfully with joblib! (Size: {round(file_size, 2)} MB)")
            else:  # Use pickle for smaller models
                self.model = self.load_model_with_pickle(model_file)
                self.show_message("Success", f"Small AI Model loaded successfully with pickle! (Size: {round(file_size, 2)} MB)")
        except Exception as e:
            self.show_message("Error", f"Failed to load model: {str(e)}", QMessageBox.Critical)

    def load_model_with_joblib(self, file_path):
        return joblib.load(file_path)

    def load_model_with_pickle(self, file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)

    def analyze_password(self):
        password = self.password_input.text()
        if not password:
            self.show_message("Error", "Please enter a password or load from file", QMessageBox.Critical)
            return

        # Calculate strength score
        strength_score = self.check_password_strength(password)
        
        # Update progress bar
        self.strength_bar.setValue(strength_score)
        
        # Set progress bar color based on strength
        self.update_strength_bar_style(strength_score)
        
        # Log the analysis start
        logging.info("Starting password analysis")

        # Perform analysis
        vulnerabilities = self.check_vulnerabilities(password)
        patterns = self.analyze_patterns(password)
        recommendations = self.generate_recommendations(password, vulnerabilities, patterns)
        
        # Prepare results for saving
        analysis_results = {
            "timestamp": datetime.now().isoformat(),
            "strength_score": strength_score,
            "vulnerabilities": vulnerabilities,
            "patterns": patterns,
            "recommendations": recommendations
        }
        
        # Save results
        self.save_analysis_output(analysis_results)
        
        # Display results
        self.display_analysis_results(strength_score, vulnerabilities, patterns, recommendations)
        
        # Log completion
        logging.info("Password analysis completed")

    def check_password_strength(self, password):
        score = 0
        if len(password) >= 8: score += 20
        if re.search(r"[A-Z]", password): score += 20
        if re.search(r"[a-z]", password): score += 20
        if re.search(r"\d", password): score += 20
        if re.search(r"[!@#$%^&*(),.?\":{}|<>]", password): score += 20
        return score

    def check_vulnerabilities(self, password):
        vulnerabilities = []
        if len(password) < 8:
            vulnerabilities.append("Password is too short (minimum 8 characters recommended)")
        if not re.search(r"[A-Z]", password):
            vulnerabilities.append("Missing uppercase letters")
        if not re.search(r"[a-z]", password):
            vulnerabilities.append("Missing lowercase letters")
        if not re.search(r"\d", password):
            vulnerabilities.append("Missing numbers")
        if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
            vulnerabilities.append("Missing special characters")
        return vulnerabilities

    def analyze_patterns(self, password):
        patterns = []
        if re.search(r"(.)\1{2,}", password):
            patterns.append("Contains repeated characters")
        if re.search(r"(123|abc|qwerty)", password.lower()):
            patterns.append("Contains common sequences")
        if password.lower() in ["password", "admin", "user"]:
            patterns.append("Common password detected")
        return patterns

    def generate_recommendations(self, password, vulnerabilities, patterns):
        recommendations = []
        if vulnerabilities:
            recommendations.append("Address the following vulnerabilities:")
            recommendations.extend([f"- {v}" for v in vulnerabilities])
        
        if patterns:
            recommendations.append("\nAvoid common patterns:")
            recommendations.extend([f"- {p}" for p in patterns])
            
        if not recommendations:
            recommendations.append("Password meets basic security requirements")
            
        return recommendations

    def display_analysis_results(self, strength_score, vulnerabilities, patterns, recommendations, ai_results=None):
        results = """=== Password Analysis Report ===\n\n"""
        
        # Strength Score
        results += f"1. Password Strength Score: {strength_score}%\n"
        results += self.get_strength_description(strength_score) + "\n\n"
        
        # Vulnerability Section
        results += "2. Vulnerability Analysis:\n"
        if vulnerabilities:
            results += "\n".join([f"• {v}" for v in vulnerabilities])
        else:
            results += "• No critical vulnerabilities found"
        results += "\n\n"
        
        # Pattern Recognition Section
        results += "3. Pattern Recognition:\n"
        if patterns:
            results += "\n".join([f"• {p}" for p in patterns])
        else:
            results += "• No concerning patterns detected"
        results += "\n\n"
        
        # AI Analysis Results (if available)
        if ai_results:
            results += "4. AI-Enhanced Analysis:\n"
            results += "\n".join([f"• {r}" for r in ai_results])
            results += "\n\n"
        
        # Recommendations
        results += f"{'5' if ai_results else '4'}. Recommendations:\n"
        results += "\n".join([f"• {r}" for r in recommendations])
        
        self.results_display.setText(results)

    def get_strength_description(self, score):
        if score < 20:
            return "Very Weak - Password is extremely vulnerable"
        elif score < 40:
            return "Weak - Password needs significant improvement"
        elif score < 60:
            return "Moderate - Password meets basic requirements but could be stronger"
        elif score < 80:
            return "Strong - Password is good but has room for improvement"
        else:
            return "Very Strong - Password meets all security requirements"

    def closeEvent(self, event):
        if self.parent:
            self.parent.show()
        event.accept()

    def show_message(self, title, message, icon=QMessageBox.Information):
        msg = QMessageBox(self)
        msg.setIcon(icon)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.setWindowModality(Qt.WindowModal)
        msg.setStyleSheet("""
            QMessageBox {
                background-color: white;
            }
            QMessageBox QLabel {
                color: black;
                font-size: 12pt;
                min-width: 300px;
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
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        msg.exec_()

    # Add these new methods for logging and output handling
    def setup_logging(self):
        """Setup logging configuration"""
        try:
            # Create base logs directory
            base_log_dir = "SourceCode/Logs"
            os.makedirs(base_log_dir, exist_ok=True)
            
            # Create specific logs directory for Password Analysis
            analysis_log_dir = os.path.join(base_log_dir, "Passwords_Analysis_logs")
            os.makedirs(analysis_log_dir, exist_ok=True)
            
            # Create run-specific directory with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            run_log_dir = os.path.join(analysis_log_dir, f"run_{timestamp}")
            os.makedirs(run_log_dir, exist_ok=True)
            
            # Create log file
            log_file = os.path.join(run_log_dir, 'password_analysis.log')
            
            # Configure logging with RotatingFileHandler
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5,
                encoding='utf-8'
            )
            
            # Set format
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M%S'
            )
            file_handler.setFormatter(formatter)
            
            # Configure logger
            logger = logging.getLogger('PasswordAnalysis')
            logger.setLevel(logging.INFO)
            logger.addHandler(file_handler)
            
            # Create outputs directory
            outputs_dir = os.path.join("SourceCode", "Analyzed_Passwords_Outputs")
            os.makedirs(outputs_dir, exist_ok=True)
            
            # Store configuration for later use
            self.log_config = type('LogConfig', (), {
                'logs_dir': analysis_log_dir,
                'outputs_dir': outputs_dir
            })()
            
            # Log initial setup information
            logging.info("Logging initialized")
            logging.info(f"Log directory: {run_log_dir}")
            logging.info(f"Outputs directory: {outputs_dir}")
            
            return log_file
            
        except Exception as e:
            self.show_message("Error", f"Could not set up logging: {str(e)}", QMessageBox.Critical)
            return None

    def save_analysis_output(self, analysis_results):
        """Save analysis results to output file"""
        try:
            # Ensure output directory exists
            os.makedirs(self.log_config.outputs_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = os.path.join(
                self.log_config.outputs_dir,
                f"analysis_result_{timestamp}.json"
            )
            
            with open(output_file, 'w') as f:
                json.dump(analysis_results, f, indent=4)
            
            logging.info(f"Analysis results saved to {output_file}")
            
        except Exception as e:
            error_msg = f"Error saving analysis results: {str(e)}"
            logging.error(error_msg)
            self.show_message("Error", error_msg, QMessageBox.Critical)

    def show_logs(self):
        """Show logs in a new window"""
        self.show_file_viewer("Log Viewer", self.log_config.logs_dir)

    def show_outputs(self):
        """Show analysis outputs in a new window"""
        self.show_file_viewer("Analysis Outputs", self.log_config.outputs_dir)

    def show_file_viewer(self, title, directory):
        """Show file viewer window"""
        viewer = QDialog(self)
        viewer.setWindowTitle(title)
        viewer.setMinimumSize(800, 600)
        
        layout = QVBoxLayout()
        
        # File selector
        file_selector = QComboBox()
        files = sorted([f for f in os.listdir(directory) if f.endswith('.log') or f.endswith('.json')])
        file_selector.addItems(files)
        layout.addWidget(file_selector)
        
        # Content display
        content_display = QTextEdit()
        content_display.setReadOnly(True)
        layout.addWidget(content_display)
        
        def load_selected_file():
            selected = file_selector.currentText()
            if selected:
                file_path = os.path.join(directory, selected)
                with open(file_path, 'r') as f:
                    content = f.read()
                    content_display.setText(content)
        
        file_selector.currentTextChanged.connect(load_selected_file)
        if files:
            load_selected_file()
        
        viewer.setLayout(layout)
        viewer.exec_()

    def update_strength_bar_style(self, score):
        """Update progress bar color based on password strength"""
        style = """
        QProgressBar {
            border: 2px solid #4CAF50;
            background-color: white;
            text-align: center;
            padding: 2px;
            height: 30px;
        }
        QProgressBar::chunk {
            background-color: %s;
        }
        """
        
        if score < 20:
            color = "#FF0000"  # Red
        elif score < 40:
            color = "#FF8C00"  # Dark Orange
        elif score < 60:
            color = "#FFD700"  # Gold
        elif score < 80:
            color = "#90EE90"  # Light Green
        else:
            color = "#4CAF50"  # Green
        
        self.strength_bar.setStyleSheet(style % color)
        self.strength_bar.setFormat(f"{score}%")
        self.strength_bar.setTextVisible(True)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PasswordAnalysisApp()
    window.show()
    sys.exit(app.exec_())
