import tkinter as tk
from tkinter import filedialog, messagebox
import os
import pickle
import joblib

class PasswordAnalysisApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("AI-Powered Password Analysis")
        self.configure(bg="black")

        # Set window to fit the screen size with default title bar
        self.geometry(f"{self.winfo_screenwidth()}x{self.winfo_screenheight()}+0+0")

        # Create the heading for the app
        self.create_heading()

        # Create the file selection button to load the AI model
        self.create_file_selection_button()

    def create_heading(self):
        """Create the app heading label."""
        heading = tk.Label(self, text="AI-Powered Password Analysis", font=("Helvetica", 24, "bold"), fg="light cyan", bg="black")
        heading.pack(pady=20)

    def create_file_selection_button(self):
        """Create the button to browse for and load the AI model file."""
        load_model_button = tk.Button(self, text="Load AI Model", command=self.load_ai_model, font=("Helvetica", 16), bg="dark cyan", fg="black")
        load_model_button.pack(pady=50)

    def load_ai_model(self):
        """Load the pre-trained AI model based on the file size (use joblib for large models)."""
        model_file = filedialog.askopenfilename(filetypes=[("Model Files", "*.pkl")])
        
        if not model_file or not os.path.isfile(model_file):
            messagebox.showerror("Error", "Please select a valid AI model file.")
            return
        
        # Determine the file size in MB
        file_size = os.path.getsize(model_file) / (1024 * 1024)  # Convert bytes to MB

        try:
            if file_size > 100:  # Use joblib if the file is larger than 100 MB
                self.model = self.load_model_with_joblib(model_file)
                messagebox.showinfo("Success", f"Large AI Model loaded successfully with joblib! (Size: {round(file_size, 2)} MB)")
            else:  # Use pickle for smaller models
                self.model = self.load_model_with_pickle(model_file)
                messagebox.showinfo("Success", f"Small AI Model loaded successfully with pickle! (Size: {round(file_size, 2)} MB)")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")

    def load_model_with_joblib(self, file_path):
        """Load a large model using joblib."""
        return joblib.load(file_path)

    def load_model_with_pickle(self, file_path):
        """Load a small model using pickle."""
        with open(file_path, "rb") as f:
            return pickle.load(f)


if __name__ == "__main__":
    app = PasswordAnalysisApp()
    app.mainloop()

