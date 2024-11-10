# Example of creating a GUI frame using Tkinter
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk, ImageFilter
import csv
import subprocess
import os

class CustomGUI:
    def __init__(self, image_path):
        self.root = tk.Tk()
        self.root.title("Hashcat GUI")
        
        # Set the size of the window
        self.root.geometry("800x600")  # Set an initial size, but no max size
        
        # Load the background image
        self.background_image = Image.open(image_path)
        self.background_photo = None
        
        # Create a frame that expands and fills the window
        self.frame = tk.Frame(self.root)
        self.frame.pack(expand=True, fill=tk.BOTH)
        
        # Configure grid to allow for resizing
        self.frame.grid_rowconfigure(0, weight=1)
        self.frame.grid_rowconfigure(1, weight=1)
        self.frame.grid_rowconfigure(2, weight=1)
        self.frame.grid_rowconfigure(3, weight=1)
        self.frame.grid_rowconfigure(4, weight=1)
        self.frame.grid_rowconfigure(5, weight=1)
        self.frame.grid_rowconfigure(6, weight=1)
        self.frame.grid_rowconfigure(7, weight=1)
        self.frame.grid_columnconfigure(0, weight=1)
        self.frame.grid_columnconfigure(1, weight=1)
        self.frame.grid_columnconfigure(2, weight=1)

        # Create a label for the background image
        self.background_label = tk.Label(self.frame)
        self.background_label.place(relwidth=1, relheight=1)
        
        # Load hash types from CSV
        self.hash_types = self.load_hash_types(r"SourceCode\GUI\Source Code GUI\Hash_Types\hash_types.csv")
        
        # Call the methods to create the hash file and dictionary file selection UI
        self.create_hash_file_selection()
        self.create_hash_type_selection()
        self.create_dict_file_selection()
        self.create_output_file_selection()  # Output file selection
        self.create_workload_profile_selection()  # Workload profile selection
        self.create_temperature_selection()  # Temperature selection
        
        # Add the Execute button at the end
        self.create_execute_button()

        # Initial display
        self.show_blurred_background()

        # Update the background image to fit the window
        self.update_background_image()

        # Bind the resize event to update the background image
        self.root.bind("<Configure>", self.update_background_image)

    def load_hash_types(self, file_path):
        hash_types = {}
        with open(file_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            for row in reader:
                if len(row) == 2 and row[1].strip():
                    hash_types[row[0]] = int(row[1])
        return hash_types

    def create_hash_file_selection(self):
        self.hash_file_label = tk.Label(self.frame, text="Select the Hash File:", fg="black", font=("Helvetica", 12, "bold"))
        self.hash_file_label.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        self.hash_file_entry = tk.Entry(self.frame, width=50, font=("Helvetica", 12))
        self.hash_file_entry.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        self.browse_hash_button = tk.Button(self.frame, text="Browse", command=self.browse_hash_file, font=("Helvetica", 12), bg="#4CAF50", fg="white", activebackground="#45a049")
        self.browse_hash_button.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")

    def create_hash_type_selection(self):
        self.hash_type_label = tk.Label(self.frame, text="Select Hash Type:", fg="black", font=("Helvetica", 12, "bold"))
        self.hash_type_label.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        
        self.hash_type_combobox = ttk.Combobox(self.frame, values=list(self.hash_types.keys()), state="readonly", width=72)
        self.hash_type_combobox.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
        
        if self.hash_types:
            self.hash_type_combobox.current(0)
        else:
            print("No hash types available.")

    def get_hash_file(self):
        hash_file = self.hash_file_entry.get()
        if not hash_file or not os.path.isfile(hash_file):
            messagebox.showerror("Error", "Invalid or empty hash file path.")
            return None
        return hash_file

    def get_hash_type_number(self):
        selected_hash_type = self.hash_type_combobox.get()
        if selected_hash_type in self.hash_types:
            hash_number = self.hash_types[selected_hash_type]
            print(f"Selected Hash Type: {selected_hash_type}, Corresponding Number: {hash_number}")
            return hash_number
        else:
            print("Selected hash type not found.")
            return None

    def create_dict_file_selection(self):
        self.dict_file_entries = []
        for i in range(3):
            dict_file_label = tk.Label(self.frame, text=f"Select Dictionary File {i + 1}:", fg="black", font=("Helvetica", 12, "bold"))
            dict_file_label.grid(row=i + 2, column=0, padx=10, pady=10, sticky="nsew")
            
            dict_file_entry = tk.Entry(self.frame, width=50, font=("Helvetica", 12))
            dict_file_entry.grid(row=i + 2, column=1, padx=10, pady=10, sticky="nsew")
            self.dict_file_entries.append(dict_file_entry)
            
            browse_dict_button = tk.Button(self.frame, text="Browse", command=lambda entry=dict_file_entry: self.browse_dict_file(entry), font=("Helvetica", 12), bg="#4CAF50", fg="white", activebackground="#45a049")
            browse_dict_button.grid(row=i + 2, column=2, padx=10, pady=10, sticky="nsew")

    def browse_hash_file(self):
        file_path = filedialog.askopenfilename(title="Select Hash File", filetypes=(("Text Files", "*.txt"), ("All Files", "*.*")))
        if file_path:
            self.hash_file_entry.delete(0, tk.END)
            self.hash_file_entry.insert(0, file_path)

    def browse_dict_file(self, entry):
        file_path = filedialog.askopenfilename(title="Select Dictionary File", filetypes=(("Text Files", "*.txt"), ("All Files", "*.*")))
        if file_path:
            entry.delete(0, tk.END)
            entry.insert(0, file_path)

    def create_output_file_selection(self):
        self.output_file_label = tk.Label(self.frame, text="Define Output File:", fg="black", font=("Helvetica", 12, "bold"))
        self.output_file_label.grid(row=5, column=0, padx=10, pady=10, sticky="nsew")
        
        self.output_file_entry = tk.Entry(self.frame, width=50, font=("Helvetica", 12))
        self.output_file_entry.grid(row=5, column=1, padx=10, pady=10, sticky="nsew")
        
        self.browse_output_file_button = tk.Button(self.frame, text="Browse", command=self.browse_output_file, font=("Helvetica", 12), bg="#4CAF50", fg="white", activebackground="#45a049")
        self.browse_output_file_button.grid(row=5, column=2, padx=10, pady=10, sticky="nsew")

    def browse_output_file(self):
        file_path = filedialog.asksaveasfilename(title="Define Output File", defaultextension=".txt", filetypes=(("Text Files", "*.txt"), ("All Files", "*.*")))
        if file_path:
            self.output_file_entry.delete(0, tk.END)
            self.output_file_entry.insert(0, file_path)

    def get_dictionary_files(self):
        dict_file_paths = [entry.get() for entry in self.dict_file_entries]
        
        if not dict_file_paths[0]:
            messagebox.showerror("Error", "Dictionary 1 is required.")
            return None
        
        command_string = ["-a", "0", dict_file_paths[0]]
        if dict_file_paths[1]:
            command_string.append(dict_file_paths[1])
        if dict_file_paths[2]:
            command_string.append(dict_file_paths[2])
        
        return command_string

    def get_output_file(self):
        output_file = self.output_file_entry.get()
        if not output_file:
            messagebox.showerror("Error", "Output file must be defined.")
            return None
        return output_file

    def create_workload_profile_selection(self):
        self.workload_profile_label = tk.Label(self.frame, text="Select Workload Profile:", fg="black", font=("Helvetica", 12, "bold"))
        self.workload_profile_label.grid(row=6, column=0, padx=10, pady=10, sticky="nsew")
        
        self.workload_profile_combobox = ttk.Combobox(self.frame, values=["1 - Low", "2 - Default", "3 - High", "4 - Nightmare"], state="readonly", width=70)
        self.workload_profile_combobox.grid(row=6, column=1, padx=10, pady=10, sticky="nsew")
        self.workload_profile_combobox.current(1)  # Default to "2 - Default"

    def get_workload_profile(self):
        selected_profile = self.workload_profile_combobox.get()
        if selected_profile:
            return selected_profile.split(" ")[0]  # Return the numeric part (e.g., "1", "2", "3", "4")
        return None

    def create_temperature_selection(self):
        self.temp_label = tk.Label(self.frame, text="Select Abort Temperature (°C):", fg="black", font=("Helvetica", 12, "bold"))
        self.temp_label.grid(row=7, column=0, padx=10, pady=10, sticky="nsew")
        
        self.temp_slider = tk.Scale(self.frame, from_=80, to=100, orient=tk.HORIZONTAL, length=300, command=self.update_temp_label)
        self.temp_slider.set(100)  # Default to 100
        self.temp_slider.grid(row=7, column=1, padx=10, pady=10, sticky="nsew")
        
        self.temp_value_label = tk.Label(self.frame, text="100 °C", fg="black", font=("Helvetica", 12))
        self.temp_value_label.grid(row=7, column=2, padx=10, pady=10, sticky="nsew")

    def update_temp_label(self, value):
        self.temp_value_label.config(text=f"{value} °C")

    def get_abort_temperature(self):
        return self.temp_slider.get()  # Return the current value of the temperature slider

    def show_blurred_background(self):
        blurred_image = self.background_image.filter(ImageFilter.GaussianBlur(radius=10))
        self.background_photo = ImageTk.PhotoImage(blurred_image)
        self.background_label.config(image=self.background_photo)

    def update_background_image(self, event=None):
        """Update the background image to fit the window size."""
        width = self.frame.winfo_width()
        height = self.frame.winfo_height()
        
        # Resize the original image to fit the window
        resized_image = self.background_image.resize((width, height), Image.LANCZOS)
        
        # Apply blur to the resized image
        blurred_image = resized_image.filter(ImageFilter.GaussianBlur(radius=50))  # Adjust radius as needed
        self.background_photo = ImageTk.PhotoImage(blurred_image)
        self.background_label.config(image=self.background_photo)

    def create_execute_button(self):
        execute_button = tk.Button(self.frame, text="Execute", command=self.run_hashcat, font=("Helvetica", 12), bg="#4CAF50", fg="white", activebackground="#45a049")
        execute_button.grid(row=8, column=1, padx=10, pady=20, sticky="nsew")

    def run_hashcat(self):
        """Run the Hashcat software with the selected options."""
        hash_file = self.get_hash_file()  # Get the hash file path and validate it
        if hash_file is None:
            return  # Exit if the hash file is invalid

        selected_hash_type = self.get_hash_type_number()  # Call the function to get the hash type number
        if selected_hash_type is None:
            return  # Exit if the hash type is not valid

        dict_args = self.get_dictionary_files()  # Get dictionary file paths as a formatted string
        if dict_args is None:
            return  # Exit if Dictionary 1 is not provided

        output_file = self.get_output_file()  # Get the output file path
        if output_file is None:
            return  # Exit if the output file is not defined

        workload_profile = self.get_workload_profile()  # Get the selected workload profile
        if workload_profile is None:
            return  # Exit if the workload profile is not defined

        abort_temp = self.get_abort_temperature()  # Get the selected abort temperature

        # Construct the command as a flat list
        command = [
            "SourceCode\\hashcat-6.2.6\\hashcat.exe",
            "-m", str(selected_hash_type),
            hash_file,
            "-w", workload_profile,  # Add workload profile option
            "-O",  # Add the -O option Enable optimized kernels (limits password length)
            f"--hwmon-temp-abort={abort_temp}"  # Add the abort temperature option
        ] + dict_args + ["-o", output_file]  # Add output file option

        print("Command to execute:", command)

        try:
            # Open Hashcat in a new console window without capturing output
            subprocess.Popen(
                command,
                cwd="SourceCode\\hashcat-6.2.6",
                creationflags=subprocess.CREATE_NEW_CONSOLE  # Open in a new console window
            )
        except Exception as e:
            print(f"Failed to execute Hashcat: {e}")

    def run(self):
        self.root.mainloop()

# Create an instance of the CustomGUI class and run it
gui = CustomGUI(r"SourceCode\GUI\Images\Background-image.jpg")
gui.run()
