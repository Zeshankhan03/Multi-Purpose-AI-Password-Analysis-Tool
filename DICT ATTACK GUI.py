import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import subprocess
import os
from PIL import Image, ImageTk


class HashcatGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Multi-purpose AI Password Analysis Tool - Dictionary Attack")
        self.geometry("700x850")

        # Load background image (JPEG format)
        self.background_image = Image.open("Background-image.jpg")  # Save your image as background.jpg in the project directory
        self.background_image = self.background_image.resize((700, 850), Image.LANCZOS)  # Resize to fit the window
        self.background_photo = ImageTk.PhotoImage(self.background_image)
        
        self.background_label = tk.Label(self, image=self.background_photo)
        self.background_label.place(relwidth=1, relheight=1)  # Set image to fill the window

        # Set theme to black with light cyan text
        self.configure(bg="black")

        # Hash File Selection
        self.hash_file_label = tk.Label(self, text="Select Hash File:", fg="light cyan", bg="black")
        self.hash_file_label.pack()
        self.hash_file_entry = tk.Entry(self, width=40, bg="black", fg="light cyan")
        self.hash_file_entry.pack()
        self.browse_hash_button = tk.Button(self, text="Browse", command=self.browse_hash_file, bg="black", fg="light cyan")
        self.browse_hash_button.pack()

        # Wordlist (Dictionary) Selection
        self.wordlist_label = tk.Label(self, text="Select Wordlist (Dictionary) File:", fg="light cyan", bg="black")
        self.wordlist_label.pack()
        self.wordlist_entry = tk.Entry(self, width=40, bg="black", fg="light cyan")
        self.wordlist_entry.pack()
        self.browse_wordlist_button = tk.Button(self, text="Browse", command=self.browse_wordlist, bg="black", fg="light cyan")
        self.browse_wordlist_button.pack()

        # Hash Type Selection (Dropdown)
        self.hash_type_label = tk.Label(self, text="Select Hash Type:", fg="light cyan", bg="black")
        self.hash_type_label.pack()

        # List of common hash types (expandable)
        self.hash_types = {
            "MD5": 0,
            "SHA1": 100,
            "SHA256": 1400,
            "SHA512": 1700,
            "NTLM": 1000,
            "bcrypt": 3200,
            "MySQL323": 200,
            "SHA3-256": 17600,
            "WPA/WPA2": 2500,
            "RIPEMD160": 6000,
            "MSSQL(2005)": 132,
            "Oracle": 3100,
            "Whirlpool": 6100,
            "vBulletin": 2711
        }

        # Tkinter variable to store selected hash type
        self.hash_type_var = tk.StringVar(self)
        self.hash_type_var.set("MD5")  # Default value

        # Dropdown menu for hash types
        self.hash_type_menu = tk.OptionMenu(self, self.hash_type_var, *self.hash_types.keys())
        self.hash_type_menu.config(bg="black", fg="light cyan", highlightbackground="black")
        self.hash_type_menu.pack()

        # GPU Selection
        self.gpu_label = tk.Label(self, text="Use GPU:", fg="light cyan", bg="black")
        self.gpu_label.pack()
        self.gpu_var = tk.BooleanVar(value=True)  # Default to True
        self.gpu_checkbox = tk.Checkbutton(self, text="Enable GPU", variable=self.gpu_var, bg="black", fg="light cyan", selectcolor="light cyan")
        self.gpu_checkbox.pack()

        # Workload Profile
        self.workload_label = tk.Label(self, text="Workload Profile (1 to 4):", fg="light cyan", bg="black")
        self.workload_label.pack()
        self.workload_entry = tk.Entry(self, width=5, bg="black", fg="light cyan")
        self.workload_entry.pack()

        # Temperature Monitoring Option
        self.temp_option_label = tk.Label(self, text="Use Temperature Abortion Threshold:", fg="light cyan", bg="black")
        self.temp_option_label.pack()
        self.temp_option_var = tk.BooleanVar(value=False)  # Default to False
        self.temp_option_checkbox = tk.Checkbutton(self, text="Enable", variable=self.temp_option_var, bg="black", fg="light cyan", selectcolor="light cyan")
        self.temp_option_checkbox.pack()

        # Temperature Abortion Threshold Slider
        self.temp_label = tk.Label(self, text="Temperature Abortion Threshold (°C):", fg="light cyan", bg="black")
        self.temp_label.pack()
        self.temp_slider = tk.Scale(self, from_=70, to=100, orient=tk.HORIZONTAL, bg="black", fg="light cyan")
        self.temp_slider.pack()

        # Kernel Execution
        self.kernel_label = tk.Label(self, text="Optimized Kernel:", fg="light cyan", bg="black")
        self.kernel_label.pack()
        self.kernel_var = tk.BooleanVar(value=True)  # Default to True
        self.kernel_checkbox = tk.Checkbutton(self, text="Use Optimized Kernel", variable=self.kernel_var, bg="black", fg="light cyan", selectcolor="light cyan")
        self.kernel_checkbox.pack()

        # Output and Run Button
        self.output_label = tk.Label(self, text="Hashcat Output:", fg="light cyan", bg="black")
        self.output_label.pack()
        self.output_text = tk.Text(self, height=10, width=60, bg="black", fg="light cyan")
        self.output_text.pack()

        # Run Hashcat Button
        self.run_button = tk.Button(self, text="Run Hashcat", command=self.run_hashcat, bg="black", fg="light cyan")
        self.run_button.pack()

    def browse_hash_file(self):
        """Browse for the hash file."""
        file_path = filedialog.askopenfilename(filetypes=[("Hash Files", "*.*")])
        self.hash_file_entry.insert(0, file_path)

    def browse_wordlist(self):
        """Browse for the wordlist file."""
        file_path = filedialog.askopenfilename(filetypes=[("Wordlist Files", "*.*")])
        self.wordlist_entry.insert(0, file_path)

    def run_hashcat(self):
        """Execute the Hashcat command based on the user's selections."""
        hash_file = self.hash_file_entry.get()
        wordlist = self.wordlist_entry.get()
        selected_hash_type = self.hash_types[self.hash_type_var.get()]  # Get selected hash type
        gpu_enabled = self.gpu_var.get()
        workload_profile = self.workload_entry.get()
        temp_abort_enabled = self.temp_option_var.get()  # Check if temperature abortion is enabled
        temp_abort_value = self.temp_slider.get()  # Get temperature abortion threshold
        optimized_kernel = self.kernel_var.get()

        # Print the state of the checkboxes
        print(f"GPU Enabled: {gpu_enabled}, Temperature Abort Enabled: {temp_abort_enabled}, Optimized Kernel: {optimized_kernel}")

        # Validate inputs
        if not os.path.isfile(hash_file):
            messagebox.showerror("Error", "Invalid hash file.")
            return
        if not os.path.isfile(wordlist):
            messagebox.showerror("Error", "Invalid wordlist file.")
            return

        # Construct the Hashcat command
        command = ["hashcat", "-m", str(selected_hash_type), hash_file, wordlist]

        # GPU settings
        if gpu_enabled:
            command.append("--opencl-device-types=1")  # Enable GPU
        else:
            command.append("--opencl-device-types=2")  # CPU only

        # Workload profile
        if workload_profile.isdigit() and 1 <= int(workload_profile) <= 4:
            command.extend(["-w", workload_profile])
        else:
            messagebox.showerror("Error", "Workload profile must be between 1 and 4.")
            return

        # Optimized kernel or normal kernel
        if optimized_kernel:
            command.append("--optimized-kernel-enable")

        # Temperature abortion threshold (only if enabled)
        if temp_abort_enabled:
            command.extend(["--gpu-temp-abort", str(temp_abort_value)])

        # Output estimated time
        file_size = os.path.getsize(wordlist)
        estimated_time = self.estimate_time(file_size)
        self.output_text.insert(tk.END, f"Estimated Time: {estimated_time} seconds\n")

        # Execute the command
        try:
            self.output_text.insert(tk.END, "Running Hashcat...\n")
            self.output_text.see(tk.END)

            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            for line in process.stdout:
                self.output_text.insert(tk.END, line)
                self.output_text.see(tk.END)
            process.stdout.close()
            process.wait()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to run Hashcat: {str(e)}")
            return

    def estimate_time(self, file_size):
        """Estimate time based on file size."""
        # This is just an arbitrary calculation for demo purposes.
        # A better estimate would depend on the specific hardware and hash type.
        estimated_time = file_size / 1e6  # Simulating a time estimate based on file size
        return round(estimated_time, 2)


if __name__ == "__main__":
    app = HashcatGUI()
    app.mainloop()
