import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import os

class HashcatGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Multi-purpose AI Password Analysis Tool - Dictionary Attack")
        self.geometry("700x850")

        # Set theme to black with light cyan text
        self.configure(bg="black")

        # Hash File Selection
        self.hash_file_label = tk.Label(self, text="Select Hash File:", fg="light cyan", bg="black", font=("Arial", 12))
        self.hash_file_label.pack(pady=5)
        self.hash_file_entry = tk.Entry(self, width=40, bg="black", fg="light cyan", font=("Arial", 12))
        self.hash_file_entry.pack(pady=5)
        self.browse_hash_button = self.create_button("Browse", self.browse_hash_file)
        self.browse_hash_button.pack(pady=5)

        # Wordlist (Dictionary) Selection
        self.wordlist_label = tk.Label(self, text="Select Wordlist (Dictionary) File:", fg="light cyan", bg="black", font=("Arial", 12))
        self.wordlist_label.pack(pady=5)
        self.wordlist_entry = tk.Entry(self, width=40, bg="black", fg="light cyan", font=("Arial", 12))
        self.wordlist_entry.pack(pady=5)
        self.browse_wordlist_button = self.create_button("Browse", self.browse_wordlist)
        self.browse_wordlist_button.pack(pady=5)

        # Hash Type Selection (Dropdown)
        self.hash_type_label = tk.Label(self, text="Select Hash Type:", fg="light cyan", bg="black", font=("Arial", 12))
        self.hash_type_label.pack(pady=5)

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
        self.hash_type_menu.config(bg="black", fg="light cyan", highlightbackground="black", font=("Arial", 12))
        self.hash_type_menu.pack(pady=5)

        # GPU Selection
        self.gpu_label = tk.Label(self, text="Use GPU:", fg="light cyan", bg="black", font=("Arial", 12))
        self.gpu_label.pack(pady=5)
        self.gpu_var = tk.BooleanVar(value=True)  # Default to True
        self.gpu_checkbox = tk.Checkbutton(self, text="Enable GPU", variable=self.gpu_var, bg="black", fg="light cyan", selectcolor="light cyan", font=("Arial", 12))
        self.gpu_checkbox.pack(pady=5)

        # Workload Profile
        self.workload_label = tk.Label(self, text="Workload Profile (1 to 4):", fg="light cyan", bg="black", font=("Arial", 12))
        self.workload_label.pack(pady=5)
        self.workload_entry = tk.Entry(self, width=5, bg="black", fg="light cyan", font=("Arial", 12))
        self.workload_entry.pack(pady=5)

        # Temperature Monitoring Option
        self.temp_option_label = tk.Label(self, text="Use Temperature Abortion Threshold:", fg="light cyan", bg="black", font=("Arial", 12))
        self.temp_option_label.pack(pady=5)
        self.temp_option_var = tk.BooleanVar(value=False)  # Default to False
        self.temp_option_checkbox = tk.Checkbutton(self, text="Enable", variable=self.temp_option_var, bg="black", fg="light cyan", selectcolor="light cyan", font=("Arial", 12))
        self.temp_option_checkbox.pack(pady=5)

        # Temperature Abortion Threshold Slider
        self.temp_label = tk.Label(self, text="Temperature Abortion Threshold (°C):", fg="light cyan", bg="black", font=("Arial", 12))
        self.temp_label.pack(pady=5)
        self.temp_slider = tk.Scale(self, from_=70, to=250, orient=tk.HORIZONTAL, bg="black", fg="light cyan")  # Updated max to 250°C
        self.temp_slider.pack(pady=5)

        # Kernel Execution
        self.kernel_label = tk.Label(self, text="Optimized Kernel:", fg="light cyan", bg="black", font=("Arial", 12))
        self.kernel_label.pack(pady=5)
        self.kernel_var = tk.BooleanVar(value=True)  # Default to True
        self.kernel_checkbox = tk.Checkbutton(self, text="Use Optimized Kernel", variable=self.kernel_var, bg="black", fg="light cyan", selectcolor="light cyan", font=("Arial", 12))
        self.kernel_checkbox.pack(pady=5)

        # Output and Run Button
        self.output_label = tk.Label(self, text="Hashcat Output:", fg="light cyan", bg="black", font=("Arial", 12))
        self.output_label.pack(pady=5)
        # Adjusted size for the output text box
        self.output_text = tk.Text(self, height=5, width=60, bg="black", fg="light cyan", font=("Arial", 12))
        self.output_text.pack(pady=5)

        # Run Hashcat Button
        self.run_button = self.create_button("Run Hashcat", self.run_hashcat)
        self.run_button.pack(pady=20)

    def create_button(self, text, command):
        """Create a styled button with hover effects."""
        button = tk.Button(self, text=text, command=command, bg="#4CAF50", fg="white", font=("Arial", 12, "bold"), bd=3)
        button.bind("<Enter>", lambda e: button.config(bg="#45a049"))  # Hover effect
        button.bind("<Leave>", lambda e: button.config(bg="#4CAF50"))  # Reset color
        return button

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
        self.output_text.insert(tk.END, f"Estimated time: {file_size // 1024} seconds\n")

        # Execute the command and handle output
        try:
            result = subprocess.run(command, capture_output=True, text=True)
            self.output_text.insert(tk.END, result.stdout)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to execute Hashcat: {e}")

if __name__ == "__main__":
    app = HashcatGUI()
    app.mainloop()
