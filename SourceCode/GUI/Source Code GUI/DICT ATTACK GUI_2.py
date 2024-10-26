import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import os
import sys

class HashcatGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Multi-purpose AI Password Analysis Tool - Dictionary Attack")
        self.geometry("900x600")
        self.configure(bg="black")

        # Add back button
        self.back_button = self.create_button(self, "Back", self.go_back)
        self.back_button.place(relx=1.0, rely=0, x=-10, y=10, anchor="ne")

        # Create main frame to hold all other widgets
        main_frame = tk.Frame(self, bg="black")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=40)  # Add top padding for the back button

        # Define hash types
        self.hash_types = {
            "MD5": 0,
            "SHA1": 100,
            "SHA256": 1400,
            "SHA512": 1700,
            "NTLM": 1000
        }

        # Create left and right frames inside the main frame
        left_frame = tk.Frame(main_frame, bg="black")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right_frame = tk.Frame(main_frame, bg="black")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Left frame elements
        self.create_left_frame(left_frame)

        # Right frame elements
        self.create_right_frame(right_frame)

    def create_left_frame(self, frame):
        # Hash File Selection
        self.hash_file_label = tk.Label(frame, text="Select Hash File:", fg="light cyan", bg="black", font=("Arial", 12))
        self.hash_file_label.pack(pady=5)
        self.hash_file_entry = tk.Entry(frame, width=40, bg="black", fg="light cyan", font=("Arial", 12))
        self.hash_file_entry.pack(pady=5)
        self.browse_hash_button = self.create_button(frame, "Browse", self.browse_hash_file)
        self.browse_hash_button.pack(pady=5)

        # Wordlist (Dictionary) Selection
        self.wordlist_label = tk.Label(frame, text="Select Wordlist (Dictionary) File:", fg="light cyan", bg="black", font=("Arial", 12))
        self.wordlist_label.pack(pady=5)
        self.wordlist_entry = tk.Entry(frame, width=40, bg="black", fg="light cyan", font=("Arial", 12))
        self.wordlist_entry.pack(pady=5)
        self.browse_wordlist_button = self.create_button(frame, "Browse", self.browse_wordlist)
        self.browse_wordlist_button.pack(pady=5)

        # Hash Type Selection (Dropdown)
        self.hash_type_label = tk.Label(frame, text="Select Hash Type:", fg="light cyan", bg="black", font=("Arial", 12))
        self.hash_type_label.pack(pady=5)
        self.hash_type_var = tk.StringVar(self)
        self.hash_type_var.set("MD5")  # Default value
        self.hash_type_menu = tk.OptionMenu(frame, self.hash_type_var, *self.hash_types.keys())
        self.hash_type_menu.config(bg="black", fg="light cyan", highlightbackground="black", font=("Arial", 12))
        self.hash_type_menu.pack(pady=5)

        # Spacer frame to push the output to the bottom
        spacer = tk.Frame(frame, bg="black")
        spacer.pack(expand=True, fill=tk.BOTH)

        # Output
        self.output_label = tk.Label(frame, text="Hashcat Output", fg="light cyan", bg="black", font=("Arial", 12))
        self.output_label.pack(pady=5, side=tk.BOTTOM)
        self.output_text = tk.Text(frame, height=10, width=50, bg="black", fg="light cyan", font=("Arial", 12))
        self.output_text.pack(pady=5, side=tk.BOTTOM)

    def create_right_frame(self, frame):
        # GPU Selection
        self.gpu_label = tk.Label(frame, text="Use GPU:", fg="light cyan", bg="black", font=("Arial", 12))
        self.gpu_label.pack(pady=5)
        self.gpu_var = tk.BooleanVar(value=False)
        self.gpu_checkbox = tk.Checkbutton(frame, text="Enable GPU", variable=self.gpu_var, 
                                           bg="black", fg="light cyan", selectcolor="black", 
                                           activebackground="black", activeforeground="light cyan",
                                           font=("Arial", 12))
        self.gpu_checkbox.pack(pady=5)

        # Workload Profile
        self.workload_label = tk.Label(frame, text="Workload Profile (1 to 4):", fg="light cyan", bg="black", font=("Arial", 12))
        self.workload_label.pack(pady=5)
        self.workload_entry = tk.Entry(frame, width=5, bg="black", fg="light cyan", font=("Arial", 12))
        self.workload_entry.pack(pady=5)

        # Temperature Monitoring Option
        self.temp_option_label = tk.Label(frame, text="Use Temperature Abortion Threshold:", fg="light cyan", bg="black", font=("Arial", 12))
        self.temp_option_label.pack(pady=5)
        self.temp_option_var = tk.BooleanVar(value=False)
        self.temp_option_checkbox = tk.Checkbutton(frame, text="Enable", variable=self.temp_option_var, 
                                                   bg="black", fg="light cyan", selectcolor="black", 
                                                   activebackground="black", activeforeground="light cyan",
                                                   font=("Arial", 12))
        self.temp_option_checkbox.pack(pady=5)

        # Temperature Abortion Threshold Slider
        self.temp_label = tk.Label(frame, text="Temperature Abortion Threshold (°C):", fg="light cyan", bg="black", font=("Arial", 12))
        self.temp_label.pack(pady=5)
        self.temp_slider = tk.Scale(frame, from_=70, to=250, orient=tk.HORIZONTAL, bg="black", fg="light cyan")
        self.temp_slider.pack(pady=5)

        # Kernel Execution
        self.kernel_label = tk.Label(frame, text="Optimized Kernel:", fg="light cyan", bg="black", font=("Arial", 12))
        self.kernel_label.pack(pady=5)
        self.kernel_var = tk.BooleanVar(value=False)
        self.kernel_checkbox = tk.Checkbutton(frame, text="Use Optimized Kernel", variable=self.kernel_var, 
                                              bg="black", fg="light cyan", selectcolor="black", 
                                              activebackground="black", activeforeground="light cyan",
                                              font=("Arial", 12))
        self.kernel_checkbox.pack(pady=5)

        # Spacer frame to push the Run Hashcat button to the bottom
        spacer = tk.Frame(frame, bg="black")
        spacer.pack(expand=True, fill=tk.BOTH)

        # Run Hashcat Button
        self.run_button = self.create_button(frame, "Run Hashcat", self.run_hashcat)
        self.run_button.pack(pady=20, side=tk.BOTTOM)

    def create_button(self, parent, text, command):
        button = tk.Button(parent, text=text, command=command, bg="#4CAF50", fg="white", font=("Arial", 12, "bold"), bd=3)
        button.bind("<Enter>", lambda e: button.config(bg="#45a049"))
        button.bind("<Leave>", lambda e: button.config(bg="#4CAF50"))
        return button

    def browse_hash_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Hash Files", "*.*")])
        self.hash_file_entry.delete(0, tk.END)
        self.hash_file_entry.insert(0, file_path)

    def browse_wordlist(self):
        file_path = filedialog.askopenfilename(filetypes=[("Wordlist Files", "*.*")])
        self.wordlist_entry.delete(0, tk.END)
        self.wordlist_entry.insert(0, file_path)

    def run_hashcat(self):
        hash_file = self.hash_file_entry.get()
        wordlist = self.wordlist_entry.get()
        selected_hash_type = self.hash_types[self.hash_type_var.get()]
        gpu_enabled = self.gpu_var.get()
        workload_profile = self.workload_entry.get()
        temp_abort_enabled = self.temp_option_var.get()
        temp_abort_value = self.temp_slider.get()
        optimized_kernel = self.kernel_var.get()

        if not os.path.isfile(hash_file):
            messagebox.showerror("Error", "Invalid hash file.")
            return
        if not os.path.isfile(wordlist):
            messagebox.showerror("Error", "Invalid wordlist file.")
            return

        command = ["hashcat", "-m", str(selected_hash_type), hash_file, wordlist]

        if gpu_enabled:
            command.append("--opencl-device-types=1")
        else:
            command.append("--opencl-device-types=2")

        if workload_profile.isdigit() and 1 <= int(workload_profile) <= 4:
            command.extend(["-w", workload_profile])
        else:
            messagebox.showerror("Error", "Workload profile must be between 1 and 4.")
            return

        if optimized_kernel:
            command.append("--optimized-kernel-enable")

        if temp_abort_enabled:
            command.extend(["--gpu-temp-abort", str(temp_abort_value)])

        file_size = os.path.getsize(wordlist)
        self.output_text.insert(tk.END, f"Estimated time: {file_size // 1024} seconds\n")

        try:
            result = subprocess.run(command, capture_output=True, text=True)
            self.output_text.insert(tk.END, result.stdout)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to execute Hashcat: {e}")

    def go_back(self):
        self.destroy()
        # Add the parent directory to sys.path
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.append(parent_dir)
        
        # Import and run the MainAPP GUI
        from MainAPP_GUI import MainAppGUI
        main_app = MainAppGUI()
        main_app.mainloop()

if __name__ == "__main__":
    app = HashcatGUI()
    app.mainloop()
else:
    # This allows the MainAppGUI to create an instance of HashcatGUI
    def run_hashcat_gui():
        app = HashcatGUI()
        app.mainloop()
