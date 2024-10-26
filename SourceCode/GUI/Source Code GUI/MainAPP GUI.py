import tkinter as tk
from tkinter import font as tkFont
import subprocess
import sys
import os

# Create the root window with a default title bar
root = tk.Tk()
root.title("Multipurpose AI Password Analysis Tool")


# Make the window resizable and set minimum size
root.geometry("800x600")
root.minsize(800, 600)  # Prevent the window from being too small

# Set a canvas for the background (with black background)
background_canvas = tk.Canvas(root, bg="black")
background_canvas.pack(fill="both", expand=True)

# Create widgets function
def create_widgets():
    # Create a custom font for the heading
    heading_font = tkFont.Font(family="Helvetica", size=30, weight="bold")

    # Create a frame for the heading and position it at the top
    heading_frame = tk.Frame(root, bg="black")
    heading_frame.pack(pady=20, side="top", fill="x")  # Positioned at the top

    # Heading label
    heading = tk.Label(heading_frame, text="Multipurpose AI Password Analysis Tool", 
                       font=heading_font, bg="black", fg="white")
    heading.pack()

    # Create a frame for the button area (with black background)
    button_frame = tk.Frame(root, bg="black")
    button_frame.pack(expand=True, pady=20)  # Use expand to center it vertically

    # Create buttons with command bindings
    dictionary_button = tk.Button(button_frame, text="Dictionary Attack", 
                                  font=tkFont.Font(size=20), width=20, height=2, 
                                  bg="light gray", fg="black",
                                  command=open_dictionary_attack)
    dictionary_button.grid(row=0, column=0, padx=20)

    analysis_button = tk.Button(button_frame, text="Password Analysis", 
                                font=tkFont.Font(size=20), width=20, height=2, 
                                bg="light gray", fg="black",
                                command=open_password_analysis)
    analysis_button.grid(row=0, column=1, padx=20)

# Navigation functions
def open_dictionary_attack():
    root.destroy()  # Close the current window
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, "DICT ATTACK GUI_2.py")
    subprocess.run([sys.executable, script_path])

def open_password_analysis():
    root.destroy()  # Close the current window
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, "Pass Analysis.py")
    subprocess.run([sys.executable, script_path])

# Create widgets after the root window is initialized
create_widgets()

# Start the Tkinter event loop
root.mainloop()
