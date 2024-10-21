import tkinter as tk
from tkinter import font as tkFont

# Create the root window
root = tk.Tk()
root.title("Multipurpose AI Password Analysis Tool")

# Set window to full screen and allow resizing
root.attributes('-fullscreen', True)

# Set a 3D mashup background (blood red and dark blue gradient)
background_canvas = tk.Canvas(root, width=root.winfo_screenwidth(), height=root.winfo_screenheight())
background_canvas.pack(fill="both", expand=True)

# Function to create a 3D gradient-like mashup background
def draw_gradient(canvas, width, height):
    r_start, g_start, b_start = 255, 0, 0   # Blood Red
    r_end, g_end, b_end = 0, 0, 139         # Dark Blue
    
    for i in range(256):
        r = int(r_start + (r_end - r_start) * i / 255)
        g = int(g_start + (g_end - g_start) * i / 255)
        b = int(b_start + (b_end - b_start) * i / 255)
        color = f'#{r:02x}{g:02x}{b:02x}'
        canvas.create_line(0, i*2, width, i*2, fill=color)

# Call the gradient function to paint the background
draw_gradient(background_canvas, root.winfo_screenwidth(), root.winfo_screenheight())

# Create a frame for the heading, and place it on top of the background canvas
heading_frame = tk.Frame(background_canvas, bg="", bd=0)
heading_frame.pack(pady=50)

# Create a custom font for the heading
heading_font = tkFont.Font(family="Helvetica", size=30, weight="bold")

# Heading label (with valid background and foreground colors)
heading = tk.Label(heading_frame, text="Multipurpose AI Password Analysis Tool", 
                   font=heading_font, bg="light gray", fg="black")
heading.pack()

# Create a frame for the buttons
button_frame = tk.Frame(background_canvas, bg="", bd=0)
button_frame.pack(pady=200)

# Create buttons (with valid background and foreground colors)
dictionary_button = tk.Button(button_frame, text="Dictionary Attack", 
                              font=tkFont.Font(size=20), width=20, height=2, 
                              bg="light gray", fg="black")
dictionary_button.grid(row=0, column=0, padx=50, pady=20)

analysis_button = tk.Button(button_frame, text="Password Analysis", 
                            font=tkFont.Font(size=20), width=20, height=2, 
                            bg="light gray", fg="black")
analysis_button.grid(row=0, column=1, padx=50, pady=20)

# Start the Tkinter event loop
root.mainloop()
