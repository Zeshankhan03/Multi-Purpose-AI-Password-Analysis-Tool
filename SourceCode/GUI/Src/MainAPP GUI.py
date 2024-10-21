import tkinter as tk
from tkinter import font as tkFont

# Create the root window
root = tk.Tk()
root.title("Multipurpose AI Password Analysis Tool")

# Set window to full screen
root.attributes('-fullscreen', True)

# Set a canvas for the background
background_canvas = tk.Canvas(root)
background_canvas.pack(fill="both", expand=True)

# Function to create a 3D gradient-like mashup background
def draw_gradient(canvas):
    width = canvas.winfo_reqwidth()
    height = canvas.winfo_reqheight()
    
    r_start, g_start, b_start = 255, 0, 0   # Blood Red
    r_end, g_end, b_end = 0, 0, 139         # Dark Blue
    
    for i in range(height):
        r = int(r_start + (r_end - r_start) * i / height)
        g = int(g_start + (g_end - g_start) * i / height)
        b = int(b_start + (b_end - b_start) * i / height)
        color = f'#{r:02x}{g:02x}{b:02x}'
        canvas.create_line(0, i, width, i, fill=color)

# Call the gradient function to paint the background after the window is fully created
def create_widgets():
    # Create a custom font for the heading
    heading_font = tkFont.Font(family="Helvetica", size=30, weight="bold")

    # Create a frame for the heading and position it
    heading_frame = tk.Frame(root, bg="light gray", bd=0)  # Use valid background color
    background_canvas.create_window(root.winfo_screenwidth() // 2, 100, window=heading_frame)

    # Heading label
    heading = tk.Label(heading_frame, text="Multipurpose AI Password Analysis Tool", 
                       font=heading_font, bg="light gray", fg="black")  # Use valid background color
    heading.pack()

    # Create a frame for the buttons and position it
    button_frame = tk.Frame(root, bg="light gray", bd=0)  # Use valid background color
    background_canvas.create_window(root.winfo_screenwidth() // 2, root.winfo_screenheight() // 2, window=button_frame)

    # Create buttons
    dictionary_button = tk.Button(button_frame, text="Dictionary Attack", 
                                  font=tkFont.Font(size=20), width=20, height=2, 
                                  bg="light gray", fg="black")
    dictionary_button.grid(row=0, column=0, padx=50, pady=20)

    analysis_button = tk.Button(button_frame, text="Password Analysis", 
                                font=tkFont.Font(size=20), width=20, height=2, 
                                bg="light gray", fg="black")
    analysis_button.grid(row=0, column=1, padx=50, pady=20)

    # Call the gradient function after widgets are created
    draw_gradient(background_canvas)

# Create widgets after the root window is initialized
create_widgets()

# Start the Tkinter event loop
root.mainloop()
