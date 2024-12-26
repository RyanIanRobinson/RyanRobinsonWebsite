import os
import time
from PIL import Image, ImageTk
import tkinter as tk

script_dir = os.path.dirname(os.path.abspath(__file__))
photos_dir = os.path.join(script_dir, "photos")

while True:
    for photo_filename in os.listdir(photos_dir):
        # Create full path to the photo
        photo_path = os.path.join(photos_dir, photo_filename)

        # Skip non-image files
        if not (photo_path.lower().endswith((".jpg", ".jpeg", ".png"))):
            continue

        # Process each image
        print(f"Processing photo: {photo_filename}...")

        try:
            # Open the image using Pillow
            image = Image.open(photo_path)
            
            # Set up Tkinter window
            root = tk.Tk()
            root.title("Image Viewer")

            # Convert the image to a format Tkinter can display
            photo = ImageTk.PhotoImage(image)

            # Create a label widget to display the image
            label = tk.Label(root, image=photo)
            label.pack()

            # Ensure the image is not garbage collected by keeping a reference
            label.image = photo

            # Function to close the window after 15 seconds
            def close_window():
                root.quit()
                root.destroy()

            # Wait for 5 seconds before closing the window
            root.after(5000, close_window)

            # Start the Tkinter main event loop
            root.mainloop()

        except Exception as e:
            print(f"Error processing {photo_filename}: {e}")
