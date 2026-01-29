"""
Project: Automated File Organizer
Author: Abhishek Tiwari
Date: January 2026
Description: 
    A Python automation script that declutters a directory by sorting files 
    into sub-folders based on their extensions.
    
    Features:
    - Scans the current directory for files.
    - Automatically creates category folders (Images, Docs, Software, etc.).
    - Moves files securely using the 'shutil' library.
    - Skips itself (organizer.py) to prevent errors.

Usage:
    Place this script in the messy folder and run:
    $ python organizer.py
"""

import os
import shutil

# 1. Define the Categories (The Logic)
# Dictionary mapping: "Extension" -> "Folder Name"
DIRECTORIES = {
    "IMAGES": [".jpg", ".jpeg", ".png", ".gif", ".bmp"],
    "DOCUMENTS": [".pdf", ".docx", ".txt", ".xlsx", ".pptx"],
    "VIDEO": [".mp4", ".mkv", ".mov", ".avi"],
    "AUDIO": [".mp3", ".wav", ".flac"],
    "SOFTWARE": [".exe", ".msi", ".zip", ".rar", ".dmg"]
}

def organize_files():
    # os.getcwd() gets the "Current Working Directory" (where this script is)
    working_dir = os.getcwd() 
    
    # Loop through every file in this folder
    for filename in os.listdir(working_dir):
        
        # SKIP the script itself so we don't move our own code!
        if filename == "organizer.py":
            continue

        # Get the file extension (e.g., '.jpg')
        # os.path.splitext("photo.jpg") returns ("photo", ".jpg")
        file_path = os.path.join(working_dir, filename)
        
        # Skip if it's a folder (we only want to move files)
        if os.path.isdir(file_path):
            continue

        _, extension = os.path.splitext(filename)
        extension = extension.lower() # Make sure .JPG and .jpg are treated the same

        # Find the correct folder for this extension
        found = False
        for folder_name, extensions_list in DIRECTORIES.items():
            if extension in extensions_list:
                
                # 2. Create the folder if it doesn't exist
                target_folder = os.path.join(working_dir, folder_name)
                if not os.path.exists(target_folder):
                    os.makedirs(target_folder)
                    print(f"Created folder: {folder_name}")

                # 3. Move the file
                # shutil.move(source, destination)
                destination_path = os.path.join(target_folder, filename)
                shutil.move(file_path, destination_path)
                
                print(f"Moved {filename} -> {folder_name}")
                found = True
                break # Stop checking other categories once found
        
        if not found:
            print(f"Skipped {filename} (Unknown Type)")

# Run the function
if __name__ == "__main__":
    organize_files()
    print("Organization Complete! 🧹")