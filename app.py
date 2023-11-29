import tkinter as tk
from tkinter import ttk, font
import cv2
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models, datasets
import sqlite3 as sql
import time

transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

students_detected = set()
is_popup_open = False

model = models.resnet18(pretrained=False, progress=True)
path_to_data = r'C:\School\csci 4353\studentData'
dataset = datasets.ImageFolder(root=path_to_data, transform=transform)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(dataset.classes))

model.load_state_dict(torch.load(r'C:\School\csci 4353\FaceCheck\facecheck2.pth'))

model.eval()

last_processed_time = 0
processing_interval = 2

def show_frame():
    global students_detected, last_processed_time
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)

    current_time = time.time()
    if current_time - last_processed_time > processing_interval:
        process_frame(cv2image)
        last_processed_time = current_time
    lmain.after(10, show_frame)

def process_frame(cv2image):
    pil_image = Image.fromarray(cv2image)
    preprocessed_img = transform(pil_image).unsqueeze(0)

    with torch.no_grad():
        predictions = model(preprocessed_img)

    _, predicted_class_idx = predictions.max(1)
    predicted_class = dataset.classes[predicted_class_idx.item()]

    if predicted_class:
        print(predicted_class)
        
        if predicted_class not in students_detected:
            show_popup(predicted_class)
            students_detected.add(predicted_class)
    
def show_popup(label):
    # Create a top-level window
    global is_popup_open

    if is_popup_open:
        return
    
    is_popup_open = True
    student_info = get_student_info(label)
    popup = tk.Toplevel()
    popup.title("Prediction")
    ttk.Label(popup, text=f"Name: {student_info[0]}\nSID: {student_info[1]}\nemail: {student_info[2]}").pack(fill='x', padx=50, pady=10)
    ttk.Button(popup, text="OK", command=lambda: close_popup(popup)).pack(fill='x')

def close_popup(popup):
    global is_popup_open
    popup.destroy()
    is_popup_open = False

def get_student_info(sid):
    conn = sql.connect('students.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM students WHERE sid = ?", (sid,))
    student_info = cursor.fetchone()
    conn.close()
    return student_info

def use_model():
    global cap
    cap = cv2.VideoCapture(0)
    btn_stop_model.pack(side="right", padx=10, pady=10)  # Use pack for the stop button
    show_frame()

def stop_model():
    global cap, students_detected, is_popup_open
    if cap.isOpened():
        cap.release()
    lmain.imgtk = None
    lmain.configure(image='')
    btn_stop_model.pack_forget()  # Hide the stop button
    if is_popup_open:
        is_popup_open = False
    if len(students_detected) != 0:
        info_window = tk.Toplevel(root)
        info_window.title("Recognized Students")

        custom_font = font.Font(size=18)

        info_text_widget = tk.Text(info_window, font=custom_font)
        info_text_widget.pack(fill='both', expand=True)
        i = 1
        for sid in students_detected:
            student_info = get_student_info(sid)
            if student_info:
                info = f" Student {i}:\nName: {student_info[0]}\nSID: {student_info[1]}\nEmail: {student_info[2]}\n\n"
                info_text_widget.insert('end', info)
            i += 1

        # Button to close the info window
        ttk.Button(info_window, text="Close", command=info_window.destroy).pack()

        # Clear the recognized_students set for the next use
        students_detected.clear()

# Create main window
root = tk.Tk()
root.title("Attendance App")
root.geometry("800x700")

# Top frame for the title
top_frame = ttk.Frame(root, padding="3 3 12 12")
top_frame.pack(fill=tk.X)
ttk.Label(top_frame, text="AI Attendance Checker", font=("Helvetica", 16)).pack(side="left", padx=10)

# Frame for video feed
main_frame = ttk.Frame(root, padding="3 3 12 12")
main_frame.pack(fill=tk.BOTH, expand=True)

# Label for video feed
lmain = ttk.Label(main_frame)
lmain.pack(padx=10, pady=10)

# Button frame for controls
button_frame = ttk.Frame(root, padding="3 3 12 12")
button_frame.pack(fill=tk.X)

# Buttons
btn_use_model = ttk.Button(button_frame, text="Use Model", command=use_model)
btn_use_model.pack(side="left", padx=10, pady=10)
btn_stop_model = ttk.Button(button_frame, text="Stop Model", command=stop_model)
# Initially hide the stop button
btn_stop_model.pack_forget()

# Status bar at the bottom
statusbar = ttk.Label(root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
statusbar.pack(fill=tk.X)

# Start the GUI
root.mainloop()

# Release the VideoCapture object
if cap.isOpened():
    cap.release()
