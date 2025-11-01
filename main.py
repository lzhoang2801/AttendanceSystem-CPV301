import customtkinter as ctk
from PIL import Image
import cv2
import os
import threading
from face_registration import face_registration
from face_detection import face_detection
from train_model import train_recognizer
from face_recognition import recognize_faces, model_path, map_path
import pandas as pd
import numpy as np

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.capture = cv2.VideoCapture(0)

        self.title("Attendance System")
        self.geometry("1000x600")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        if not self.capture.isOpened():
            self.show_message("Camera not found.\nPlease check your camera connection and try again.", on_close=self.on_closing)
            return

        self.capture.set(3, 640)
        self.capture.set(4, 480)

        self.recognizer = None
        self.identity_map = None
        self.last_logged = {}
        self.mode = "Attendance"

        self.registration_info = None
        self.saved_face_count = 0
        self.target_face_count = 20
        self.face_capture_frame_count = 0
        self.face_capture_wait_frames = 5

        self.main_frame = ctk.CTkFrame(self, corner_radius=0)
        self.main_frame.pack(fill="both", expand=True)

        self.left_panel = ctk.CTkFrame(self.main_frame, width=250, corner_radius=0)
        self.left_panel.pack(side="left", fill="y")
        
        self.title_label = ctk.CTkLabel(self.left_panel, text="Controls", font=ctk.CTkFont(size=20, weight="bold"))
        self.title_label.pack(pady=20)

        self.register_button = ctk.CTkButton(self.left_panel, text="Register New User", command=self.start_registration)
        self.register_button.pack(pady=10, padx=20, fill="x")

        self.right_panel = ctk.CTkFrame(self.main_frame)
        self.right_panel.pack(side="right", fill="both", expand=True)

        self.video_label = ctk.CTkLabel(self.right_panel, text="No camera found")
        self.video_label.pack(pady=10, padx=10, expand=True, fill="both")

        self.log_label = ctk.CTkLabel(self.left_panel, text="Attendance Status", font=ctk.CTkFont(size=20, weight="bold"))
        self.log_textbox = ctk.CTkTextbox(self.left_panel)

        self.on_start()

    def switch_mode(self, mode):
        self.mode = mode
        if self.mode == "Attendance":
            self.log_label.pack(pady=20)
            self.log_textbox.pack(pady=10, padx=10, fill="both", expand=True)
        else:
            self.log_label.pack_forget()
            self.log_textbox.pack_forget()

        if self.mode == "Training" or self.mode == "Registration":
            self.register_button.configure(state="disabled")
        else:
            self.register_button.configure(state="normal")

    def start_registration(self):
        input_dialog = ctk.CTkToplevel(self)
        input_dialog.title("Input User Information")
        input_dialog.resizable(False, False)

        label_username = ctk.CTkLabel(input_dialog, text="Enter name:")
        label_username.pack(pady=(20, 5), padx=20)
        entry_username = ctk.CTkEntry(input_dialog)
        entry_username.pack(padx=20, fill="x")

        label_userid = ctk.CTkLabel(input_dialog, text="Enter user ID:")
        label_userid.pack(pady=(20, 5), padx=20)
        entry_userid = ctk.CTkEntry(input_dialog)
        entry_userid.pack(padx=20, fill="x")

        button_frame = ctk.CTkFrame(input_dialog, fg_color="transparent")
        button_frame.pack(pady=(20, 10))

        def on_confirm():
            username = entry_username.get().replace(' ', '').lower().strip()
            user_id = entry_userid.get().replace(' ', '').lower().strip()
            if username and user_id:
                self.registration_info = {"name": username, "id": user_id}
                self.saved_face_count = 0
                self.face_capture_frame_count = 0
                self.switch_mode("Registration")
                input_dialog.destroy()

        def on_cancel():
            input_dialog.destroy()

        confirm_btn = ctk.CTkButton(button_frame, text="Confirm", command=on_confirm, width=120)
        confirm_btn.pack(side="left", padx=10)
        cancel_btn = ctk.CTkButton(button_frame, text="Cancel", command=on_cancel, width=120)
        cancel_btn.pack(side="right", padx=10)

        input_dialog.protocol("WM_DELETE_WINDOW", on_cancel)

        self.update_idletasks()
        input_dialog.update()
        req_height = label_username.winfo_reqheight() + label_userid.winfo_reqheight() + button_frame.winfo_reqheight() + 40
        
        input_dialog.geometry(f"400x{req_height}")

        input_dialog.transient(self)
        input_dialog.grab_set()

    def start_training(self):
        self.switch_mode("Training")
        threading.Thread(target=self.run_training, daemon=True).start()

    def run_training(self):
        trained = train_recognizer()
        self.after(0, self.on_training_complete, trained)
    
    def on_training_complete(self, trained):
        self.load_model()
        if trained:
            self.show_message("Model has been trained and reloaded successfully!")
        else:
            self.show_message("Model training failed. Please try again.")
        self.switch_mode("Attendance")

    def prompt_registration(self):
        msg_box = ctk.CTkToplevel(self)
        msg_box.title("Registration Required")

        label = ctk.CTkLabel(msg_box, text="Model not found.\nPlease register a user to continue.", wraplength=380)
        label.pack(pady=20, padx=20, expand=True, fill="both")

        button_frame = ctk.CTkFrame(msg_box, fg_color="transparent")
        button_frame.pack(pady=(0, 20))

        def register_handler():
            msg_box.destroy()
            self.start_registration()

        def cancel_handler():
            msg_box.destroy()
            self.on_closing()

        register_button = ctk.CTkButton(button_frame, text="Register", command=register_handler, width=120)
        register_button.pack(side="left", padx=10)

        cancel_button = ctk.CTkButton(button_frame, text="Cancel", command=cancel_handler, width=120)
        cancel_button.pack(side="right", padx=10)

        msg_box.protocol("WM_DELETE_WINDOW", cancel_handler)
        
        msg_box.transient(self)
        msg_box.grab_set()

    def load_model(self):
        if not os.path.exists(model_path) or not os.path.exists(map_path):
            self.prompt_registration()
            return False

        try:
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            self.recognizer.read(model_path)
            self.identity_map = pd.read_csv(map_path)
            return True
        except Exception as e:
            self.show_message("Model failed to load.\nReload application to try again or notify the administrator.", on_close=self.on_closing)
        
        return False

    def on_start(self):
        self.load_model()
        self.switch_mode(self.mode)
        self.video_loop()
        self.update_attendance_log()

    def video_loop(self):
        ret, frame = self.capture.read()
        if ret:
            frame_processed = frame.copy()
            
            if self.mode == "Attendance":
                detected_faces = face_detection(frame_processed)
                frame_processed, self.last_logged, logged_new = recognize_faces(frame_processed, detected_faces, self.recognizer, self.identity_map, self.last_logged)
                if logged_new:
                    self.update_attendance_log()
            
            elif self.mode == "Registration":
                detected_faces = face_detection(frame_processed)
                if self.face_capture_frame_count % self.face_capture_wait_frames == 0:
                    if len(detected_faces) > 0:
                        face_registration(self.registration_info["name"], self.registration_info["id"], frame, detected_faces)
                        self.saved_face_count += 1

                for (x, y, w, h, _) in detected_faces:
                    cv2.rectangle(frame_processed, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                progress_text = f'Saved {self.saved_face_count}/{self.target_face_count} images'
                cv2.putText(frame_processed, progress_text, (10, 30), cv2.FONT_ITALIC, 0.7, (0, 0, 255), 2)
                self.face_capture_frame_count += 1
                
                if self.saved_face_count >= self.target_face_count:
                    self.start_training()

            elif self.mode == "Training":
                frame_processed = np.zeros((480, 640, 3), dtype=np.uint8)
                text = "Training model, please wait..."
                text_size = cv2.getTextSize(text, cv2.FONT_ITALIC, 1, 2)[0]
                text_x = (frame_processed.shape[1] - text_size[0]) // 2
                text_y = (frame_processed.shape[0] + text_size[1]) // 2
                cv2.putText(frame_processed, text, (text_x, text_y), cv2.FONT_ITALIC, 1, (255, 255, 255), 2)

            frame_rgb = cv2.cvtColor(frame_processed, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ctk.CTkImage(light_image=img, dark_image=img, size=(img.width, img.height))
            self.video_label.configure(image=imgtk, text="")

        self.after(10, self.video_loop)

    def update_attendance_log(self):
        try:
            self.log_textbox.delete('1.0', 'end')
            if len(self.last_logged) > 0:
                content = "\n".join([f"{index}. {name}" for index, (name, _) in enumerate(sorted(self.last_logged.items(), key=lambda x: (x[1][0], x[1][1])), start=1)])
            else:
                content = "No attendance records yet."
            self.log_textbox.insert('end', content + "\n")
            self.log_textbox.see("end")
        except Exception as e:
            print(f"Error updating attendance log: {e}")

    def show_message(self, message, on_close=None):
        msg_box = ctk.CTkToplevel(self)
        msg_box.title("Information")
        
        label = ctk.CTkLabel(msg_box, text=message, wraplength=380)
        label.pack(pady=20, padx=20, expand=True, fill="both")

        def close_handler():
            msg_box.destroy()
            if on_close:
                on_close()

        button = ctk.CTkButton(msg_box, text="OK", command=close_handler)
        button.pack(pady=10, padx=20)
        
        msg_box.protocol("WM_DELETE_WINDOW", close_handler)

        self.update_idletasks()
        msg_box.update()
        req_height = label.winfo_reqheight() + button.winfo_reqheight() + 40
        
        msg_box.geometry(f"400x{req_height}")
        
        msg_box.transient(self)
        msg_box.grab_set()

    def on_closing(self):
        self.capture.release()
        self.destroy()

if __name__ == "__main__":
    app = App()
    app.mainloop()