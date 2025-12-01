import tkinter as tk
from tkinter import filedialog, messagebox
import os 
import cv2 
import threading 
import datetime as date
from llama_cpp import Llama
import time
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
import os
import sys
import pandas as pd

#our imports
from .modelCode import Model
from ...utils.preprocess import initialize_FrameModelOutput_Data, showLandmarksFrame
from ...utils.preprocess import getLandmarkOutput, append_FrameModelOutput_Data




class GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ASL GUI")
        self.camera_on = False
        self.video_dir = None
        self.video_count = 1
        self.llm_words = []
        self.landmark = False
        self.setup_ui()
        self.output_dir = "./turnIn/src/frontend/recordings" 
        
        self.model = Model("./turnIn/src/models/christian/ASL_Model_.ckpt")
    
    def setup_ui(self):
        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=10)
        
        video_frame = tk.Frame(self.root)
        video_frame.pack(pady=10)
        
        action_frame = tk.Frame(self.root)
        action_frame.pack(pady=10)
        
        # Camera controls in top row
        self.camera_btn = tk.Button(control_frame, text="Open Camera", command=self.toggle_camera, width=15)
        self.camera_btn.grid(row=0, column=0, padx=5)
        
        self.landmark_btn = tk.Button(control_frame, text="Start Landmarks", command=self.toggle_landmark, width=15)
        self.landmark_btn.grid(row=0, column=1, padx=5)
        
        # Video directory controls in middle row
        self.dir_btn = tk.Button(video_frame, text="Select Video Directory", command=self.select_directory, width=20)
        self.dir_btn.grid(row=0, column=0, padx=5)
        
        self.clean_dir = tk.Button(video_frame, text="Clean Directory", command=self.clean_directory, width=15)
        self.clean_dir.grid(row=0, column=1, padx=5)
        
        # Processing controls in bottom row
        self.process_btn = tk.Button(action_frame, text="Run Model on Videos", command=self.run_model_on_directory, width=20)
        self.process_btn.grid(row=0, column=0, padx=5)
        
        self.llama_btn = tk.Button(action_frame, text="Show Corrected Sentence", command=lambda: self.run_llm(self.llm_words), width=20)
        self.llama_btn.grid(row=0, column=1, padx=5)
        
        self.clear_btn = tk.Button(action_frame, text="Clear Console", command=lambda: self.result_text.delete('1.0', tk.END), width=15)
        self.clear_btn.grid(row=0, column=2, padx=5)
    
        # Status and results
        self.status_label = tk.Label(self.root, text="Status: Idle", font=('Arial', 10))
        self.status_label.pack(pady=5)
        
        # Results text box with scrollbar
        text_frame = tk.Frame(self.root)
        text_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.result_text = tk.Text(text_frame, height=15, width=60, yscrollcommand=scrollbar.set)
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar.config(command=self.result_text.yview)

    def clean_directory(self):
        for fname in os.listdir(self.video_dir):
            if fname.endswith('.mp4'):
                file_path = os.path.join(self.video_dir, fname)
                os.remove(file_path)
        self.video_count = 1


    def toggle_camera(self):
        if not self.camera_on:
            self.camera_on = True
            self.camera_btn.config(text="Close Camera")
            self.llm_words = []
            threading.Thread(target=self.camera_loop, daemon=True).start()
            
        else:
            self.camera_on = False
            self.camera_btn.config(text="Open Camera")

    def toggle_landmark(self):
        if not self.camera_on:
            if not self.landmark:
                self.landmark = True
                self.landmark_btn.config(text="Stop Landmarks")
            else:
                self.landmark = False
                self.landmark_btn.config(text="Start Landmarks")

    
    def camera_loop(self):
        cap = None
        out = None
        
        try:
            cap = cv2.VideoCapture(0)
            recording = False
            output_dir = self.output_dir
            os.makedirs(output_dir, exist_ok=True)
            datetime = date.datetime
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30 
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

            data:dict = initialize_FrameModelOutput_Data()  
            with mp_holistic.Holistic(min_detection_confidence=0.50, min_tracking_confidence=0.5) as holistic:    
                while self.camera_on:
                    ret, frame = cap.read()
                    key = cv2.waitKey(1) & 0xFF
                    if ret:
                        cv2.imshow('Camera', frame)



                        if recording:
                            modelOutput, image = getLandmarkOutput(frame, holistic) 
                            append_FrameModelOutput_Data(modelOutput, data)
                            if self.landmark:
                                showLandmarksFrame(image, modelOutput)

                        if key == ord("r") and not recording:
                            output_filename = os.path.join(output_dir, f"{self.video_count}.mp4") # This is where we change the word for recording!
                            out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
                            recording = True
                            self.video_count += 1
                            print(f"Recording started: {output_filename}")  

                        elif key == ord('s') and recording:
                            recording = False
                            out.release()
                            print("Recording Stopped")

                            prediction, confidence = self.model.predictDict(data)
                            confidence = float(confidence)
                            self.llm_words.append(prediction)
                            self.result_text.insert(tk.END, f"{prediction}, {confidence*100}%\n")
                            self.result_text.see(tk.END)
                            data = initialize_FrameModelOutput_Data()


                        if recording:
                            out.write(frame)

                        if key == ord('q'):
                            break
                        
                        if not self.camera_on:
                            break
        finally:
            time.sleep(0.1)
            if cap is not None:
                cap.release()
            if out is not None:
                out.release()
                
            time.sleep(0.1)    
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        
            self.camera_on = False
            self.camera_btn.config(text="Open Camera")
        
        
    def select_directory(self):
        self.video_dir = filedialog.askdirectory()
        self.status_label.config(text=f"Selected dir: {self.video_dir}")

    def run_model_on_directory(self):
        if not self.video_dir:
            messagebox.showerror("Error", "Please select a video directory first.")
            return
        if not self.camera_on:
            self.status_label.config(text="Processing...")
            threading.Thread(target=self.process_videos, daemon=True).start()
        else:
            messagebox.showerror("Error", "Please close camera first")

    def process_videos(self):
        input_list = []
        video_files = []
        for fname in os.listdir(self.video_dir):
            if fname.endswith('.mp4') or fname.endswith('.avi'):
                video_path = os.path.join(self.video_dir, fname)
                video_files.append(video_path)    
         
        predictions, confidence = self.model.predictFiles(video_files)
        
        
        for fname, res, confidence in zip(video_files, predictions, confidence):
            if fname.endswith('.mp4') or fname.endswith('.avi'):
                fname = os.path.basename(fname)
                self.result_text.insert(tk.END, f"{fname}: {res}, {int(confidence*100)}%\n")
                self.result_text.see(tk.END)
                self.result_text.update_idletasks()
                input_list.append(res) ##creating a str list to pass to llm 
                
        self.run_llm(input_list)
        
        
    
    def run_llm(self, input_list):
        # This is where we will call the llm to combine ouputs into sentence
        llm_input = ", ".join(input_list)
        # MODEL_PATH = "../../src/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
        MODEL_PATH = "./turnIn/src/frontend/llm_model/phi-2.Q4_K_M.gguf"
        # MODEL_PATH = os.path.join(PROJECT_ROOT, "llm_model", "phi-2.Q4_K_M.gguf")
        
        llm = Llama(model_path=MODEL_PATH)
        prompt = f"Given these words: {llm_input} Rearange them to make a correct english sentence."
        response = llm(prompt, max_tokens=1000, stop=["\n"])
        llm_output = response["choices"][0]["text"].strip()
        
        self.result_text.insert(tk.END, "\nCombined/Corrected Sentence:\n")
        self.result_text.insert(tk.END, f"{llm_output}\n")
        self.status_label.config(text="Done.")
        



if __name__ == "__main__":
    root = tk.Tk()
    app = GUI(root)
    root.mainloop()