import tkinter as tk
from tkinter import filedialog, messagebox
import os 
import cv2 
import threading 
import datetime as date
from llama_cpp import Llama




class GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ASL GUI")
        self.camera_on = False
        self.video_dir = None
        self.video_count = 1
        self.setup_ui()

    def setup_ui(self):
        self.camera_btn = tk.Button(self.root, text="Open Camera", command=self.toggle_camera)
        self.camera_btn.pack()
        
        self.dir_btn = tk.Button(self.root, text="Select Video Directory", command=self.select_directory)
        self.dir_btn.pack()
        
        self.clean_dir = tk.Button(self.root, text="Clean Directory", command=self.clean_directory)
        self.clean_dir.pack()
        
        self.process_btn = tk.Button(self.root, text="Run Model on Videos", command=self.run_model_on_directory)
        self.process_btn.pack()

        self.status_label = tk.Label(self.root, text="Status: Idle")
        self.status_label.pack()
        self.result_text = tk.Text(self.root, height=10)
        self.result_text.pack()
        
        self.clear_btn = tk.Button(self.root,text="Clear Console", command=lambda: self.result_text.delete('1.0', tk.END))
        self.clear_btn.pack()


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
            threading.Thread(target=self.camera_loop, daemon=True).start()
        else:
            self.camera_on = False
            self.camera_btn.config(text="Open Camera")

    def camera_loop(self):
        cap = cv2.VideoCapture(0)
        recording = False
        out = None
        output_dir = "recordings/val/"
        os.makedirs(output_dir, exist_ok=True)
        datetime = date.datetime
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30 
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        while self.camera_on:
            ret, frame = cap.read()
            key = cv2.waitKey(1) & 0xFF
            if ret:
                cv2.imshow('Camera', frame)
                
                if key == ord("r") and not recording:
                    output_filename = os.path.join(output_dir, f"{self.video_count}-We.mp4") # This is where we change the word for recording!
                    out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
                    recording = True
                    self.video_count += 1
                    print(f"Recording started: {output_filename}")
                    
                elif key == ord('s') and recording:
                    recording = False
                    out.release()
                    print("Recording Stopped")
                    
                if recording:
                    out.write(frame)
                
                if key == ord('q'):
                    break
        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()

    def select_directory(self):
        self.video_dir = filedialog.askdirectory()
        self.status_label.config(text=f"Selected dir: {self.video_dir}")

    def run_model_on_directory(self):
        if not self.video_dir:
            messagebox.showerror("Error", "Please select a video directory first.")
            return
        self.status_label.config(text="Processing...")
        threading.Thread(target=self.process_videos, daemon=True).start()

    def process_videos(self):
        results = []
        input_list = []
        for fname in os.listdir(self.video_dir):
            if fname.endswith('.mp4') or fname.endswith('.avi'):
                video_path = os.path.join(self.video_dir, fname)
                res = self.run_model(video_path) 
                results.append(f"{fname}: {res}")
                self.result_text.insert(tk.END, f"{fname}: {res}\n")
                self.result_text.see(tk.END)
                self.result_text.update_idletasks()
                input_list.append(res) ##creating a str list to pass to llm 
        
        # This is where we will call the llm to combine ouputs into sentence
        llm_input = " ".join(input_list)
        # MODEL_PATH = "../../src/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
        MODEL_PATH = "../../src/models/phi-2.Q4_K_M.gguf"
        
        llm = Llama(model_path=MODEL_PATH)
        prompt = f"Make a correct and fluent English sentence using these words: {llm_input}"
        # prompt = f"Make a correct and fluent English sentence using these words: I go School Yesterday"
        response = llm(prompt, max_tokens=40, stop=["\n"])
        llm_output = response["choices"][0]["text"].strip()
        
        self.result_text.insert(tk.END, "\nCombined/Corrected Sentence:\n")
        self.result_text.insert(tk.END, f"{llm_output}")
        self.status_label.config(text="Done.")
        

    def run_model(self, video_path):
        # this should call the class with the trained model and then return a str or something
        # Placeholder implementation to avoid indentation error; replace with real model invocation.
        
        
        return "Model not implemented"

if __name__ == "__main__":
    root = tk.Tk()
    app = GUI(root)
    root.mainloop()