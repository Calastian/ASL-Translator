#!/myenv/bin/python
"""
frontend_gui.py — tkinter GUI.
"""
from __future__ import annotations
import os
import sys
import threading
import queue
import subprocess
import shlex
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

DEFAULT_POLL_MS = 100

class ScriptRunnerGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("ASL GUI")
        root.geometry("800x520")
        
         # State
        self.scripts_dir = os.getcwd()
        self.process: subprocess.Popen | None = None
        self.proc_threads = []  # threads reading stdout/stderr
        self.output_q: "queue.Queue[tuple[str,str]]" = queue.Queue()
        self.stop_requested = False

        # Top controls frame
        top = ttk.Frame(root)
        top.pack(fill="x", padx=8, pady=6)

        ttk.Label(top, text="Directory:").pack(side="left")
        self.dir_label = ttk.Label(top, text=self.scripts_dir, relief="sunken")
        self.dir_label.pack(side="left", fill="x", expand=True, padx=(6,6))
        ttk.Button(top, text="Browse...", command=self.browse_dir).pack(side="left")

        # Middle left: file listbrowse_dir
        mid = ttk.Frame(root)
        mid.pack(fill="both", expand=True, padx=8, pady=(0,6))

        left = ttk.Frame(mid)
        left.pack(side="left", fill="both", expand=False)

        ttk.Label(left, text="Python scripts (.py)").pack(anchor="w")
        self.listbox = tk.Listbox(left, width=36, height=20)
        self.listbox.pack(side="left", fill="y", expand=False, padx=(0,6))
        self.listbox.bind("<Double-Button-1>", self.on_listbox_double)
        scrollbar = ttk.Scrollbar(left, orient="vertical", command=self.listbox.yview)
        scrollbar.pack(side="left", fill="y")
        self.listbox.config(yscrollcommand=scrollbar.set)

        # Middle right: controls and console
        right = ttk.Frame(mid)
        right.pack(side="left", fill="both", expand=True)

        # Interpreter selection and args
        interp_frame = ttk.Frame(right)
        interp_frame.pack(fill="x", pady=(0,6))
        ttk.Label(interp_frame, text="Interpreter:").pack(side="left")
        self.interp_var = tk.StringVar(value=sys.executable)
        self.interp_entry = ttk.Entry(interp_frame, textvariable=self.interp_var)
        self.interp_entry.pack(side="left", fill="x", expand=True, padx=(6,6))
        ttk.Button(interp_frame, text="Browse", command=self.browse_interpreter).pack(side="left")

        args_frame = ttk.Frame(right)
        args_frame.pack(fill="x", pady=(0,6))
        ttk.Label(args_frame, text="Arguments:").pack(side="left")
        self.args_var = tk.StringVar(value="")
        self.args_entry = ttk.Entry(args_frame, textvariable=self.args_var)
        self.args_entry.pack(side="left", fill="x", expand=True, padx=(6,6))

        buttons = ttk.Frame(right)
        buttons.pack(fill="x", pady=(0,6))
        self.run_btn = ttk.Button(buttons, text="Run", command=self.run_selected)
        self.run_btn.pack(side="left")
        self.stop_btn = ttk.Button(buttons, text="Stop", command=self.stop_process, state="disabled")
        self.stop_btn.pack(side="left", padx=(6,6))
        ttk.Button(buttons, text="Clear Log", command=self.clear_console).pack(side="left")
        ttk.Button(buttons, text="Open Folder", command=self.open_scripts_dir).pack(side="left", padx=(6,0))

        # Console area
        console_label = ttk.Label(right, text="Console output:")
        console_label.pack(anchor="w")
        console_frame = ttk.Frame(right)
        console_frame.pack(fill="both", expand=True)
        self.console = tk.Text(console_frame, wrap="none", height=20, state="disabled")
        self.console.pack(side="left", fill="both", expand=True)
        self.console.tag_configure("stderr", foreground="red")
        self.console.tag_configure("stdout", foreground="black")
        con_v = ttk.Scrollbar(console_frame, orient="vertical", command=self.console.yview)
        con_v.pack(side="right", fill="y")
        self.console.config(yscrollcommand=con_v.set)

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status = ttk.Label(root, textvariable=self.status_var, relief="sunken", anchor="w")
        status.pack(fill="x", padx=0, pady=(0,0))

        # Fill list
        self.refresh_file_list()

        # Polling queue
        self.root.after(DEFAULT_POLL_MS, self.poll_output_queue)


    def poll_output_queue(self):
        try:
            while True:
                kind, text = self.output_q.get_nowait()
                if kind == "__control__" and text == "finished":
                    # process finished cleanup
                    self.status_var.set("Ready")
                    self.run_btn.config(state="normal")
                    self.stop_btn.config(state="disabled")
                    self.process = None
                    continue
                tag = "stderr" if kind == "stderr" else "stdout"
                self.console.config(state="normal")
                self.console.insert(tk.END, text, tag)
                self.console.see(tk.END)
                self.console.config(state="disabled")
        except queue.Empty:
            pass
        finally:
            self.root.after(DEFAULT_POLL_MS, self.poll_output_queue)


    def get_selected_script(self):
        sel = self.listbox.curselection()
        if not sel:
            messagebox.showinfo("No selection", "Please select a script to run.")
            return None
        filename = self.listbox.get(sel[0])
        return os.path.join(self.scripts_dir, filename)


    def clear_console(self):
        self.console.config(state="normal")
        self.console.delete("1.0", tk.END)
        self.console.config(state="disabled")


    def browse_interpreter(self):
        p = filedialog.askopenfilename(title="select python interpreter", initialdir=os.path.dirname(sys.executable))
        if p:
            self.interp_var.set(p)


    def browse_dir(self):
        d = filedialog.askdirectory(initialdir=self.scripts_dir)
        if d:
            self.scripts_dir = d
            self.dir_label.config(text=self.scripts_dir)
            self.refresh_file_list()
    
    def open_scripts_dir(self):
        try:
            if sys.platform == "win32":
                os.startfile(self.scripts_dir)
            elif sys.platform == "darwin":
                subprocess.Popen(["xdg-open", self.scripts_dir])
        except Exception as e:
            messagebox.showerror("Open Folder", f"Could not open folder: {e}")

    def refresh_file_list(self):
        self.listbox.delete(0, tk.END)
        try:
            files = sorted([f for f in os.listdir(self.scripts_dir) if f.endswith(".py")])
            for f in files:
                self.listbox.insert(tk.END, f)
            self.status_var.set(f"{len(files)} script(s) found")
        except Exception as e:
            self.status_var.set(f"Error listing directory: {e}")
            
    def on_listbox_double(self,event):
        self.run_selected()
        
    def run_selected(self):
        if self.process:
            messagebox.showinfo("Already running", "A script is already running stop it first.")
            return
        script = self.get_selected_script()
        if not script:
            return
        interp = self.interp_var.get().strip() or sys.executable
        args_text = self.args_var.get().strip()
        args = shlex.split(args_text) if args_text else []
        cmd = [interp, script] + args
        
        # trying to force python child to be unbuffered and ensure the utf as well as the env
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        
        #possibly start a subprocess, Ayyy it worked!! 
        try:
            self.process = subprocess.Popen(
                args=cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                bufsize=1,
                universal_newlines=True,
                encoding="utf-8",
                env=env
            )
        except Exception as e:
            messagebox.showerror("Failed to run", f"Could not start process: {e}")
            self.process = None
            return

        self.status_var.set(f"Running: {os.path.basename(script)} (pid={self.process.pid})")
        self.run_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.stop_requested = False
        
        # Threading to read stdout
        thread_out = threading.Thread(target=self._reader_thread, args=(self.process.stdout, "stdout"), daemon=True)
        thread_error = threading.Thread(target=self._reader_thread, args=(self.process.stderr, "stderr"), daemon=True)
        thread_out.start(); thread_error.start()
        self.proc_threads = [thread_out, thread_error]
        
        # Threading to wait for process exit
        thread_wait = threading.Thread(target=self._wait_thread, args=(self.process,), daemon=True)
        thread_wait.start()
        
            
    
    def _reader_thread(self, pipe, kind: str):
        try:
            with pipe:
                for line in pipe: # maybe we ensure str and normalize??
                    if not isinstance(line, str):
                        try:
                            line = line.decode(errors="replace")
                        except Exception:
                            line = str(line)
                    self.output_q.put((kind, line))
        except Exception as e:
            self.output_q.put(("stderr", f"[reader error] {e} \n"))
    
    def _wait_thread(self, proc: subprocess.Popen):
        ret = None
        try: 
            ret = proc.wait()
        except Exception as e:
            self.output_q.put(("stderr", f"[wait error] {e}\n"))
            ret = -1
        
        # process may have finnished
        self.output_q.put(("stdout", f"\n[Process exited with code {ret}]\n"))
        # Update the GUIIIII
        self.output_q.put(("__control__", "finished"))
        
    def stop_process(self):
        if not self.process:
            return
        proc = self.process
        self.status_var.set(f"Stopping pid={proc.pid}...")
        self.stop_btn.config(state="disabled")
        self.stop_requested = True
        def terminator(p):
            try:
                p.terminate()
            except Exception:
                pass
            # wait a bit
            t0 = time.time()
            while True:
                if p.poll() is not None:
                    break
                if time.time() - t0 > 2.0:
                    break
                time.sleep(0.05)
            if p.poll() is None:
                try:
                    p.kill()
                except Exception:
                    pass
        t = threading.Thread(target=terminator, args=(proc,), daemon=True)
        t.start()

   
def on_close(root: tk.Tk, app: ScriptRunnerGUI):
    if app.process:
        if messagebox.askyesno("Quit", "Script still running. Stop script and quit?"):
            app.stop_process()
            
            time.sleep(0.2)
        else:
            return
    root.destroy()

def main():
    root = tk.Tk()
    app = ScriptRunnerGUI(root)
    root.protocol("WM_DELETE_WINDOW", lambda: on_close(root, app))
    root.mainloop()
    
if __name__ == "__main__":
    main()