import sys, threading, queue, time

q = queue.Queue()

def stdin_reader():
    """Standard in reader for controls
    """
    for line in sys.stdin:
        q.put(line.rstrip("\n"))
    q.put(None)

threading.Thread(target=stdin_reader, daemon=True).start()

running = True
paused = False
count = 0
print("Processed Backend started")
sys.stdout.flush()

while running:
    try:
        try:
            cmd = q.get(timeout=0.2)
        except queue.Empty:
            cmd = None
        if cmd is None:
            # EOF -> exit
            break
        if cmd:
            c = cmd.strip().lower()
            if c == "stop":
                print("[backend] stopping")
                sys.stdout.flush()
                running = False
                break
            elif c == "pause":
                paused = True
                print("[backend] paused")
                sys.stdout.flush()
            elif c == "resume":
                paused = False
                print("[backend] resumed")
                sys.stdout.flush()
            elif c.startswith("do "): # this will be start command
                arg = c[3:]
                print(f"[backend] doing {arg}")
                sys.stdout.flush()
                time.sleep(0.5)
                print(f"[backend] done {arg}")
                sys.stdout.flush()
            else:
                print(f"[backend] unknown: {cmd!r}")
                sys.stdout.flush()

        if not paused:
            print(f"[backend] heartbeat {count}")
            sys.stdout.flush()
            count += 1

        time.sleep(1 if not paused else 0.2)
    except KeyboardInterrupt:
        break

print("Processed Backend exited")
sys.stdout.flush()