import psutil, os

my_pid = os.getpid()
for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
    try:
        cmd = " ".join(proc.info['cmdline'] or [])
        if 'run_1h_champion_challenger' in cmd:
            print(f"Terminating lingering process {proc.info['pid']}: {cmd[:80]}")
            proc.kill()
    except Exception:
        pass
print("Cleanup complete.")
