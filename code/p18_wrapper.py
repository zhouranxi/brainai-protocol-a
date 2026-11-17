
import json, os, subprocess, sys, time
PY = r"C:/Python313/python.exe"
SCRIPT = r"C:/Users/Lance/WorkBuddy/2026-10-25-21-15-17/github-repo/code/p18_hier_prompt.py"
LOG = r"C:/Users/Lance/WorkBuddy/2026-10-25-21-15-17/github-repo/logs_p18.log"
for attempt in range(100):
    with open(LOG, "a", encoding="utf-8") as lf:
        lf.write(f"[wrapper] attempt {attempt} {time.ctime()}\n"); lf.flush()
    p = subprocess.Popen([PY, SCRIPT], stdout=open(LOG, "ab"), stderr=subprocess.STDOUT,
                         cwd=r"C:/Users/Lance/WorkBuddy/2026-10-25-21-15-17/github-repo",
                         creationflags=0x00000008)
    p.wait()
    with open(LOG, "a", encoding="utf-8") as lf:
        lf.write(f"[wrapper] exited rc={p.returncode} {time.ctime()}\n"); lf.flush()
    time.sleep(10)
