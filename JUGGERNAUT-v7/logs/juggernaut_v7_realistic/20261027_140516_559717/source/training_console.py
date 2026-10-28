"""Run the V7 pipeline with live Command Prompt output and a UTF-8 console log."""
import os
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['PYTHONIOENCODING'] = 'utf-8'
with (root / 'training_console.log').open('a', encoding='utf-8', buffering=1) as log:
    process = subprocess.Popen([sys.executable, '-u', str(root / 'pipeline.py'), *sys.argv[1:]], cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', bufsize=1)
    try:
        for line in process.stdout:
            print(line, end='', flush=True)
            log.write(line)
        code = process.wait()
    except KeyboardInterrupt:
        try:
            code = process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.terminate()
            code = process.wait()
    print(f'\nTraining pipeline exited with code {code}.', flush=True)
    log.write(f'\nTraining pipeline exited with code {code}.\n')
sys.exit(code)
