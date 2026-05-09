import subprocess
import time
import os
import sys
from tqdm import tqdm
from sentence_transformers import CrossEncoder

# -----------------------------
# INSTALLERS
# -----------------------------
def install_faiss():
    """Installs FAISS Library."""
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'faiss-cpu'], 
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)

def install_system_dependencies():
    """Installs zstd and other necessary system tools."""
    subprocess.run(['sudo', 'apt-get', 'update'], stdout=subprocess.DEVNULL, check=True)
    subprocess.run(['sudo', 'apt-get', 'install', 'zstd', '-y'], stdout=subprocess.DEVNULL, check=True)

def install_ollama_libraries():
    """Installs Ollama CLI and Python client library."""
    # Install CLI
    subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=True, stdout=subprocess.DEVNULL)
    # Install Client
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'ollama'], 
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
def install_rouge():
    """Installs ROUGE scoring library."""
    subprocess.run(
        [sys.executable, '-m', 'pip', 'install', 'rouge-score'],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=True
    )

# -----------------------------
# SERVER & MODEL MGMT
# -----------------------------
def start_ollama_server(wait_time=10):
    """Starts the Ollama server in the background and verifies availability."""
    # Start server as a background process
    subprocess.Popen(['ollama', 'serve'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(wait_time) 
    print(f"[INFO] Server: Ollama | Status: Started | Port: 11434")
# -----------------------------
# MAIN SETUP PIPELINE
# -----------------------------
def run_system_setup():
    """Runs the full environment initialization."""
    
    steps = [
        ("System Dependencies (zstd)", install_system_dependencies),
        ("Ollama CLI & SDK", install_ollama_libraries),
        ("FAISS Engine", install_faiss),
        ("ROUGE Scorer", install_rouge),   # 👈 add this line
        ("Ollama Background Service", lambda: start_ollama_server(wait_time=10))
        ]

    print(f"[INFO] Environment: {os.name} | Action: Initializing Setup")

    for desc, func in tqdm(steps, desc="Setup Progress", ncols=100):
        try:
            func()
        except Exception as e:
            print(f"\n[ERROR] Step: {desc} | Status: Failed | Error: {e}")
            continue

    print(f"[SUCCESS] Setup: Environment Ready | All dependencies provisioned")

if __name__ == "__main__":
    run_system_setup()