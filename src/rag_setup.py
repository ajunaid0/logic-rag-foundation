import subprocess
import time
from tqdm import tqdm

def install_faiss():
  """Installs FAISS Library."""
  subprocess.run(['pip', 'install', 'faiss-cpu'], check=True)


def install_system_dependencies():
    """Installs zstd system dependency."""
    subprocess.run(['sudo', 'apt-get', 'update'], check=True)
    subprocess.run(['sudo', 'apt-get', 'install', 'zstd', '-y'], check=True)

def install_ollama_libraries():
    """Installs Ollama CLI and Python client library."""
    # Using shell=True for the curl command to mimic '!curl | sh' in a script context
    subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=True)
    subprocess.run(['pip', 'install', 'ollama'], check=True)

def start_ollama_server(wait_time=5):
    """Starts the Ollama server in the background."""
    # Detach the process to run in background if desired, but Popen alone often suffices for simple backgrounding.
    # Using preexec_fn=os.setsid can fully detach it if needed for long-running services.
    subprocess.Popen(['ollama', 'serve'])
    time.sleep(wait_time) # Wait for the server to initialize
    

def pull_ollama_model(model_name):
    """Pulls the specified Ollama embedding model."""
    import ollama
    try:
        ollama.pull(model_name)
    except Exception as e:
        print(f"Error pulling model {model_name}: {e}")

def util_files():
    steps = [
        ("Installing zstd", install_system_dependencies),
        ("Installing Ollama", install_ollama_libraries),
        ("Starting Ollama server", lambda: start_ollama_server(wait_time=10)),
        ("Installing FAISS", install_faiss),
    ]

    print("Installing System Dependencies")

    for desc, func in tqdm(steps, desc="Setup Progress", ncols=80):
        func()
    print("System Dependencies Installed")
