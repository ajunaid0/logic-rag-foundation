import subprocess
import time

def install_system_dependencies():
    """Installs zstd system dependency."""
    #print("Installing zstd...")
    subprocess.run(['sudo', 'apt-get', 'update'], check=True)
    subprocess.run(['sudo', 'apt-get', 'install', 'zstd', '-y'], check=True)
    print("zstd installation complete.")

def install_ollama_libraries():
    """Installs Ollama CLI and Python client library."""
    #print("Installing Ollama CLI...")
    # Using shell=True for the curl command to mimic '!curl | sh' in a script context
    subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=True)
    #print("Installing Ollama Python client library...")
    subprocess.run(['pip', 'install', 'ollama'], check=True)
    print("Ollama installation complete.")

def start_ollama_server(wait_time=5):
    """Starts the Ollama server in the background."""
    #print("Starting Ollama server...")
    # Detach the process to run in background if desired, but Popen alone often suffices for simple backgrounding.
    # Using preexec_fn=os.setsid can fully detach it if needed for long-running services.
    subprocess.Popen(['ollama', 'serve'])
    time.sleep(wait_time) # Wait for the server to initialize
    print("Ollama server started.")

def pull_ollama_model(model_name):
    """Pulls the specified Ollama embedding model."""
    import ollama
    #print(f"Pulling {model_name} model...")
    try:
        ollama.pull(model_name)
        print(f"\n{model_name} model pulled successfully!")
    except Exception as e:
        print(f"Error pulling model {model_name}: {e}")

def ollama_utils():
    install_system_dependencies()
    install_ollama_libraries()
    start_ollama_server(wait_time=10) # Give it extra time in Colab
