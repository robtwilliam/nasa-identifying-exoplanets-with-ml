import os, subprocess, webbrowser

print("Starting Exoplanet Classifier API...\n")
print("\n🌐  Visit http://127.0.0.1:8000/form in your browser once application has loaded.")

# venv_path = "venv" if os.path.isdir("venv") else ".venv" if os.path.isdir(".venv") else None
venv_path = "../ml_env"
model_dir = "model_api"

if not os.path.isdir(model_dir):
    print(f"⚠️  Warning: '{model_dir}' not found — ensure your model files are exported.\n")

# --- Build uvicorn command relative to subfolder ---
cmd = ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
cwd = os.path.join(os.getcwd(), "model_api")

# --- Activate environment + run ---
if venv_path:
    print(f"🔹 Activating virtual environment: {venv_path}")
    activate = os.path.join(venv_path, "Scripts" if os.name == "nt" else "bin", "activate")
    if os.name == "nt":
        subprocess.run(f'cmd /k "{activate} && cd model_api && uvicorn app:app --host 0.0.0.0 --port 8000 --reload"', shell=True)
    else:
        subprocess.run(f'bash -c "source {activate} && cd model_api && uvicorn app:app --host 0.0.0.0 --port 8000 --reload"', shell=True)
else:
    print("⚠️  No virtual environment found. Using system Python.")
    subprocess.Popen(cmd, cwd=cwd)