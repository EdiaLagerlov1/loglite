"""
scripts/check_setup.py

Run this after M1 setup to verify the environment is ready.
Usage: python scripts/check_setup.py
"""
import sys
import os


def check_imports():
    required = [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("datasets", "datasets"),
        ("sklearn", "scikit-learn"),
        ("anthropic", "anthropic"),
        ("dotenv", "python-dotenv"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
    ]
    print("Checking imports...")
    all_ok = True
    for module, package in required:
        try:
            __import__(module)
            print(f"  OK  {package}")
        except ImportError:
            print(f"  MISSING  {package}  →  pip install {package}")
            all_ok = False
    return all_ok


def check_torch_gpu():
    import torch
    has_gpu = torch.cuda.is_available()
    print(f"\nGPU available: {has_gpu}")
    if has_gpu:
        print(f"  Device: {torch.cuda.get_device_name(0)}")
    else:
        print("  Running on CPU — training will be slow. Use Colab T4 for M4.")
    return True   # not a hard failure on local machine


def check_env():
    from dotenv import load_dotenv
    load_dotenv()
    print("\nEnvironment variables:")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    print(f"  ANTHROPIC_API_KEY: {'set' if anthropic_key else 'NOT SET (needed for live agent mode)'}")
    return True


def check_config():
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from config.settings import RANDOM_SEED, MODEL_NAME, MODEL_REVISION, WINDOW_SIZE
    print(f"\nConfig loaded:")
    print(f"  RANDOM_SEED    = {RANDOM_SEED}")
    print(f"  MODEL_NAME     = {MODEL_NAME}")
    print(f"  MODEL_REVISION = {MODEL_REVISION[:12]}...")
    print(f"  WINDOW_SIZE    = {WINDOW_SIZE}")
    return True


if __name__ == "__main__":
    ok = True
    ok &= check_imports()
    check_torch_gpu()
    check_env()
    check_config()

    print("\n" + ("=" * 40))
    if ok:
        print("M1 environment check PASSED")
    else:
        print("M1 environment check FAILED — fix missing packages above")
        sys.exit(1)
