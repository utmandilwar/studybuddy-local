"""Environment and hardware diagnostic script for StudyBuddy Local.

Verifies:
1. Python runtime and platform
2. Essential local dependencies
3. Local-only network configuration (zero cloud endpoints)
4. Ollama daemon connectivity and local model availability
5. Storage directories and permissions
"""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

# Add project root to sys.path so config and src can be imported directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import AppConfig
from src.privacy import assert_local_configuration, get_privacy_badge


def check_python_environment() -> bool:
    print("\n--- 1. Python Environment ---")
    version = sys.version.split()[0]
    major, minor = sys.version_info.major, sys.version_info.minor
    print(f"  Python executable: {sys.executable}")
    print(f"  Python version:    {version} ({platform.architecture()[0]})")
    print(f"  Platform:          {platform.platform()}")

    if major == 3 and minor >= 9:
        print("  [OK] Python version meets requirements (3.9+).")
        return True
    else:
        print("  [WARNING] Python 3.9+ recommended for best compatibility.")
        return False


def check_dependencies() -> bool:
    print("\n--- 2. Package Dependencies ---")
    required = [
        ("streamlit", "Streamlit UI"),
        ("fitz", "PyMuPDF (fitz)"),
        ("ollama", "Ollama Python client"),
        ("chromadb", "ChromaDB local vector store"),
        ("sentence_transformers", "Sentence Transformers"),
        ("pydantic", "Pydantic validation"),
        ("pytest", "Pytest framework"),
    ]

    all_passed = True
    for module_name, display_name in required:
        try:
            mod = __import__(module_name)
            ver = getattr(mod, "__version__", "installed")
            print(f"  [OK] {display_name:<30} ({ver})")
        except ImportError:
            print(f"  [MISSING] {display_name:<30} -> run `pip install -r requirements.txt`")
            all_passed = False

    return all_passed


def check_privacy_configuration() -> bool:
    print("\n--- 3. Privacy & Network Configuration ---")
    try:
        assert_local_configuration()
        badge, verified = get_privacy_badge()
        print(f"  [OK] Privacy verification: {badge}")
        print(f"  [OK] Ollama binding:       {AppConfig.OLLAMA_BASE_URL} (Localhost only)")
        print(f"  [OK] Telemetry status:     Disabled (No analytics or cloud reporting)")
        return True
    except Exception as e:
        print(f"  [ERROR] Privacy check failed: {e}")
        return False


def check_storage_paths() -> bool:
    print("\n--- 4. Local Storage Paths ---")
    data_dir = AppConfig.DATA_DIR
    chroma_dir = AppConfig.CHROMA_PATH

    print(f"  Data directory:   {data_dir}")
    print(f"  Chroma directory: {chroma_dir}")

    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        chroma_dir.mkdir(parents=True, exist_ok=True)
        test_file = data_dir / ".test_write"
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink()
        print("  [OK] Local storage directories are writable.")
        return True
    except Exception as e:
        print(f"  [ERROR] Cannot write to local storage directory: {e}")
        return False


def check_ollama_status() -> bool:
    print("\n--- 5. Ollama Local LLM Service ---")
    from src.llm import LocalLLM

    llm = LocalLLM()
    connected, message, models = llm.check_connection()

    if connected:
        print(f"  [OK] {message}")
        print(f"  Available local models: {models if models else 'None downloaded yet'}")
        if llm.is_model_available():
            print(f"  [OK] Default target model '{AppConfig.DEFAULT_MODEL}' is ready for inference!")
            return True
        else:
            print(f"  [ACTION REQUIRED] Run `ollama pull {AppConfig.DEFAULT_MODEL}` to download the model.")
            return False
    else:
        print(f"  [NOT RUNNING] {message}")
        print("  Action steps:")
        print("    1. Download & install Ollama from https://ollama.com if not installed.")
        print("    2. Start Ollama: run `ollama serve` in a terminal.")
        print(f"    3. Download the model: run `ollama pull {AppConfig.DEFAULT_MODEL}`.")
        return False


def main():
    print("=" * 60)
    print(f" {AppConfig.APP_NAME} v{AppConfig.APP_VERSION} - Environment Check")
    print(f" {AppConfig.TAGLINE}")
    print("=" * 60)

    py_ok = check_python_environment()
    deps_ok = check_dependencies()
    priv_ok = check_privacy_configuration()
    store_ok = check_storage_paths()
    ollama_ok = check_ollama_status()

    print("\n" + "=" * 60)
    print(" DIAGNOSTIC SUMMARY")
    print("=" * 60)
    print(f"  Python Environment:     {'PASSED' if py_ok else 'WARN'}")
    print(f"  Dependencies Installed: {'PASSED' if deps_ok else 'FAILED'}")
    print(f"  Privacy Verification:   {'PASSED' if priv_ok else 'FAILED'}")
    print(f"  Local Storage:          {'PASSED' if store_ok else 'FAILED'}")
    print(f"  Ollama & Model:         {'READY' if ollama_ok else 'ACTION NEEDED'}")
    print("=" * 60)

    if deps_ok and priv_ok and store_ok:
        print("\nCore application framework is configured successfully.")
        if not ollama_ok:
            print("Note: Start Ollama and download the model to enable local AI text generation.")
    else:
        print("\nPlease resolve any [MISSING] or [ERROR] items above.")


if __name__ == "__main__":
    main()
