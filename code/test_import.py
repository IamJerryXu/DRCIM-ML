
import sys
import os

# Add the current directory to path
sys.path.append(os.getcwd())

try:
    from gma_core import alignment
    from gma_core import encoder
    print("Imports successful")
except Exception as e:
    print(f"Import failed: {e}")
