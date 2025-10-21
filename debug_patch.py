import os
import sys
import json
from pathlib import Path

# Set the project root
os.chdir(r"C:\Users\James\PycharmProjects\ANTAgent")
sys.path.insert(0, os.getcwd())

# Now import the patch module
from AntAgent.autodev.patch import apply_unified_diff

# Create a minimal test diff
test_diff = """diff --git a/AntAgent/app.py b/AntAgent/app.py
--- a/AntAgent/app.py
+++ b/AntAgent/app.py
@@ -1,4 +1,5 @@
+# DEBUG-TEST-MARKER
 from black import diff
 from fastapi import FastAPI, UploadFile, File
 from typing import List"""

print(f"Current directory: {os.getcwd()}")
print(f"Target file exists: {os.path.exists('AntAgent/app.py')}")

# Read first line before
with open("AntAgent/app.py", "r") as f:
    first_line_before = f.readline()
print(f"First line before: {repr(first_line_before)}")

# Apply the patch
result = apply_unified_diff(test_diff)
print(f"Result: {json.dumps(result, indent=2)}")

# Read first line after
with open("AntAgent/app.py", "r") as f:
    first_line_after = f.readline()
print(f"First line after: {repr(first_line_after)}")
print(f"File modified: {first_line_before != first_line_after}")