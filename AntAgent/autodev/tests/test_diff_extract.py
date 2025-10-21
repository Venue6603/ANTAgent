import unittest
from AntAgent.autodev.manager import _extract_unified_diff

class TestDiffExtract(unittest.TestCase):
    def test_extract_unified_diff_removes_markdown_fences(self):
        text = """```diff
diff --git a/file1.py b/file1.py
index 83db48f..f735c3b 100644
--- a/file1.py
+++ b/file1.py
@@ -1,3 +1,3 @@
 print("Hello, World!")
```"""
        expected = """diff --git a/file1.py b/file1.py
index 83db48f..f735c3b 100644
--- a/file1.py
++ b/file1.py
@@ -1,3 +1,3 @@
 print("Hello, World!")"""
        self.assertEqual(_extract_unified_diff(text), expected)
