"""pytest configuration.

Puts the repo root on sys.path so the test files under tests/ can
`import main`, `import http_server`, etc. — the engine modules live at
the repo root, not in a package.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
