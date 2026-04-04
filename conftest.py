"""
conftest.py  (project root)
Ensures the project root is on sys.path so that
  from env.environment import ...
  from tasks.task1 import ...
work without installing the package.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
