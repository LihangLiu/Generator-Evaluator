"""
sys path related
"""

import sys
from os.path import dirname, join, realpath

sys.path.insert(0, realpath(join(dirname(__file__), '../..')))

