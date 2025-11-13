import warnings
from urllib3.exceptions import NotOpenSSLWarning
import sys, os

# Ensure project root is on path for module imports
PROJECT_ROOT = os.path.abspath(os.getcwd())
if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)

# Silence urllib3 NotOpenSSLWarning in CI/dev environments where LibreSSL is used
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

