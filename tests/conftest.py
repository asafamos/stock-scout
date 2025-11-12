import warnings
from urllib3.exceptions import NotOpenSSLWarning

# Silence urllib3 NotOpenSSLWarning in CI/dev environments where LibreSSL is used
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

