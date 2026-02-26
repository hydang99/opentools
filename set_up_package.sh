set -Eeuo pipefail

# phase 1: normal resolve for your big set
pip install -r requirements_mac.txt

# phase 2: add the conflicting package without deps
pip install "browser-use==0.5.0"