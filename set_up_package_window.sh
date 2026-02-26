set -Eeuo pipefail

# phase 1: normal resolve for your big set
pip install -r requirements_window.txt

python -m pip install paddlepaddle -i https://www.paddlepaddle.org.cn/packages/stable/cpu/


# phase 2: add the conflicting package without deps
pip install "browser-use==0.5.0"


