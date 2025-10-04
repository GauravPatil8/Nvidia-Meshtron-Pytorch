set -e

echo "updating pip"
python3 -m pip install --upgrade pip

if [ -f "requirements.txt"]; then
    echo "installing from requirements.txt"
    pip install -r requirements.txt
else
    echo "requirements.txt not found"
fi

PACKAGE_LINK = "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.2/flash_attn-2.8.2+cu12torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
echo "Installing package from link: $PACKAGE_LINK"
pip install "$PACKAGE_LINK"

echo "All installations completed successfully!"
https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl