#!/usr/bin/env sh
set -eu

REPO="AarambhDevHub/manas"
BIN_NAME="manas"
ASSET_NAME="manas-linux-x86_64.tar.gz"
INSTALL_DIR="${INSTALL_DIR:-/usr/local/bin}"

echo "Installing Manas..."

OS="$(uname -s)"
ARCH="$(uname -m)"

if [ "$OS" != "Linux" ]; then
  echo "Error: this install script currently supports Linux only."
  echo
  echo "For macOS or Windows, please build from source:"
  echo "  git clone https://github.com/${REPO}.git"
  echo "  cd manas"
  echo "  cargo build --workspace --release"
  exit 1
fi

case "$ARCH" in
  x86_64|amd64)
    ;;
  *)
    echo "Error: this install script currently supports Linux x86_64 only."
    echo "Detected architecture: $ARCH"
    exit 1
    ;;
esac

if ! command -v curl >/dev/null 2>&1; then
  echo "Error: curl is required."
  exit 1
fi

if ! command -v tar >/dev/null 2>&1; then
  echo "Error: tar is required."
  exit 1
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

URL="https://github.com/${REPO}/releases/latest/download/${ASSET_NAME}"

echo "Downloading latest Linux release..."
curl -fsSL "$URL" -o "$TMP_DIR/$ASSET_NAME"

echo "Extracting..."
tar -xzf "$TMP_DIR/$ASSET_NAME" -C "$TMP_DIR"

chmod +x "$TMP_DIR/manas-linux-x86_64"

echo "Installing to $INSTALL_DIR/$BIN_NAME..."

if [ -w "$INSTALL_DIR" ]; then
  mv "$TMP_DIR/manas-linux-x86_64" "$INSTALL_DIR/$BIN_NAME"
else
  sudo mv "$TMP_DIR/manas-linux-x86_64" "$INSTALL_DIR/$BIN_NAME"
fi

echo "Manas installed successfully."
echo
"$INSTALL_DIR/$BIN_NAME" --help
