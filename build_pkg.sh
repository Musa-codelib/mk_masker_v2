#!/bin/bash
# build_pkg.sh — Build the Mk Masker Pro .pkg installer
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="$SCRIPT_DIR/app"
PKG_DIR="$SCRIPT_DIR/PKG_Install"
DIST_DIR="$SCRIPT_DIR/dist"
BUILD_DIR="$SCRIPT_DIR/build"

echo "=========================================="
echo "Building Mk Masker Pro .pkg Installer"
echo "=========================================="

# Clean previous builds
rm -rf "$DIST_DIR" "$BUILD_DIR"
mkdir -p "$DIST_DIR" "$BUILD_DIR"

# Step 1: Build Electron app
echo ""
echo "Step 1: Building Electron app..."
cd "$APP_DIR"
npm run build

APP_BUNDLE="$APP_DIR/dist/mac-arm64/Mk Masker Pro.app"
if [ ! -d "$APP_BUNDLE" ]; then
    echo "ERROR: Electron app build failed — $APP_BUNDLE not found"
    exit 1
fi
echo "✓ Electron app built: $APP_BUNDLE"

# Step 2: Prepare PKG payload
echo ""
echo "Step 2: Preparing PKG payload..."
PAYLOAD_DIR="$BUILD_DIR/payload"
rm -rf "$PAYLOAD_DIR"
mkdir -p "$PAYLOAD_DIR/Library/Application Support/com.mkmasker.pro"

# Copy Electron .app
cp -R "$APP_BUNDLE" "$PAYLOAD_DIR/Library/Application Support/com.mkmasker.pro/Mk Masker Pro.app"
echo "✓ App bundle copied"

# Copy server files
cp -R "$SCRIPT_DIR/server" "$PAYLOAD_DIR/Library/Application Support/com.mkmasker.pro/server"
echo "✓ Server files copied"

# Copy ffmpeg
mkdir -p "$PAYLOAD_DIR/Library/Application Support/com.mkmasker.pro/bin"
cp "$SCRIPT_DIR/bin/ffmpeg" "$PAYLOAD_DIR/Library/Application Support/com.mkmasker.pro/bin/ffmpeg"
chmod +x "$PAYLOAD_DIR/Library/Application Support/com.mkmasker.pro/bin/ffmpeg"
echo "✓ FFmpeg copied"

# Copy requirements.txt
cp "$PKG_DIR/Library/Application Support/com.mkmasker.pro/requirements.txt" "$PAYLOAD_DIR/Library/Application Support/com.mkmasker.pro/requirements.txt"
echo "✓ Requirements copied"

# Copy installer scripts
INSTALL_SCRIPTS_DIR="$BUILD_DIR/installer_scripts"
mkdir -p "$INSTALL_SCRIPTS_DIR"
cp "$PKG_DIR/installer_scripts/preinstall" "$INSTALL_SCRIPTS_DIR/preinstall"
cp "$PKG_DIR/installer_scripts/postinstall" "$INSTALL_SCRIPTS_DIR/postinstall"
cp "$PKG_DIR/installer_scripts/postinstall.zsh" "$INSTALL_SCRIPTS_DIR/postinstall.zsh"
chmod +x "$INSTALL_SCRIPTS_DIR/preinstall" "$INSTALL_SCRIPTS_DIR/postinstall" "$INSTALL_SCRIPTS_DIR/postinstall.zsh"
echo "✓ Installer scripts copied"

# Step 3: Build component pkg
echo ""
echo "Step 3: Building component pkg..."
COMPONENT_PKG="$BUILD_DIR/mk-masker-pro.pkg"
pkgbuild --root "$PAYLOAD_DIR" \
    --identifier "com.mkmasker.pro" \
    --version "1.0.0" \
    --scripts "$INSTALL_SCRIPTS_DIR" \
    --install-location "/" \
    "$COMPONENT_PKG"
echo "✓ Component pkg built: $COMPONENT_PKG"

# Step 4: Create distribution pkg
echo ""
echo "Step 4: Creating distribution pkg..."
DIST_PKG="$DIST_DIR/Mk_Masker_Pro_1.0.0.pkg"

productbuild --package "$COMPONENT_PKG" \
    --resources "$BUILD_DIR/resources" \
    "$DIST_PKG"
echo "✓ Distribution pkg built: $DIST_PKG"

# Step 5: Cleanup
rm -rf "$BUILD_DIR"

echo ""
echo "=========================================="
echo "Build complete!"
echo "Output: $DIST_PKG"
echo "=========================================="
