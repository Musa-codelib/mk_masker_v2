// Electron main process: owns the native window and loads the UI (index.html).
// All app logic lives in the renderer (renderer.js); this file just boots the window.
const { app, BrowserWindow } = require('electron');
const path = require('path');

function createWindow() {
  const win = new BrowserWindow({
    width: 1280,
    height: 850,
    backgroundColor: '#1a1a1a',
    titleBarStyle: 'hiddenInset', // Native macOS traffic-light buttons + hidden title bar
    webPreferences: {
      // NOTE: nodeIntegration/contextIsolation are relaxed so the renderer can use
      // require() and talk to the Python socket server directly. Fine for a local app.
      nodeIntegration: true,
      contextIsolation: false,
      sandbox: false
    }
  });

  // The UI is a plain static page rendered from the app/ folder.
  win.loadFile('index.html');
}

// Create the window once Electron is ready.
app.whenReady().then(createWindow);

// On non-macOS platforms, quit when all windows close. macOS keeps the app alive.
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});
