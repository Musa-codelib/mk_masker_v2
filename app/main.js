const { app, BrowserWindow } = require('electron');
const path = require('path');

function createWindow() {
  const win = new BrowserWindow({
    width: 1280,
    height: 850,
    backgroundColor: '#1a1a1a',
    titleBarStyle: 'hiddenInset', // Modern Mac look
    webPreferences: {
    nodeIntegration: true,
    contextIsolation: false,
    sandbox: false
    }
  });

  win.loadFile('index.html');
}

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});