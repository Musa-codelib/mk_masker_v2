// Electron main process: owns the native window and loads the UI (index.html).
// Also manages the Python SocketIO server lifecycle.
const { app, BrowserWindow } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

let serverProcess = null;
let mainWindow = null;

function getServerScript() {
  const possiblePaths = [
    path.join(process.resourcesPath, 'server', 'server.py'),
    path.join(__dirname, 'server', 'server.py'),
  ];
  for (const p of possiblePaths) {
    if (require('fs').existsSync(p)) return p;
  }
  return path.join(__dirname, 'server', 'server.py');
}

function getPythonCmd() {
  // In production, the venv is created by the postinstall script in ~/Library/Caches/com.mkmasker.pro/venv
  const homeDir = require('os').homedir();
  const venvPython = path.join(homeDir, 'Library', 'Caches', 'com.mkmasker.pro', 'venv', 'bin', 'python');
  if (require('fs').existsSync(venvPython)) {
    return venvPython;
  }
  // Fallback to system python
  return 'python3';
}

function startServer() {
  return new Promise((resolve, reject) => {
    const serverScript = getServerScript();
    const pythonCmd = getPythonCmd();

    console.log(`Starting server: ${pythonCmd} ${serverScript}`);

    serverProcess = spawn(pythonCmd, [serverScript], {
      cwd: path.dirname(serverScript),
      env: {
        ...process.env,
        PYTHONUNBUFFERED: '1',
      },
    });

    serverProcess.stdout.on('data', (data) => {
      console.log(`[server] ${data}`);
    });

    serverProcess.stderr.on('data', (data) => {
      console.error(`[server] ${data}`);
    });

    serverProcess.on('error', (err) => {
      console.error('Failed to start server:', err);
      reject(err);
    });

    // Wait for server to be ready (it prints "Running on http://0.0.0.0:8080")
    const timeout = setTimeout(() => {
      reject(new Error('Server startup timeout'));
    }, 30000);

    const checkServer = () => {
      // Server is ready when we can connect - we'll resolve and let the renderer handle connection
      clearTimeout(timeout);
      resolve();
    };

    // Give the server a moment to start, then resolve
    setTimeout(checkServer, 2000);
  });
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 850,
    backgroundColor: '#1a1a1a',
    titleBarStyle: 'hiddenInset',
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
      sandbox: false,
    },
  });

  mainWindow.loadFile('index.html');

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

app.whenReady().then(async () => {
  try {
    await startServer();
    createWindow();
  } catch (err) {
    console.error('Failed to start app:', err);
    // Still create window so user can see error
    createWindow();
  }
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    if (serverProcess) {
      serverProcess.kill();
    }
    app.quit();
  }
});

app.on('before-quit', () => {
  if (serverProcess) {
    serverProcess.kill();
  }
});
