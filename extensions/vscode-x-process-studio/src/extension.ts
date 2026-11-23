import * as vscode from 'vscode';

const PROVIDER_ID = 'quipu.x-process-studio';
const LABEL = 'x_ProcessStudio (QUIPU)';
const TOKEN_KEY = 'xProcessStudio.token';
const SERVER_VERSION = '0.1.0';

/** Plain http is acceptable only for this machine or a private network. */
export function isSafeServerUrl(raw: string): boolean {
  let u: URL;
  try {
    u = new URL(raw);
  } catch {
    return false;
  }
  if (u.protocol === 'https:') {
    return true;
  }
  if (u.protocol !== 'http:') {
    return false;
  }
  const h = u.hostname.replace(/^\[|\]$/g, '').toLowerCase();
  if (h === 'localhost' || h === '127.0.0.1' || h === '::1' || !h.includes('.')) {
    return true;
  }
  const m = /^(\d+)\.(\d+)\.(\d+)\.(\d+)$/.exec(h);
  if (!m) {
    return false;
  }
  const [a, b] = [Number(m[1]), Number(m[2])];
  return a === 10 || a === 127 || (a === 172 && b >= 16 && b <= 31) || (a === 192 && b === 168);
}

export function activate(context: vscode.ExtensionContext): void {
  const changed = new vscode.EventEmitter<void>();
  context.subscriptions.push(changed);

  const provider: vscode.McpServerDefinitionProvider<vscode.McpHttpServerDefinition> = {
    onDidChangeMcpServerDefinitions: changed.event,

    provideMcpServerDefinitions: () => {
      const cfg = vscode.workspace.getConfiguration('xProcessStudio');
      if (!cfg.get<boolean>('enabled', true)) {
        return [];
      }
      const url = cfg.get<string>('serverUrl', 'http://127.0.0.1:7110/mcp');
      if (!isSafeServerUrl(url)) {
        void vscode.window.showWarningMessage(
          `x_ProcessStudio: refusing ${url} — use https:// for servers off this machine or private network.`);
        return [];
      }
      // No token here: providing must not prompt. The token is attached in resolve.
      return [new vscode.McpHttpServerDefinition(LABEL, vscode.Uri.parse(url), {}, SERVER_VERSION)];
    },

    resolveMcpServerDefinition: async (server) => {
      let token = await context.secrets.get(TOKEN_KEY);
      if (!token) {
        token = await vscode.window.showInputBox({
          title: 'x_ProcessStudio server token',
          prompt: 'The QUIPU_MCP_TOKEN configured on the quipu-mcp server (stored in VS Code secret storage).',
          password: true,
          ignoreFocusOut: true,
        });
        if (!token) {
          return undefined; // do not start without a token: the server fails closed anyway
        }
        await context.secrets.store(TOKEN_KEY, token);
      }
      server.headers = { ...server.headers, Authorization: `Bearer ${token}` };
      return server;
    },
  };

  context.subscriptions.push(
    vscode.lm.registerMcpServerDefinitionProvider(PROVIDER_ID, provider),
    vscode.commands.registerCommand('xProcessStudio.setToken', async () => {
      const token = await vscode.window.showInputBox({
        title: 'x_ProcessStudio server token', password: true, ignoreFocusOut: true,
      });
      if (token) {
        await context.secrets.store(TOKEN_KEY, token);
        changed.fire();
      }
    }),
    vscode.commands.registerCommand('xProcessStudio.clearToken', async () => {
      await context.secrets.delete(TOKEN_KEY);
      changed.fire();
    }),
    vscode.workspace.onDidChangeConfiguration((e) => {
      if (e.affectsConfiguration('xProcessStudio')) {
        changed.fire();
      }
    }),
  );
}

export function deactivate(): void {
  // disposables are released through context.subscriptions
}
