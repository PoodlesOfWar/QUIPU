export const version: string;
export class IntegrationError extends Error {}
export interface Tool { name: string; description?: string; inputSchema?: unknown }
export class Client {
  constructor(url: string, token: string, opts?: { fetchImpl?: typeof fetch });
  readonly url: string;
  connect(): Promise<this>;
  tools(): Promise<Tool[]>;
  call(name: string, args?: Record<string, unknown>): Promise<unknown>;
  status(): Promise<unknown>;
  list(status?: string): Promise<unknown>;
  hubActions(): Promise<unknown>;
  /** Stage a proposal. It is held; a human authorises it elsewhere. */
  propose(tool: string, reason: string, args?: Record<string, unknown>): Promise<unknown>;
  voice(): Promise<unknown>;
}
