// Loadopoly integrations SDK (ES module, no dependencies). Same surface as the Python SDK:
// status / list / hubActions / propose / voice. There is no authorise or realise: the server has no such tool.
export const version = "0.1.0";
const PROTOCOL = "2025-03-26";

export class IntegrationError extends Error {}

function privateHost(h) {
  if (!h) return false;
  if (h === "localhost" || h.endsWith(".local") || !h.includes(".")) return true;
  return /^(127\.|10\.|192\.168\.|169\.254\.|172\.(1[6-9]|2\d|3[01])\.)/.test(h);
}

function parse(body) {
  const t = body.trim();
  if (!t) return null;
  try { if (t.startsWith("{")) return JSON.parse(t); } catch { return null; }
  for (const line of t.split("\n")) {
    if (line.startsWith("data:")) { try { return JSON.parse(line.slice(5).trim()); } catch { return null; } }
  }
  return null;
}

export class Client {
  #token; #session = null; #id = 0;
  constructor(url, token, { fetchImpl } = {}) {
    const u = new URL(url);
    if (!["http:", "https:"].includes(u.protocol)) throw new IntegrationError("url must be http(s)");
    if (u.protocol === "http:" && !privateHost(u.hostname)) throw new IntegrationError("plain http to a public address is refused; use https");
    if (!token) throw new IntegrationError("a token is required");
    this.url = url; this.#token = token; this.fetch = fetchImpl ?? globalThis.fetch;
  }
  toString() { return `Client(${this.url})`; }          // never show the token
  async #rpc(method, params, notify = false) {
    const msg = { jsonrpc: "2.0", method, ...(params !== undefined ? { params } : {}) };
    if (!notify) msg.id = ++this.#id;
    const headers = { "Content-Type": "application/json", Accept: "application/json, text/event-stream", Authorization: `Bearer ${this.#token}` };
    if (this.#session) headers["Mcp-Session-Id"] = this.#session;
    let r;
    try { r = await this.fetch(this.url, { method: "POST", headers, body: JSON.stringify(msg) }); }
    catch (e) { throw new IntegrationError(`cannot reach the server: ${e.message}`); }
    if (!r.ok) throw new IntegrationError(`server answered ${r.status}`);
    this.#session = r.headers.get("Mcp-Session-Id") ?? this.#session;
    if (notify) return null;
    const out = parse(await r.text());
    if (!out) throw new IntegrationError("empty or unreadable reply");
    if (out.error) throw new IntegrationError(out.error.message ?? JSON.stringify(out.error));
    return out.result;
  }
  async connect() {
    if (this.#session === null && this.#id === 0) {
      await this.#rpc("initialize", { protocolVersion: PROTOCOL, capabilities: {}, clientInfo: { name: "loadopoly-integrations", version } });
      await this.#rpc("notifications/initialized", undefined, true);
    }
    return this;
  }
  async tools() { await this.connect(); return (await this.#rpc("tools/list", {})).tools ?? []; }
  async call(name, args = {}) {
    await this.connect();
    const r = await this.#rpc("tools/call", { name, arguments: args });
    const text = (r.content ?? []).filter((c) => c.type === "text").map((c) => c.text).join("");
    if (r.isError) throw new IntegrationError(text || "tool error");
    try { return JSON.parse(text); } catch { return text; }
  }
  status() { return this.call("process_studio_status"); }
  list(status) { return this.call("process_studio_list", status ? { status } : {}); }
  hubActions() { return this.call("hub_actions"); }
  /** Stage a proposal. It is held; a human authorises it elsewhere. */
  propose(tool, reason, args = {}) { return this.call("process_studio_propose", { tool, reason, arguments: args }); }
  /** Refused by the server unless a human Admin has the voice on. */
  voice() { return this.call("perceptopoly_voice"); }
}
