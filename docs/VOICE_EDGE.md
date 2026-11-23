# Speak a note on the phone — `quipu_voice.py`

```
Termux:API mic ─► speech service on the Hideout (faster-whisper) ─► you read and confirm ─► held proposal
```

A spoken note becomes **one `hub.body_directive`, severity `info`, owner role `Anyone`, held as
`PROPOSED`**. The words never choose the tool, the role or the severity, and a voice is never an
authorisation: a human decides in QUIPU (HubCore Part 0.3.10, V10-SEC-010). The transcript is untrusted
text. It is shown to you and staged only if you confirm (`y`), or with `--yes` in a script.

## The two halves

- **Service (Hideout)** — `python -m src.quipu.voice_stt --host 0.0.0.0 --port 7120`, with `QUIPU_STT_TOKEN`
  set (it answers 401 to everything without one). It takes audio and returns text; it has no access to the
  ledger, observer or HubCore. It needs `pip install 'faster-whisper>=1,<2'` (model `base.en`, CPU;
  `QUIPU_STT_MODEL=small.en` for more accuracy); without it the service answers 503 and says so. Clips are
  capped at 8 MB, held in a private temp file and deleted straight after.
- **Client (phone)** — `clients/termux/quipu_voice.py`. Recording uses `termux-microphone-record`; the script
  checks that this install lists the flags it uses (`-f -l -e`) and refuses rather than guesses. Grant
  **Termux:API** the *Microphone* permission.

## Configure (in Termux)

MCP settings come from `edge_setup.sh` (see `EDGE_DESKTOP_COMMANDER.md`). Then `~/.config/quipu/stt.env`:

```sh
QUIPU_STT_URL=https://<your-host>/transcribe
QUIPU_STT_TOKEN_FILE=/data/data/com.termux/files/home/.config/quipu/stt.token   # mode 600; type it in Termux, not chat
```

Expose :7120 like the MCP (a named-tunnel hostname such as `stt.<your-domain>` -> `http://<host>:7120`).
Plain `http` is accepted only for private hosts.

## Use

```sh
python clients/termux/quipu_voice.py doctor
python clients/termux/quipu_voice.py listen --seconds 10          # speak, read it back, confirm
python clients/termux/quipu_voice.py listen --file note.m4a --dry-run
```

## Verified, and not

Verified: 13 tests run the client against a real MCP server and observer and the real service with a
stand-in transcriber, plus a stand-in `termux-microphone-record`: held-only staging, the words
unable to pick tool/role/severity, refusals, no token in any output, the clip deleted after use.

Not verified: faster-whisper itself (not installed in the session that built this, so no real
transcription was run), a real microphone or Termux:API on a phone, the aac codec flag on your device, a
tunnel. This also only does speech *input*; there is no text-to-speech.
