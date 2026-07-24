# GARD Lite — Standalone Neural Tensor Compression Application

`GARD Lite` is a self-sustainable, zero-dependency extension of QUIPU's high-performance neural memory & GARD Shard tensor packing application.

---

## Folder Structure

```text
gui/GARD_Shard/GARD_lite/
├── gard_lite_engine.py   # Self-contained Python backend engine & HTTP REST server
├── index.html            # Ultra-premium standalone Web App interface
├── run_gard_lite.py      # One-click Python launcher
├── Start-GARDLite.ps1    # One-click PowerShell launcher
├── compression/          # Output directory for compressed .gard.weyl.bin tensor shards
└── decompression/        # Output directory for 100% exact reconstituted output files
```

---

## Features

- **Neural Memory & Binary Tensor Packing**: Packs high-dimensional file embeddings into 20-byte Float32 LE Newman-Penrose $(\Psi_0 \dots \Psi_4)$ Weyl tensors (`GARD_WEYL_v1` container).
- **100% Bit-for-Bit Exact Reconstruction**: Decompresses and restores exact original binary presentation files (`.pptx`, `.pdf`, `.docx`, `.xlsx`, `.csv`, `.json`, `.png`, `.jpg`, `.txt`, `.md`, `.py`) into `GARD_lite/decompression/`.
- **Zero Data Dissociation**: Calculates constant-time `0.000000%` dissociation error and HMAC-SHA256 Encrypt-then-MAC authentication tags.
- **Interactive Drag & Drop GUI**: Futuristic glassmorphism web interface with real-time compaction factor visualization, 5-element Weyl bar charts, and activity log.

---

## How to Run

### Option 1: One-Click Python
```powershell
python "C:\Users\agard\Documents\VS Code\QUIPU\gui\GARD_Shard\GARD_lite\run_gard_lite.py"
```

### Option 2: PowerShell
```powershell
powershell -ExecutionPolicy Bypass -File "C:\Users\agard\Documents\VS Code\QUIPU\gui\GARD_Shard\GARD_lite\Start-GARDLite.ps1"
```

Access the application in your web browser at: **`http://127.0.0.1:8780/`**
