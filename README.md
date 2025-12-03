## Prerequisites

- Windows PowerShell (the workflow script assumes PowerShell)
- Python 3.10+ available as `python` or `py`
- Internet access the first time you run the setup (downloads models and npm-less assets)

## Quick Start

1. **Clone the repo** and open a PowerShell window at the project root:
   ```powershell
   cd C:\path\to\MultiM_Exp_Sys\cis4930_multimedia_project
   ```

2. **Run the helper script**. It upgrades `pip`, installs requirements, downloads the spaCy model, opens the UI, and starts the Flask API:
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\start_story_insight.ps1
   ```

3. **Paste a script** into the browser textarea and click **Analyze Story**. The frontend posts to `http://127.0.0.1:5000/analyze` and renders the real rule scores.

4. **Stop the server** with `Ctrl+C` in the PowerShell window when you’re done.

## Testing & Troubleshooting

- **Health check**: `curl http://127.0.0.1:5000/health` should return `{"status": "ok"}`.
- **Metadata inputs**: Genre/Length/Mood/Audience/Setting dropdowns map directly to backend heuristics. Choose “N/A” to skip a field.
- **Common fixes**:
  - If spaCy model download fails, re-run `python -m spacy download en_core_web_sm`.
  - If the frontend shows `ERR_CONNECTION_REFUSED`, ensure `main.py` is running and the API URL matches the host/port.

## Project Structure

```
cis4930_multimedia_project/
├── main.py                # Flask API + scoring engine
├── index.html             # Tailwind UI
├── requirements.txt       # Python dependencies
└── start_story_insight.ps1# One-command setup/run script
```

