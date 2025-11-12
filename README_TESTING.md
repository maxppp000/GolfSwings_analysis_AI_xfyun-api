# Testing the GolfSwings AI app on iPhone

This document explains how to prepare and test the Flask app locally, and how to open it from an iPhone on the same Wi-Fi network.

Quick steps (recommended):

1. Open a terminal and change into the repository directory:

```bash
cd "/Users/cleanup/AI Golf swing/GolfSwings_analysis_AI_xfyun-api"
```

2. Run the helper script (it will create a `.venv`, install requirements, and start the app):

```bash
./run_local.sh
```

3. Get your Mac's local IP address (example):

```bash
ipconfig getifaddr en0
# if using a different interface or the command fails, try:
ifconfig | grep "inet " | grep -v 127.0.0.1
```

4. On your iPhone (connected to the same Wi‑Fi) open Safari and visit:

```
http://<your-mac-ip>:5000
```

Notes and troubleshooting

- The script uses `python3 -m venv .venv`. If your system python is different, replace `python3` with the path to the Python 3 binary you want to use.
- Installing requirements may take a long time if heavy libs (torch, etc.) are present. Consider installing inside a conda env if you prefer.
- If you need HTTPS (recommended for some Safari features), use a tunneling service like `ngrok`:

```bash
ngrok http 5000
```

Then open the generated `https://...` URL on your iPhone.

- To stop the app: run `pkill -f 'python3 app.py'` or check `app.log` for the process ID.

- If video playback is not inline on iOS, make sure your video files are served as MP4/H.264; the templates include `playsinline` and `webkit-playsinline` attributes.

If you'd like, I can:
- Start the app now and show the local IP.
- Install requirements for you (this could take time).
- Set up an ngrok tunnel and provide an HTTPS URL for testing on-device.
