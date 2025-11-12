# Golf Swing Analysis System

## Overview
This project delivers a computer-vision powered golf swing analysis workflow. It extracts the key phases of a swing from uploaded videos, highlights posture differences, and generates AI-driven coaching notes. The stack is built with Flask, YOLO pose detection, and the Spark large-model API. The goal is to give golfers actionable, professional-grade feedback without leaving the browser.

## Feature Highlights
- **Intelligent phase detection:** Preparation, top of backswing, impact, and finish.
- **High-precision pose estimation:** YOLO-based keypoint extraction, angle measurement, and skeleton overlays.
- **Automatic keyframe capture:** Surfaces the clearest frame for every swing phase.
- **AI assistant insights:** Summaries, posture critiques, and improvement tips powered by Spark.
- **Visual reporting:** Web UI shows side-by-side comparisons and downloadable result videos.
- **History tracking:** Every processed swing stays available for review and re-download.

## UI Experience
- Responsive layout that works across desktop and mobile.
- Real-time progress indicators while analysis runs.
- Demo video, guide, and history pages for fast onboarding.
- Download button for the processed swing with overlays.

## Project Structure
```
├── app.py                   # Flask entry point and routes
├── golf_analysis.py         # Core video analysis pipeline
├── golfswingsAssistant.py   # Spark API integration
├── config.py                # Application configuration
├── cache_manager.py         # Caching helpers
├── common_utils.py          # Shared utilities
├── performance_optimization.py
├── static/                  # CSS, demo assets, uploads
└── templates/               # HTML templates
```

## Quick Start
1. **Clone the repo**
   ```bash
   git clone https://github.com/KongTilly/GolfSwings_analysis_AI_xfyun-api.git
   cd GolfSwings_analysis_AI_xfyun-api
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure Spark credentials**
   Update `SparkAPIConfig` in `config.py` with your `APPID`, `API_SECRET`, and `API_KEY`.
4. **Run the server**
   ```bash
   python3 app.py
   ```
5. **View the app** at `http://localhost:5001`.

Optional: enable CUDA for GPU acceleration when running pose detection.

## Usage Guide
1. **Upload a swing video** (MP4/AVI/MOV up to 100 MB).
2. **Track progress** in real time while the detector processes frames.
3. **Review the results**:
   - Keyframe gallery with user vs. standard images.
   - AI-generated notes per swing phase.
   - Downloadable annotated video.
4. **Open the history tab** to revisit previous analyses.

## Key Configuration
`config.py` centralizes runtime knobs:
- `FlaskConfig`: upload folder, max upload size, and allowed extensions.
- `GolfAnalysisConfig`: angle thresholds, drawing styles, and overlay colors.
- `SparkAPIConfig`: credentials and endpoint information.
- `DemoConfig`: demo asset metadata for the built-in showcase.
- `ErrorMessages`: human-readable messages returned by the backend.

## API Endpoints
- `POST /upload`: accept a video and kick off processing.
- `GET /progress/<subdir>/<filename>`: poll analysis progress.
- `GET /results/<subdir>/<filename>`: render the analysis page.
- `GET /ai_analysis/<subdir>/<filename>`: AI insights per keyframe.
- `GET /download/<subdir>/<filename>`: download annotated video.

## Development Tips
- **Add metrics:** extend `golf_analysis.py` with new calculations, then surface them in templates.
- **Customize the AI assistant:** adjust prompt building in `golfswingsAssistant.py`.
- **Support new formats:** expand `ALLOWED_EXTENSIONS` and verify OpenCV compatibility.

## License
Released under the MIT License. Contributions via Issues and Pull Requests are welcome.
