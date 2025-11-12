import os
from flask import Flask, render_template, send_from_directory, request

app = Flask(__name__, static_folder='static', template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/demo_results')
def demo_results():
    # Provide demo result data matching what templates expect
    result_data = {
        'input_video': 'dome_show/demo_show.mp4',
        'output_video': 'dome_show/analyzed_demo_show.mp4',
        'key_frames': [
            {
                'user_image': 'dome_show/key_frames/sample1.jpg',
                'standard_image': 'uploads/standard/key_frames/uoload_standard.jpg',
                'action': 'Preparation',
                'user_json': ''
            }
        ],
        'differences_summary': [],
        'detected_states': []
    }
    return render_template('results.html', result=result_data)

@app.route('/ai_analysis_demo')
def ai_analysis_demo():
    ai_results = [
        {'action': 'Preparation', 'description': 'Example: the preparation posture looks solid.', 'timestamp': '', 'image_path': 'static/uploads/standard/key_frames/uoload_standard.jpg'}
    ]
    return render_template('ai_analysis.html', ai_results=ai_results, subdir='demo', filename='demo.mp4')

# Minimal download endpoints used in templates; if the file isn't present, fall back to a demo static file
@app.route('/download/<subdir>/<filename>')
def download_file(subdir, filename):
    # try to serve from uploads folder
    path = os.path.join(app.root_path, 'static', subdir, 'result_video')
    file_path = os.path.join(path, filename)
    if os.path.exists(file_path):
        return send_from_directory(path, filename, as_attachment=True)
    # fallback to demo file
    fallback = os.path.join(app.root_path, 'static', 'dome_show', 'demo_show.mp4')
    if os.path.exists(fallback):
        return send_from_directory(os.path.join(app.root_path, 'static', 'dome_show'), 'demo_show.mp4', as_attachment=True)
    return 'File not found', 404

@app.route('/demo_download/<filename>')
def demo_download_file(filename):
    path = os.path.join(app.root_path, 'static', 'dome_show')
    if os.path.exists(os.path.join(path, filename)):
        return send_from_directory(path, filename, as_attachment=True)
    return 'File not found', 404

@app.route('/ai_analysis/<subdir>/<filename>')
def show_ai_analysis(subdir, filename):
    # Provide minimal demo AI analysis page
    ai_results = [
        {'action': 'Preparation', 'description': 'Example: the preparation posture looks solid.', 'timestamp': '', 'image_path': 'uploads/standard/key_frames/uoload_standard.jpg'}
    ]
    return render_template('ai_analysis.html', ai_results=ai_results, subdir=subdir, filename=filename)

@app.route('/guide')
def show_guide():
    # Minimal guide page rendering
    result_data = {
        'key_frames': request.args.getlist('key_frames'),
        'differences_summary': request.args.getlist('differences_summary'),
        'detected_states': request.args.getlist('detected_states')
    }
    return render_template('guide.html', result=result_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
