# -*- coding: utf-8 -*-
"""
Configuration management module.
Centralizes all configuration constants and settings.
"""

import os
import logging
from datetime import datetime

# Configure logging for the entire application.
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# Flask application configuration
class FlaskConfig:
    """Flask application configuration."""
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'static/uploads')
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB limit
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}
    MODEL_PATH = os.getenv('MODEL_PATH', '.')

# Golf analysis settings
class GolfAnalysisConfig:
    """Golf analysis configuration."""
    CONFIDENCE_THRESHOLD = 0.5
    DISPLACEMENT_THRESHOLD = 20
    WINDOW_SIZE = 20
    OFFSET_MM = 5
    OFFSET_PX = 20  # 5mm * 4px
    
    # Rendering configuration
    TEXT_FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX
    TEXT_SCALE = 2
    TEXT_THICKNESS = 5
    TEXT_COLOR = (0, 0, 0)
    TEXT_BG_COLOR = (255, 255, 255)
    POINT_COLOR = (0, 0, 255)
    LINE_COLOR = (0, 255, 0)
    
    # Pose key point visibility options
    SHOW_KEYPOINTS = {
        "nose": False, "left_eye": False, "right_eye": False, "left_ear": False, "right_ear": False,
        "left_shoulder": True, "right_shoulder": True, "left_elbow": True, "right_elbow": True,
        "left_wrist": True, "right_wrist": True, "middle_club": True, "head_club": True,
        "left_hip": True, "right_hip": True, "left_knee": True, "right_knee": True, "left_ankle": True,
        "right_ankle": True
    }
    
    # Skeleton line visibility options
    SHOW_SKELETON = {
        "left_ankle_to_left_knee": True, "left_knee_to_left_hip": True, "right_ankle_to_right_knee": True,
        "right_knee_to_right_hip": True, "left_hip_to_right_hip": True, "left_shoulder_to_left_hip": True,
        "right_shoulder_to_right_hip": True, "left_shoulder_to_right_shoulder": True,
        "left_shoulder_to_left_elbow": True, "right_shoulder_to_right_elbow": True,
        "left_elbow_to_left_wrist": True, "right_elbow_to_right_wrist": True,
        "middle_club_to_head_club": True, "left_wrist_to_middle_club": True,
        "left_shoulder_to_left_ankle": True, "right_shoulder_to_left_ankle": True,
    }
    
    # Angle overlay options
    SHOW_ANGLES = {
        "right_hip_angle": False, "right_knee_angle": False, "right_elbow_angle": False, "right_shoulder_angle": False,
        "left_hip_angle": False, "left_knee_angle": False, "left_elbow_angle": True, "left_shoulder_angle": False,
        "club_angle": True, "left_wrist_angle": True
    }
    
    # Angle color palette
    ANGLE_COLORS = {
        "right_hip_angle": (0, 255, 255), "right_knee_angle": (255, 255, 0), "right_elbow_angle": (255, 0, 255),
        "right_shoulder_angle": (0, 165, 255), "left_hip_angle": (0, 255, 0), "left_knee_angle": (255, 0, 0),
        "left_elbow_angle": (128, 0, 128), "left_shoulder_angle": (0, 128, 128), "club_angle": (255, 192, 203),
        "left_wrist_angle": (255, 100, 0)
    }
    
    # Angle name abbreviations
    ANGLE_SHORT_NAMES = {
        "right_hip_angle": "RH", "right_knee_angle": "RK", "right_elbow_angle": "RE", "right_shoulder_angle": "RS",
        "left_hip_angle": "LH", "left_knee_angle": "LK", "left_elbow_angle": "LE", "left_shoulder_angle": "LS",
        "club_angle": "CLUB", "left_wrist_angle": "LW"
    }

# Spark API configuration
class SparkAPIConfig:
    """Spark API configuration."""
    APPID = "XXXXXXXX"
    API_SECRET = "XXXXXXXXXXXXXXXXXXXXXXXX"
    API_KEY = "XXXXXXXXXXXXXXXXXXXXXXXX"
    IMAGE_UNDERSTANDING_URL = "wss://spark-api.cn-huabei-1.xf-yun.com/v2.1/image"

# Golf action types
GOLF_ACTIONS = ['Preparation', 'Top_of_Backswing', 'Impact', 'Finish']

# Demo data configuration
class DemoConfig:
    """Settings for demo data served by the app."""
    DEMO_SUBDIR = 'dome_show'
    DEMO_FILENAME = 'demo_show.mp4'
    
    # Helper for demo keyframe metadata
    def _create_demo_keyframe(action, frame_num):
        return {
            'action': action,
            'user_image': f'dome_show/key_frames/{action}_{frame_num}.jpg',
            'standard_image': f'uploads/standard/key_frames/standard_{action}.jpg',
            'user_json': f'/static/dome_show/result_video/desc_dome_show_{action}_{frame_num}.json'
        }
    
    def _create_demo_ai_result(action, frame_num):
        return {
            'action': action,
            'description': 'LoadingAnalysisResults...',
            'timestamp': '025-03-04T00:00:00',
            'image_path': f'dome_show/key_frames/{action}_{frame_num}.jpg',
            'user_json': f'/static/dome_show/result_video/desc_dome_show_{action}_{frame_num}.json'
        }
    
    DEMO_KEY_FRAMES = [
        _create_demo_keyframe('Preparation', 46),
        _create_demo_keyframe('Top_of_Backswing', 154),
        _create_demo_keyframe('Impact', 187),
        _create_demo_keyframe('Finish', 248)
    ]
    
    DEMO_AI_RESULTS = [
        _create_demo_ai_result('Preparation', 46),
        _create_demo_ai_result('Top_of_Backswing', 154),
        _create_demo_ai_result('Impact', 187),
        _create_demo_ai_result('Finish', 248)
    ]

# Error message templates
class ErrorMessages:
    """Standardized error messages."""
    UNRECOGNIZED_MSG = (
        "GolfSwingsAssistant is stumped.\n"
        "The club path was not detected as a complete swing.\n"
        "Verify the recording shows the full motion, compare with the reference video, and upload again."
    )
    ANALYSIS_FAILED = 'AI analysis failed'
    MISSING_PARAMS = 'Missing required parameters'
    PATH_FORMAT_ERROR = 'Image path format is invalid'
    ACTION_NOT_RECOGNIZED = 'Action type could not be recognized'

def allowed_file(filename):
    """Check whether the file extension is allowed."""
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in FlaskConfig.ALLOWED_EXTENSIONS
