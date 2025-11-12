# -*- coding: utf-8 -*-
import os
import json
import glob
import logging
from datetime import datetime
from typing import Optional, Dict, Any

class CacheManager:
    """Manage cached AI analysis data stored alongside uploads."""
    
    def __init__(self, upload_folder: str):
        """Initialize the cache manager."""
        self.upload_folder = upload_folder
    
    def get_cache_path(self, subdir: str, action: str) -> Optional[str]:
        """Return the path to the cached JSON file for a given action."""
        cache_dir = os.path.join(self.upload_folder, subdir, 'result_video')
        
        cache_filename = f"desc_{subdir}_{action}_*.json"
        pattern = os.path.join(cache_dir, cache_filename)
        matches = glob.glob(pattern)
        
        if matches:
            matches.sort(key=os.path.getmtime, reverse=True)
            return matches[0]
        
        unrecognized_filename = f"desc_{subdir}_Unrecognized_{action}.json"
        unrecognized_pattern = os.path.join(cache_dir, unrecognized_filename)
        if os.path.exists(unrecognized_pattern):
            return unrecognized_pattern
        
        return None
    
    def load_cache_data(self, cache_path: str) -> Optional[Dict[str, Any]]:
        """Load cached analysis data from disk."""
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Failed to read cache file: {e}")
            return None
    
    def save_cache_data(self, subdir: str, filename: str, data: Dict[str, Any]) -> bool:
        """Persist cached analysis data."""
        try:
            cache_dir = os.path.join(self.upload_folder, subdir, 'result_video')
            os.makedirs(cache_dir, exist_ok=True)
            
            cache_filename = f"desc_{subdir}_{filename.replace('.jpg', '.json')}"
            cache_path = os.path.join(cache_dir, cache_filename)
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logging.info(f"Saved cache data: {cache_path}")
            return True
        except Exception as e:
            logging.error(f"Failed to save cache file: {e}")
            return False
    
    def get_cached_analysis(self, subdir: str, action: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached analysis results if they exist."""
        cache_path = self.get_cache_path(subdir, action)
        if cache_path:
            return self.load_cache_data(cache_path)
        return None
    
    def save_analysis_result(self, subdir: str, action: str, img_filename: str, 
                           analysis_result: str, image_path: str) -> bool:
        """Save analysis results to the cache."""
        cache_data = {
            'image_path': image_path,
            'subdir': subdir,
            'filename': img_filename,
            'description': analysis_result,
            'prompt': f"{action} frame analysis",
            'action': action,
            'frame_idx': None,
            'timestamp': datetime.now().isoformat()
        }
        
        return self.save_cache_data(subdir, img_filename, cache_data)
    
    def clear_cache(self, subdir: str, filename: str) -> bool:
        """Remove cached progress data for the provided upload."""
        try:
            progress_path = os.path.join(self.upload_folder, subdir, f'progress_{filename}.txt')
            if os.path.exists(progress_path):
                os.remove(progress_path)
                logging.info(f"Cleared progress cache: {progress_path}")
                return True
        except Exception as e:
            logging.error(f"Failed to clear cache: {e}")
            return False
        return False
