"""
Performance Optimization Module for Golf Swing Analysis App
Adds: caching, compression, lazy loading, async operations, database connection pooling
"""

from functools import wraps, lru_cache
import time
import gzip
import io
from flask import request, Response
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# CACHING DECORATORS
# ============================================================================

def cache_result(timeout=3600):
    """
    Decorator to cache function results for specified timeout (seconds).
    Useful for expensive operations that don't change frequently.
    """
    def decorator(func):
        cache_dict = {}
        cache_time = {}
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from args and kwargs
            key = str((args, tuple(sorted(kwargs.items()))))
            now = time.time()
            
            # Return cached result if still valid
            if key in cache_dict and now - cache_time.get(key, 0) < timeout:
                logger.debug(f"Cache hit for {func.__name__}")
                return cache_dict[key]
            
            # Compute new result
            result = func(*args, **kwargs)
            cache_dict[key] = result
            cache_time[key] = now
            logger.debug(f"Cache miss for {func.__name__} - computed new result")
            return result
        
        return wrapper
    return decorator

# ============================================================================
# COMPRESSION
# ============================================================================

def gzip_response(func):
    """
    Decorator to gzip Flask response if client accepts it.
    Reduces bandwidth usage significantly for large responses.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        response = func(*args, **kwargs)
        
        # Check if client accepts gzip
        if 'gzip' not in request.headers.get('Accept-Encoding', ''):
            return response
        
        # If response is already a Response object, get data
        if isinstance(response, Response):
            data = response.get_data()
        else:
            data = str(response).encode('utf-8')
        
        # Compress
        gzip_buffer = io.BytesIO()
        with gzip.GzipFile(fileobj=gzip_buffer, mode='wb') as gz:
            gz.write(data)
        
        gzip_data = gzip_buffer.getvalue()
        
        # Create response
        response = Response(gzip_data)
        response.headers['Content-Encoding'] = 'gzip'
        response.headers['Content-Length'] = len(gzip_data)
        
        logger.debug(f"Compressed response: {len(data)} -> {len(gzip_data)} bytes")
        return response
    
    return wrapper

# ============================================================================
# QUERY OPTIMIZATION
# ============================================================================

@lru_cache(maxsize=128)
def get_cached_action_names():
    """Cache golf action names since they don't change."""
    return [
        'Preparation',
        'Top_of_Backswing',
        'Impact',
        'Finish'
    ]

# ============================================================================
# IMAGE OPTIMIZATION
# ============================================================================

def optimize_image_for_mobile(image_path, max_width=420, quality=85):
    """
    Optimize images for mobile viewing:
    - Resize to mobile screen width
    - Reduce quality to balance size/appearance
    - Strip unnecessary metadata
    
    Args:
        image_path: Path to image file
        max_width: Maximum width in pixels
        quality: JPEG quality (1-100)
    """
    try:
        from PIL import Image
        import os
        
        img = Image.open(image_path)
        
        # Calculate new height maintaining aspect ratio
        if img.width > max_width:
            ratio = max_width / img.width
            new_height = int(img.height * ratio)
            img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert RGBA to RGB if necessary (for JPEG compatibility)
        if img.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
            img = background
        
        # Save optimized image
        img.save(image_path, 'JPEG', quality=quality, optimize=True)
        
        logger.info(f"Optimized image: {image_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to optimize image {image_path}: {e}")
        return False

# ============================================================================
# VIDEO OPTIMIZATION
# ============================================================================

def generate_video_thumbnail(video_path, output_path, timestamp=1):
    """
    Generate a thumbnail from a video at specified timestamp.
    Used for lazy-loading video previews.
    
    Args:
        video_path: Path to video file
        output_path: Where to save thumbnail
        timestamp: Seconds into video to capture (default: 1 second)
    """
    try:
        import cv2
        
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(timestamp * cap.get(cv2.CAP_PROP_FPS)))
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            cv2.imwrite(output_path, frame)
            # Optimize the thumbnail
            optimize_image_for_mobile(output_path, max_width=420, quality=70)
            logger.info(f"Generated thumbnail: {output_path}")
            return True
        
        logger.error(f"Failed to extract frame from {video_path}")
        return False
    except Exception as e:
        logger.error(f"Failed to generate thumbnail: {e}")
        return False

# ============================================================================
# DATABASE CONNECTION POOLING (for future use)
# ============================================================================

class ConnectionPool:
    """Simple connection pool for database connections."""
    def __init__(self, create_func, max_size=5):
        self.create_func = create_func
        self.max_size = max_size
        self.pool = []
        self.in_use = set()
    
    def get(self):
        """Get a connection from the pool."""
        if self.pool:
            conn = self.pool.pop()
        else:
            conn = self.create_func()
        self.in_use.add(id(conn))
        return conn
    
    def release(self, conn):
        """Return a connection to the pool."""
        self.in_use.discard(id(conn))
        if len(self.pool) < self.max_size:
            self.pool.append(conn)
        else:
            try:
                conn.close()
            except:
                pass

# ============================================================================
# PERFORMANCE MONITORING
# ============================================================================

def log_performance(func):
    """
    Decorator to log execution time of functions.
    Helps identify performance bottlenecks.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        
        if elapsed > 1.0:  # Log slow operations (>1 second)
            logger.warning(f"Slow operation: {func.__name__} took {elapsed:.2f}s")
        else:
            logger.debug(f"{func.__name__} took {elapsed:.3f}s")
        
        return result
    return wrapper

# ============================================================================
# LAZY LOADING SUPPORT
# ============================================================================

def create_lazy_load_html(image_src, placeholder_color="#f0f0f0"):
    """
    Generate HTML for lazy-loaded images using intersection observer.
    Reduces initial page load time by deferring image loading.
    """
    svg_data = f"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 400 300'%3E%3Crect fill='{placeholder_color}' width='400' height='300'/%3E%3C/svg%3E"
    return f'''
    <img class="lazy-load" 
         src="{svg_data}"
         data-src="{image_src}"
         alt="Loading...">
    <script>
        if ('IntersectionObserver' in window) {{
            const observer = new IntersectionObserver((entries) => {{
                entries.forEach(entry => {{
                    if (entry.isIntersecting) {{
                        const img = entry.target;
                        img.src = img.dataset.src;
                        observer.unobserve(img);
                    }}
                }});
            }});
            document.querySelectorAll('.lazy-load').forEach(img => observer.observe(img));
        }}
    </script>
    '''

# ============================================================================
# RESPONSE OPTIMIZATION
# ============================================================================

def minimize_json_response(data):
    """
    Minimize JSON response size by removing unnecessary fields.
    Reduces bandwidth for mobile users.
    """
    import json
    # Remove None values and empty strings
    def clean_dict(d):
        if isinstance(d, dict):
            return {k: clean_dict(v) for k, v in d.items() if v not in (None, '', [], {})}
        elif isinstance(d, list):
            return [clean_dict(item) for item in d]
        return d
    
    return json.dumps(clean_dict(data), separators=(',', ':'))

if __name__ == '__main__':
    print("Golf Swing Analysis - Performance Optimization Module")
    print("Available decorators and functions:")
    print("  - @cache_result(timeout): Cache function results")
    print("  - @gzip_response: Compress Flask responses")
    print("  - @log_performance: Log slow operations")
    print("  - optimize_image_for_mobile: Optimize images for mobile")
    print("  - generate_video_thumbnail: Create video thumbnails")
    print("  - create_lazy_load_html: Generate lazy-loading HTML")
