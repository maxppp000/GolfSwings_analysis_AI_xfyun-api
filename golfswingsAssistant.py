import _thread as thread
import logging
import base64
import datetime
import threading
import json
import websocket
import hashlib
import hmac
from urllib.parse import urlparse, urlencode
import ssl
from datetime import datetime
from time import mktime, sleep
from wsgiref.handlers import format_date_time
import os

# Logging configuration is centralized in config.py.

class Ws_Param(object):
    """Spark API WebSocket parameter helper."""
    
    def __init__(self, APPID, APIKey, APISecret, imageunderstanding_url):
        """
        Initialize WebSocket parameters.
        
        Args:
            APPID (str): Spark API application ID
            APIKey (str): Spark API key
            APISecret (str): Spark API secret
            imageunderstanding_url (str): Image understanding WebSocket URL
        """
        self.APPID = APPID
        self.APIKey = APIKey
        self.APISecret = APISecret
        self.host = urlparse(imageunderstanding_url).netloc
        self.path = urlparse(imageunderstanding_url).path
        self.ImageUnderstanding_url = imageunderstanding_url

    def create_url(self):
        """Generate a signed WebSocket URL."""
        # Build an RFC1123 timestamp.
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        signature_origin = "host: " + self.host + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + self.path + " HTTP/1.1"

        signature_sha = hmac.new(self.APISecret.encode('utf-8'),
                                 signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()

        signature_sha_base64 = base64.b64encode(signature_sha).decode(encoding='utf-8')

        authorization_origin = f'api_key="{self.APIKey}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha_base64}"'

        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')

        v = {
            "authorization": authorization,
            "date": date,
            "host": self.host
        }
        url = self.ImageUnderstanding_url + '?' + urlencode(v)
        return url

class SparkWebSocketClient:
    """Generic Spark API WebSocket client."""
    
    def __init__(self, appid, api_secret, api_key, url):
        """
        Initialize the WebSocket client.
        
        Args:
            appid (str): Spark API application ID
            api_secret (str): Spark API secret
            api_key (str): Spark API key
        """
        self.appid = appid
        self.api_secret = api_secret
        self.api_key = api_key
        self.url = url
        self.ws_param = Ws_Param(appid, api_key, api_secret, url)
        self.result = {"answer": ""}
        self.done_event = threading.Event()
    
    def gen_params(self, question):
        """Build the payload for the Spark API."""
        data = {
            "header": {
                "app_id": self.appid,
                "uid": "12345"
            },
            "parameter": {
                "chat": {
                    "domain": "general",
                    "temperature": 0.5,
                    "max_tokens": 2048
                }
            },
            "payload": {
                "message": {
                    "text": question
                }
            }
        }
        return data
    
    def on_message(self, ws, message):
        """Handle messages from the WebSocket."""
        data = json.loads(message)
        code = data['header']['code']
        if code != 0:
            logging.error(f'Request error: {code}, {data}')
            ws.close()
        else:
            choices = data["payload"]["choices"]
            status = choices["status"]
            content = choices["text"][0]["content"]
            self.result["answer"] += content
            if status == 2:
                self.done_event.set()
                ws.close()
    
    def on_error(self, ws, error):
        logging.error(f"WebSocketError: {error}")
        self.done_event.set()
    
    def on_close(self, ws, *args):
        logging.info("WebSocket connection closed")
    
    def on_open(self, ws):
        logging.info("WebSocket connection established")
        
        def run(*args):
            data = self.gen_params(self.question)
            ws.send(json.dumps(data))
        thread.start_new_thread(run, ())
    
    def send_request(self, question):
        self.question = question
        self.result = {"answer": ""}
        self.done_event.clear()
        
        websocket.enableTrace(False)
        ws_url = self.ws_param.create_url()
        ws = websocket.WebSocketApp(ws_url,
                                   on_message=self.on_message,
                                   on_error=self.on_error,
                                   on_close=self.on_close,
                                   on_open=self.on_open)
        ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
        
        # Wait for the final response.
        self.done_event.wait(timeout=30)
        return self.result["answer"]


def assistant_answer(action, img_path, subdir=None):
    from config import SparkAPIConfig
    
    appid = SparkAPIConfig.APPID
    api_secret = SparkAPIConfig.API_SECRET
    api_key = SparkAPIConfig.API_KEY
    imageunderstanding_url = SparkAPIConfig.IMAGE_UNDERSTANDING_URL

    if subdir:
        img_path_full = f'static/uploads/{subdir}/key_frames/{img_path}'
    else:
        img_path_full = img_path
    
    try:
        with open(img_path_full, 'rb') as f:
            imagedata = f.read()
    except Exception as e:
        logging.error(f"Failed to read image: {e}")
        return f"Failed to read image: {e}"

    # Build prompts tailored to each action.
    action_prompts = {
        'Preparation': (
            "Provide a concise golf-swing preparation analysis. Focus on stance width, grip, shoulder alignment, and weight "
            "distribution. Limit the response to approximately 150 words."
        ),
        'Top_of_Backswing': (
            "Provide a concise top-of-backswing analysis. Highlight rotation depth, trail leg stability, wrist hinge, and "
            "spine tilt. Limit the response to approximately 150 words."
        ),
        'Impact': (
            "Analyze the impact frame. Comment on shaft lean, hip clearance, connection between arms and torso, and head "
            "position. Limit the response to approximately 150 words."
        ),
        'Finish': (
            "Analyze the finish position. Discuss balance, rotation completeness, arm extension, and tempo. "
            "Limit the response to approximately 150 words."
        )
    }
    
    prompt = action_prompts.get(
        action,
        "Provide a concise analysis of the golf-swing frame and highlight the most important technical checkpoints within roughly 150 words."
    )
    
    text = [
        {"role": "user", "content": str(base64.b64encode(imagedata), 'utf-8'), "content_type": "image"},
        {"role": "user", "content": prompt}
    ]

    # Send the request through the generic WebSocket client.
    client = SparkWebSocketClient(appid, api_secret, api_key, imageunderstanding_url)
    result = client.send_request(text)
    
    return result

def batch_assistant_analysis(actions_and_images, subdir=None):
    results = []
    
    def process_batch(batch, batch_index):
        """Process a batch of prompts."""
        batch_results = []
        for action, img_path in batch:
            try:
                def process_single_request(action, img_path):
                    """Send a single request to the assistant."""
                    result = assistant_answer(action, img_path, subdir)
                    return {"action": action, "result": result, "success": True}
                
                batch_results.append(process_single_request(action, img_path))
            except Exception as e:
                batch_results.append({"action": action, "result": str(e), "success": False})
        return batch_results
    
    batch_size = 2
    for i in range(0, len(actions_and_images), batch_size):
        batch = actions_and_images[i:i + batch_size]
        batch_results = process_batch(batch, i // batch_size)
        results.extend(batch_results)
    
    return results

