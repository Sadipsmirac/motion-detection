# Save this complete file as app.py
from flask import Flask, render_template_string, jsonify, request
import base64
import numpy as np
import cv2
import json
import os
import urllib.request
import threading
import time

app = Flask(_name_)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Global variables
model_loaded = False
net = None
output_layers = None
classes = []

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Motion & Object Detection</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 30px;
        }
        .mode-selector {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-bottom: 30px;
        }
        .mode-btn {
            padding: 12px 30px;
            font-size: 16px;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.3s;
            background: #e0e7ff;
            color: #4c51bf;
        }
        .mode-btn.active {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .video-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }
        .video-wrapper {
            position: relative;
            background: #000;
            border-radius: 15px;
            overflow: hidden;
            min-height: 300px;
        }
        video, canvas {
            width: 100%;
            height: auto;
            display: block;
        }
        .controls {
            display: flex;
            gap: 15px;
            justify-content: center;
            margin: 20px 0;
        }
        button {
            padding: 12px 30px;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            font-size: 16px;
            transition: all 0.3s;
        }
        button:hover:not(:disabled) {
            transform: translateY(-2px);
        }
        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .start-btn {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white;
        }
        .stop-btn {
            background: #ef4444;
            color: white;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .stat-card {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 20px;
            border-radius: 15px;
            text-align: center;
        }
        .motion-bar {
            height: 40px;
            background: #e5e7eb;
            border-radius: 10px;
            overflow: hidden;
            margin: 20px 0;
        }
        .motion-fill {
            height: 100%;
            background: linear-gradient(90deg, #10b981 0%, #34d399 100%);
            width: 0%;
            transition: width 0.3s;
        }
        @media (max-width: 768px) {
            .video-container { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 Motion & Object Detection</h1>
        
        <div class="mode-selector">
            <button class="mode-btn active" onclick="switchMode('simple')">
                🎥 Simple Motion
            </button>
            <button class="mode-btn" onclick="switchMode('ai')">
                🤖 AI Detection
            </button>
        </div>

        <div class="video-container">
            <div class="video-wrapper">
                <video id="video" autoplay muted></video>
            </div>
            <div class="video-wrapper">
                <canvas id="canvas"></canvas>
            </div>
        </div>

        <div class="motion-bar">
            <div class="motion-fill" id="motionBar"></div>
        </div>

        <div class="stats">
            <div class="stat-card">
                <div>Current Motion</div>
                <div id="motionLevel" style="font-size: 2em;">0%</div>
            </div>
            <div class="stat-card">
                <div>Objects Detected</div>
                <div id="objectCount" style="font-size: 2em;">0</div>
            </div>
            <div class="stat-card">
                <div>FPS</div>
                <div id="fps" style="font-size: 2em;">0</div>
            </div>
        </div>

        <div class="controls">
            <button class="start-btn" onclick="startDetection()">▶ Start</button>
            <button class="stop-btn" onclick="stopDetection()">⏹ Stop</button>
        </div>
    </div>

    <script>
        let stream = null;
        let animationId = null;
        let previousFrame = null;
        let mode = 'simple';
        let detecting = false;

        function switchMode(newMode) {
            mode = newMode;
            document.querySelectorAll('.mode-btn').forEach(btn => {
                btn.classList.remove('active');
            });
            event.target.classList.add('active');
        }

        async function startDetection() {
            const video = document.getElementById('video');
            const canvas = document.getElementById('canvas');
            const ctx = canvas.getContext('2d');
            
            try {
                stream = await navigator.mediaDevices.getUserMedia({video: true});
                video.srcObject = stream;
                
                video.addEventListener('loadedmetadata', () => {
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                    detecting = true;
                    detect();
                });
            } catch(err) {
                alert('Camera access denied: ' + err.message);
            }
        }

        function stopDetection() {
            detecting = false;
            if(stream) {
                stream.getTracks().forEach(track => track.stop());
                stream = null;
            }
            if(animationId) {
                cancelAnimationFrame(animationId);
                animationId = null;
            }
        }

        function detect() {
            if(!detecting) return;
            
            const video = document.getElementById('video');
            const canvas = document.getElementById('canvas');
            const ctx = canvas.getContext('2d');
            
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            
            if(mode === 'simple') {
                // Simple motion detection
                const currentFrame = ctx.getImageData(0, 0, canvas.width, canvas.height);
                
                if(previousFrame) {
                    let motion = 0;
                    for(let i = 0; i < currentFrame.data.length; i += 4) {
                        const diff = Math.abs(currentFrame.data[i] - previousFrame.data[i]) +
                                    Math.abs(currentFrame.data[i+1] - previousFrame.data[i+1]) +
                                    Math.abs(currentFrame.data[i+2] - previousFrame.data[i+2]);
                        
                        if(diff > 30) {
                            motion++;
                            currentFrame.data[i] = 255;
                            currentFrame.data[i+1] = 0;
                            currentFrame.data[i+2] = 0;
                        }
                    }
                    
                    const motionPercent = (motion / (currentFrame.data.length/4)) * 100;
                    document.getElementById('motionLevel').textContent = motionPercent.toFixed(1) + '%';
                    document.getElementById('motionBar').style.width = Math.min(motionPercent * 5, 100) + '%';
                    
                    ctx.putImageData(currentFrame, 0, 0);
                }
                previousFrame = currentFrame;
                
            } else {
                // AI detection mode - send to server
                canvas.toBlob(async (blob) => {
                    const formData = new FormData();
                    formData.append('image', blob);
                    
                    try {
                        const response = await fetch('/detect', {
                            method: 'POST',
                            body: formData
                        });
                        const result = await response.json();
                        document.getElementById('objectCount').textContent = result.objects || 0;
                    } catch(err) {
                        console.error('Detection error:', err);
                    }
                });
            }
            
            animationId = requestAnimationFrame(detect);
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/detect', methods=['POST'])
def detect():
    # Simple detection endpoint - you can enhance this with YOLO later
    try:
        # For now, return mock data
        return jsonify({
            'success': True,
            'objects': np.random.randint(0, 5),
            'message': 'Detection completed'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/download_model')
def download_model():
    global model_loaded, net, output_layers
    
    try:
        # Download YOLO files if needed
        files = {
            'yolov3-tiny.cfg': 'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg',
            'yolov3-tiny.weights': 'https://pjreddie.com/media/files/yolov3-tiny.weights'
        }
        
        for filename, url in files.items():
            if not os.path.exists(filename):
                print(f"Downloading {filename}...")
                urllib.request.urlretrieve(url, filename)
        
        # Load model
        net = cv2.dnn.readNet('yolov3-tiny.weights', 'yolov3-tiny.cfg')
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        model_loaded = True
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if _name_ == '_main_':
    print("="*50)
    print("Motion Detection Server Starting...")
    print("Open http://localhost:5000 in your browser")
    print("="*50)
    app.run(host='0.0.0.0', port=5000, debug=False)