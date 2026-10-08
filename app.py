from flask import Flask, render_template, request, jsonify, Response
import os
import subprocess
import time
from datetime import datetime

from skelly import Skeleton

app = Flask(__name__)

# GPIO goes through skelly.hardware (python-periphery on the board, a
# simulated backend on a dev machine). Constructing it touches no pins.
skeleton = Skeleton()

# Audio control
I2S_ENABLED = False


def register_photobooth(flask_app):
    """Optional: needs the photobooth dependencies (opencv, mediapipe) --
    absent from the old system-python setup. Never let a missing or broken
    booth take the animatronic controls down with it.
    """
    config_path = os.environ.get("PHOTOBOOTH_CONFIG", "configs/skelly_zero3w.yaml")
    try:
        from photobooth.config import PhotoBoothConfig
        from photobooth.service import PhotoBoothService
        from photobooth.web import create_blueprint

        config = PhotoBoothConfig.from_yaml(config_path)
        flask_app.register_blueprint(create_blueprint(PhotoBoothService(config)))
        return True
    except Exception as e:  # noqa: BLE001
        print(f"Photo booth disabled: {type(e).__name__}: {e}")
        return False


@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html')

# Animatronic Control Endpoints
@app.route('/api/mouth/open', methods=['POST'])
def open_mouth():
    """Open mouth motor"""
    try:
        skeleton.mouth_open()
        time.sleep(0.5)  # Keep motor active for half a second
        skeleton.mouth_close()
        return jsonify({"status": "success", "message": "Mouth opened"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/eyes/on', methods=['POST'])
def turn_on_eyes():
    """Turn on eye LEDs"""
    try:
        skeleton.eyes_on()
        return jsonify({"status": "success", "message": "Eyes turned on"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/eyes/off', methods=['POST'])
def turn_off_eyes():
    """Turn off eye LEDs"""
    try:
        skeleton.eyes_off()
        return jsonify({"status": "success", "message": "Eyes turned off"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/body/move', methods=['POST'])
def move_body():
    """Move body motor (dance)"""
    try:
        skeleton.body_start()
        time.sleep(1)  # Keep motor active for 1 second
        skeleton.body_stop()
        return jsonify({"status": "success", "message": "Body moved"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# Audio Control Endpoints
@app.route('/api/audio/play', methods=['POST'])
def play_audio():
    """Play audio through I2S"""
    try:
        # Enable I2S if not already enabled
        global I2S_ENABLED
        if not I2S_ENABLED:
            # This would normally require system-level changes
            # For now, we'll just simulate enabling
            I2S_ENABLED = True

        # In a real implementation, this would play audio files
        # For demo purposes, we'll just return success
        return jsonify({"status": "success", "message": "Audio playback started"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/audio/stop', methods=['POST'])
def stop_audio():
    """Stop audio playback"""
    try:
        # In a real implementation, this would stop audio playback
        global I2S_ENABLED
        I2S_ENABLED = False
        return jsonify({"status": "success", "message": "Audio playback stopped"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# Webcam Capture Endpoints
@app.route('/api/camera/capture', methods=['POST'])
def capture_photo():
    """Capture photo from webcam"""
    try:
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"image_{timestamp}.jpg"

        # Capture image using fswebcam
        cmd = [
            "fswebcam",
            "-d", "/dev/video0",
            "--no-banner",
            "-r", "1920x1080",
            filename
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            return jsonify({
                "status": "success",
                "message": f"Photo captured: {filename}",
                "file": filename
            })
        else:
            return jsonify({
                "status": "error",
                "message": f"Failed to capture photo: {result.stderr}"
            }), 500

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/camera/stream')
def stream_video():
    """Stream video from webcam"""
    def generate():
        # This would implement streaming logic
        # For demo purposes, we'll just yield a placeholder
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Status Endpoints
@app.route('/api/status')
def get_status():
    """Get system status"""
    try:
        state = skeleton.state()
        status = {
            "gpio_backend": type(skeleton.backend).__name__,
            "gpio_pins": {
                "mouth_open": state["mouth_open"],
                "eye_leds": state["eyes"],
                "body_motor": state["body"]
            },
            "audio": {
                "i2s_enabled": I2S_ENABLED
            },
            "webcam": {
                "detected": True  # Simplified for demo
            }
        }

        return jsonify({"status": "success", "data": status})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

register_photobooth(app)

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=True)
    finally:
        skeleton.cleanup()
