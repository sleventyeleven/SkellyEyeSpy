from flask import Flask, render_template, request, jsonify, Response
import gpiod
import subprocess
import time
import os
from datetime import datetime

app = Flask(__name__)

# Pin definitions
MOUTH_OPEN_PIN = 20  # PIN_7 (gpiochip3 20)
EYE_LEDS_PIN = 1     # PIN_11 (gpiochip3 1)
BODY_MOTOR_PIN = 8   # PIN_15 (gpiochip3 8)

# GPIO chip
GPIO_CHIP = "gpiochip3"

# Audio control
I2S_ENABLED = False

def get_gpio_line(pin):
    """Get GPIO line for a specific pin"""
    try:
        chip = gpiod.Chip(GPIO_CHIP)
        line = chip.get_line(pin)
        return line
    except Exception as e:
        print(f"Error getting GPIO line: {e}")
        return None

def set_pin_state(pin, state):
    """Set GPIO pin state (True for HIGH, False for LOW)"""
    line = get_gpio_line(pin)
    if not line:
        return False

    try:
        line.request(consumer="animatronic_control", type=gpiod.LINE_REQ_DIR_OUT)
        line.set_value(1 if state else 0)
        line.release()
        return True
    except Exception as e:
        print(f"Error setting pin {pin} state: {e}")
        return False

def release_all_pins():
    """Release all GPIO pins"""
    try:
        # Release each pin individually
        for pin in [MOUTH_OPEN_PIN, EYE_LEDS_PIN, BODY_MOTOR_PIN]:
            line = get_gpio_line(pin)
            if line:
                try:
                    line.release()
                except:
                    pass  # Already released or not requested
    except Exception as e:
        print(f"Error releasing pins: {e}")

@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html')

# Animatronic Control Endpoints
@app.route('/api/mouth/open', methods=['POST'])
def open_mouth():
    """Open mouth motor"""
    try:
        success = set_pin_state(MOUTH_OPEN_PIN, True)
        if success:
            time.sleep(0.5)  # Keep motor active for half a second
            set_pin_state(MOUTH_OPEN_PIN, False)
            release_all_pins()
            return jsonify({"status": "success", "message": "Mouth opened"})
        else:
            return jsonify({"status": "error", "message": "Failed to open mouth"}), 500
    except Exception as e:
        release_all_pins()
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/eyes/on', methods=['POST'])
def turn_on_eyes():
    """Turn on eye LEDs"""
    try:
        success = set_pin_state(EYE_LEDS_PIN, True)
        if success:
            return jsonify({"status": "success", "message": "Eyes turned on"})
        else:
            return jsonify({"status": "error", "message": "Failed to turn on eyes"}), 500
    except Exception as e:
        release_all_pins()
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/eyes/off', methods=['POST'])
def turn_off_eyes():
    """Turn off eye LEDs"""
    try:
        success = set_pin_state(EYE_LEDS_PIN, False)
        if success:
            return jsonify({"status": "success", "message": "Eyes turned off"})
        else:
            return jsonify({"status": "error", "message": "Failed to turn off eyes"}), 500
    except Exception as e:
        release_all_pins()
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/body/move', methods=['POST'])
def move_body():
    """Move body motor (dance)"""
    try:
        success = set_pin_state(BODY_MOTOR_PIN, True)
        if success:
            time.sleep(1)  # Keep motor active for 1 second
            set_pin_state(BODY_MOTOR_PIN, False)
            release_all_pins()
            return jsonify({"status": "success", "message": "Body moved"})
        else:
            return jsonify({"status": "error", "message": "Failed to move body"}), 500
    except Exception as e:
        release_all_pins()
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
        # Check GPIO pins state (simplified for demo)
        status = {
            "gpio_pins": {
                "mouth_open": "unknown",
                "eye_leds": "unknown",
                "body_motor": "unknown"
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

if __name__ == '__main__':
    # Ensure pins are released on startup
    release_all_pins()

    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
