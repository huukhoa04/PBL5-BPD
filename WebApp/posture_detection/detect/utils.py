import cv2
import numpy as np
import tensorflow as tf
import pickle
from utils.keypoints_utils import extract_keypoints, get_pose_results
from utils.visualization import draw_landmarks

model = tf.keras.models.load_model("models/best_model.resolved.h5")
with open("models/label_encoder.resolved.pkl", "rb") as f:
    label_encoder = pickle.load(f)

posture_colors = {
    "good_sitting": (0, 255, 0),     # Green
    "bad_sitting": (0, 0, 255),      # Red
    "sitting_forward": (0, 165, 255), # Orange
    "sitting_leanback": (0, 165, 255),# Orange
    "sitting_left": (0, 0, 255),      # Red
    "sitting_right": (0, 0, 255)      # Red
}

def get_multiple_predictions(prediction, threshold=0.5):
    """Get all postures with confidence above threshold"""
    postures = []
    for idx, conf in enumerate(prediction[0]):
        if conf >= threshold:
            posture = label_encoder.inverse_transform([idx])[0]
            postures.append((posture, conf))
    return sorted(postures, key=lambda x: x[1], reverse=True)

def wrap_text(text, max_width, font_face, font_scale, thickness):
    """Wrap text to fit within max_width pixels"""
    words = text.split(' ')
    lines = []
    current_line = []
    
    for word in words:
        current_line.append(word)
        line_text = ' '.join(current_line)
        (line_width, _) = cv2.getTextSize(line_text, font_face, font_scale, thickness)[0]
        
        if line_width > max_width:
            current_line.pop()
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return lines

def get_recommendations(postures):
    """Provide recommendations based on detected postures"""
    recommendations = set()
    for posture, _ in postures:
        if posture == "bad_sitting":
            recommendations.add("Straighten your back and adjust your position")
        elif posture == "sitting_forward":
            recommendations.add("Sit back and straighten your back")
        elif posture == "sitting_leanback":
            recommendations.add("Sit up and move closer to desk")
        elif posture in ["sitting_left", "sitting_right"]:
            recommendations.add("Center yourself with the screen")
    return list(recommendations)

def predict_posture(frame):
    pose_results = get_pose_results(frame)
    keypoints = extract_keypoints(frame)
    frame = draw_landmarks(frame, pose_results)

    if keypoints is not None:
        keypoints = np.expand_dims(keypoints, axis=0)
        keypoints = keypoints.reshape((1, 33, 3, 1))
        prediction = model.predict(keypoints, verbose=0)
        return prediction, frame
    return None, frame
def gen_frames():
    url="http://192.168.141.3:81/stream"
    cap = cv2.VideoCapture(url)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        prediction, frame = predict_posture(frame)
        
        if prediction is not None:
            postures = get_multiple_predictions(prediction, threshold=0.5)
            recommendations = get_recommendations(postures)
            
            for i, (posture, confidence) in enumerate(postures):
                color = posture_colors.get(posture, (255, 255, 255))
                cv2.putText(frame, f"{posture}: {confidence:.1%}", (10, 50 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            y_offset = 100 + len(postures) * 30
            for recommendation in recommendations:
                wrapped_lines = wrap_text("- " + recommendation, 400, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                for line in wrapped_lines:
                    cv2.putText(frame, line, (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    y_offset += 20

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
