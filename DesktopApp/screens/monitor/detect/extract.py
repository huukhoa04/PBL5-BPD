import cv2
import numpy as np
def extract_and_preprocess_keypoints(keypoints, scaler_posture, scaler_leg):
    """Extract and preprocess keypoints for both models"""
    # Extract upper body keypoints for posture
    upper_indices = list(range(0, 23))
    upper_keypoints = []
    for idx in upper_indices:
        start_idx = idx * 3
        upper_keypoints.extend(keypoints[start_idx:start_idx + 3])
    upper_keypoints = np.array(upper_keypoints).reshape(1, -1)
    
    # Extract leg keypoints
    leg_indices = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
    leg_keypoints = []
    for idx in leg_indices:
        start_idx = idx * 3
        leg_keypoints.extend(keypoints[start_idx:start_idx + 3])
    leg_keypoints = np.array(leg_keypoints).reshape(1, -1)
    
    # Normalize keypoints
    upper_keypoints_normalized = scaler_posture.transform(upper_keypoints)
    leg_keypoints_normalized = scaler_leg.transform(leg_keypoints)
    
    return upper_keypoints_normalized, leg_keypoints_normalized

def draw_results(frame, leg_pred, posture_pred, leg_classes, posture_classes):
    """Draw detection results on frame with probabilities"""
    # Get frame dimensions
    height, width = frame.shape[:2]
    
    # Draw leg position result
    leg_prob = leg_pred[0][0]
    leg_text = f"{leg_classes[1]}: {leg_prob:.1%}" if leg_prob > 0.5 else f"{leg_classes[0]}: {(1-leg_prob):.1%}"
    leg_color = (0, 255, 0) if leg_prob <= 0.5 else (0, 0, 255)  # Green for correct, Red for wrong
    cv2.putText(frame, leg_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, leg_color, 2)

    # Only show posture predictions if legs are in correct position
    if leg_prob <= 0.5:  # Correct leg position
        # Draw all posture probabilities
        posture_probs = posture_pred[0]
        max_prob_idx = np.argmax(posture_probs)
        
        # Draw probabilities for all postures
        y_offset = 70
        for i, (posture, prob) in enumerate(zip(posture_classes, posture_probs)):
            # Format text
            text = f"{posture}: {prob:.1%}"
            
            # Determine color and size based on whether this is the highest probability
            if i == max_prob_idx:
                color = (0, 255, 0)  # Green for highest probability
                font_scale = 1.0
                thickness = 2
            else:
                color = (200, 200, 200)  # Gray for others
                font_scale = 0.6
                thickness = 1
            
            # Draw text
            cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale, color, thickness)
            y_offset += 30
    else:
        # Show warning about incorrect leg position
        warning = "Please correct leg position first!"
        cv2.putText(frame, warning, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, (0, 0, 255), 2)

    return frame





# import cv2
# import numpy as np
# def extract_keypoints( results):
#         """Extract keypoints from MediaPipe results directly"""
#         if results.pose_landmarks:
#             keypoints = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten()
#             return keypoints
#         return None

# def extract_head_keypoints(results):
#         """Extract head keypoints (nose, eyes, ears)"""
#         if results.pose_landmarks:
#             landmarks = results.pose_landmarks.landmark
#             head_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # MediaPipe indices for head landmarks
#             head_points = np.array([[landmarks[i].x, landmarks[i].y, landmarks[i].z] for i in head_indices]).flatten()
#             return head_points
#         return None

# def extract_body_keypoints(results):
#         """Extract torso keypoints (shoulders, chest, hips)"""
#         if results.pose_landmarks:
#             landmarks = results.pose_landmarks.landmark
#             body_indices = [11, 12, 23, 24]  # MediaPipe indices for body landmarks
#             body_points = np.array([[landmarks[i].x, landmarks[i].y, landmarks[i].z] for i in body_indices]).flatten()
#             return body_points
#         return None

# def extract_arm_keypoints(results):
#         """Extract arm keypoints (shoulders to wrists)"""
#         if results.pose_landmarks:
#             landmarks = results.pose_landmarks.landmark
#             arm_indices = [11, 13, 15, 12, 14, 16]  # MediaPipe indices for arm landmarks
#             arm_points = np.array([[landmarks[i].x, landmarks[i].y, landmarks[i].z] for i in arm_indices]).flatten()
#             return arm_points
#         return None

# def extract_leg_keypoints(results):
#         """Extract leg keypoints (hips to ankles)"""
#         if results.pose_landmarks:
#             landmarks = results.pose_landmarks.landmark
#             leg_indices = [23, 25, 27, 24, 26, 28]  # MediaPipe indices for leg landmarks
#             leg_points = np.array([[landmarks[i].x, landmarks[i].y, landmarks[i].z] for i in leg_indices]).flatten()
#             return leg_points
#         return None

# def extract_foot_keypoints(results):
#         """Extract foot keypoints (feet landmarks)"""
#         if results.pose_landmarks:
#             landmarks = results.pose_landmarks.landmark
#             foot_indices = [29, 30, 31, 32]  # MediaPipe indices for foot landmarks
#             foot_points = np.array([[landmarks[i].x, landmarks[i].y, landmarks[i].z] for i in foot_indices]).flatten()
#             return foot_points
#         return None

# def get_multiple_predictions(label_encoder,prediction, threshold=0.5):
#         """Get prediction labels above threshold"""
#         predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
#         return predicted_label

# def display_postures(frame, postures):
#         """Display posture labels on the frame"""
#         # Convert posture name to string if it's not already
#         posture_text = str(postures[0]) if isinstance(postures, (list, np.ndarray)) else str(postures)
        
#         # Add text to frame with proper font scale and positioning
#         cv2.putText(
#             frame,
#             posture_text,
#             (20, 40),  # Position at top-left with some margin
#             cv2.FONT_HERSHEY_SIMPLEX,
#             1,  # Font scale
#             (255, 0, 0),  # Blue color in RGB
#             2,  # Line thickness
#             cv2.LINE_AA  # Anti-aliased line type
#         )


