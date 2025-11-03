import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic


def draw_landmarks_on_image(modelOutput, image):
    mp_drawing.draw_landmarks(
        image,
        modelOutput.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
        connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style()
    ) 
    
    mp_drawing.draw_landmarks(
        image,
        modelOutput.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
        connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style()
    )
    
    mp_drawing.draw_landmarks(
        image,
        modelOutput.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
    ) 


def get_landmark_output(frame, landmark_solution):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    landmark_output = landmark_solution.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    return landmark_output, image


def launch_camera():
    cap = cv2.VideoCapture(0)  
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    with mp_holistic.Holistic(
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5
    ) as holistic:
        
        while cap.isOpened():
            success, frame = cap.read()
            
            if not success:
                print("Ignoring empty camera frame.")
                continue
            
            cv2.imshow('Original Camera Feed', frame)
            
            model_output, processed_image = get_landmark_output(frame, holistic)
            
            draw_landmarks_on_image(model_output, processed_image)
            
            cv2.imshow('Camera Feed with Landmarks', processed_image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Camera closed")


if __name__ == "__main__":
    launch_camera()
    
