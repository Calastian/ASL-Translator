import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic



def getLandmarkOutput(frame, landmarkSolution):
    """
    Returns:
    A NamedTuple with fields describing the landmarks on the most prominate
    person detected:
    1) "pose_landmarks" field that contains the pose landmarks.
    2) "pose_world_landmarks" field that contains the pose landmarks in
    real-world 3D coordinates that are in meters with the origin at the
    center between hips.
    3) "left_hand_landmarks" field that contains the left-hand landmarks.
    4) "right_hand_landmarks" field that contains the right-hand landmarks.
    5) "face_landmarks" field that contains the face landmarks.
    6) "segmentation_mask" field that contains the segmentation mask if
        "enable_segmentation" is set to true.
        
        
    Enum values for pose_landmarks and pose_world_landmarks are the same
    """

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    
    # Make detection
    landmarkOutput = landmarkSolution.process(image)
    #from MediaPipe documentation

    # Recolor back to BGR, openCV is weird like that
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    return landmarkOutput, image



def processVideo(cap, capType='video', showVideo=False, controllable=False)->dict:
    """
    This is a helper function with the purpose of encapsulation of the repeated pattern inside
    the different ways to process things.
    
    capType should be 'video' or 'feed'
    """
    data:dict = initialize_FrameModelOutput_Data()
    # Setup mediapipe holistic solution instance
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            #get input
            success, frame = cap.read()
            
            #error handling
            if not success:
                # If loading a video, use 'break' if using live feed, use 'continue'.
                # continue
                if capType == 'video':
                    break
                if capType == 'feed':
                    continue
                else:
                    break
            
            modelOutput, image = getLandmarkOutput(frame, holistic) 
            # tmp = modelOutputToLandmarkDict(modelOutput) #This line is just to see output laid out in a nicer dict, it's useful for trouble shooting
            append_FrameModelOutput_Data(modelOutput, data)
            
            if showVideo:
                showFrame(image,modelOutput)


            if controllable: 
                ##############
                #Control Logic
                ##############
                keyPressed = cv2.waitKey(1) & 0xFF
                #quit if q is pressed
                if keyPressed == ord('q'):
                    break
                
                #pause if p is pressed
                if keyPressed == ord('p'):
                    while cap.isOpened():
                        #unpause if p is pressed again
                        if cv2.waitKey(10) & 0xFF == ord('p'):
                            break
                    
    return data
    