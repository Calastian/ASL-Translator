import os
import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic


# 0 stands for the default video capture device, you can also file a file path in this parameter instead to read a video
# cap = cv2.VideoCapture(0)

rootdir = '../ASL_Citizen/videos' #replace with your own directory
for dirpath, dirnames, filenames in os.walk(rootdir):
    for filename in filenames:
        if filename.endswith(".mp4") or filename.endswith(".mov"): #add more video formats if needed
            cap = cv2.VideoCapture(os.path.join(dirpath, filename))
# cap = cv2.VideoCapture()
# Setup mediapipe holistic solution instance
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        #get input
        success, frame = cap.read()
        
        #error handling
        if not success:
            # If loading a video, use 'break' instead of 'continue'.
            #continue
            break

        # Recolor image to RGB, openCV is weird like that
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        modelOutput = holistic.process(image)
        #from MediaPipe documentation
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
        """
    
        # Recolor back to BGR, openCV is weird like that
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        ###################
        # Render detections
        ###################
        #draws face contours
        mp_drawing.draw_landmarks(
            image,
           modelOutput.face_landmarks,
           mp_holistic.FACEMESH_CONTOURS,
           landmark_drawing_spec=None,
           connection_drawing_spec=mp_drawing_styles
          .get_default_face_mesh_contours_style(0))# there are currently 2 styles for .get_default_face_mesh_contours_style(), integer argument determines which is used 
        # #draws face tesselation # I have it turned off because I think its ugly with the default style
        # mp_drawing.draw_landmarks(
        #     image,
        #     modelOutput.face_landmarks,
        #     mp_holistic.FACEMESH_TESSELATION,
        #     landmark_drawing_spec=mp_drawing_styles
        #     .get_default_face_mesh_tesselation_style())
        #draws left hand
        mp_drawing.draw_landmarks(
            image,
            modelOutput.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles
            .get_default_hand_landmarks_style()
            ,connection_drawing_spec=mp_drawing_styles
            .get_default_hand_connections_style()
            ) 
        #draws right hand 
        mp_drawing.draw_landmarks(
            image,
            modelOutput.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles
            .get_default_hand_landmarks_style(),
            connection_drawing_spec=mp_drawing_styles
            .get_default_hand_connections_style())
        #draws pose
        mp_drawing.draw_landmarks(
            image,
            modelOutput.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles
            .get_default_pose_landmarks_style()) 
        
        #show new image in python window
        # cv2.imshow('Mediapipe Feed', image)

        ################
        #print landmarks
        ################
        #could raise an error so we need to surround it in try catch
        try:
            # print(vars(modelOutput)) # astrick unwraps tuple
           
            if modelOutput.pose_landmarks:
                print(modelOutput.pose_landmarks)
            if modelOutput.pose_world_landmarks:
                print(modelOutput.pose_world_landmarks)
            if modelOutput.left_hand_landmarks:
                print(modelOutput.left_hand_landmarks)
            if modelOutput.right_hand_landmarks:
                print(modelOutput.right_hand_landmarks)
            if modelOutput.face_landmarks:
                print(modelOutput.face_landmarks)
            # if modelOutput.segmentation_mask:
            #    print(modelOutput.segmentation_mask)

            

        except:
            pass

        #quit if q is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        
        break
        
    cap.release()
    cv2.destroyAllWindows()
