


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
    
    
    
    
def processVideoFeed(deviceNum=0, showVideo:bool=False):
    cap = cv2.VideoCapture(deviceNum) #0 is for default video
    
    processVideo(cap, 'feed', showVideo)
    
    cap.release()
    cv2.destroyAllWindows()
            
    cap.release()
