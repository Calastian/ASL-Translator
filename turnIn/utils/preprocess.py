# %%
import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

import pandas as pd

# %% [markdown]
# # Showing Video

# %%
def drawLandmarksOnImage(modelOutput, image):
    ###################
    # Render detections on image
    ###################
    #draws face contours
    # mp_drawing.draw_landmarks(
    #     image,
    #     modelOutput.face_landmarks,
    #     mp_holistic.FACEMESH_CONTOURS,
    #     landmark_drawing_spec=None,
    #     connection_drawing_spec=mp_drawing_styles
    #     .get_default_face_mesh_contours_style(0))# there are currently 2 styles for .get_default_face_mesh_contours_style(), integer argument determines which is used 
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

# %%
def showFrame(image, modelOutput):
    #show original frame
    cv2.imshow("Original Feed", image)
    
    #show new image in python window
    drawLandmarksOnImage(modelOutput, image)
    cv2.imshow('Mediapipe Feed', image)
    
    #show only detections in python window
    blackScreen = np.zeros(image.shape)
    justDetections = blackScreen
    drawLandmarksOnImage(modelOutput, justDetections)
    cv2.imshow('Detections Feed', justDetections)

# %% [markdown]
# # Processesing tools

# %%
def getLandmarkOutput(frame, landmarkSolution):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    
    # Make detection
    landmarkOutput = landmarkSolution.process(image)
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
        
        
    Enum values for pose_landmarks and pose_world_landmarks are the same
    """

    # Recolor back to BGR, openCV is weird like that
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    return landmarkOutput, image

# %%
def modelOutputToLandmarkDict(modelOutput) -> dict[str,dict[object,object]]:
    
    poseLandmarks = modelOutput.pose_landmarks.landmark if modelOutput.pose_landmarks is not None else None
    poseWorldLandmarks = modelOutput.pose_world_landmarks.landmark if modelOutput.pose_world_landmarks is not None else None
    leftHandLandmarks = modelOutput.left_hand_landmarks.landmark if modelOutput.left_hand_landmarks is not None else None
    rightHandLandmarks = modelOutput.right_hand_landmarks.landmark if modelOutput.right_hand_landmarks is not None else None
    
    poseEnum = mp.solutions.pose.PoseLandmark #enum/name-values of pose landmarks in mediaPipe
    handEnum = mp.solutions.hands.HandLandmark #enum/name-values of hand landmarks in mediaPipe
    
    #tuple for what we want to include in our data
    landmarkTypesWithEnums = (("poseLandmarks", poseLandmarks,poseEnum),("poseWorldLandmarks",poseWorldLandmarks,poseEnum),
                                ("leftHandLandmarks",leftHandLandmarks,handEnum),("rightHandLandmarks",rightHandLandmarks,handEnum))
    
    #logic for turning data we want into a unified customly formated dictionary, instead of an unnamed tuple
    formatedLandmarks:dict[str,dict[object,object]] = dict()
    formatedLandmarkType:dict[object,object] = dict()
    for pair in landmarkTypesWithEnums:
        landmarkTypeName,landmarkType, enum = pair
        
        if landmarkType is not None: # have to check to make sure landmark was defined
            landmarkTypeIsPresent = 1
            formatedLandmarkType.update({'present':landmarkTypeIsPresent})
            
            for enumerate in enum:
                
                #.value is the value of the enum For example it is 1 in the enum NUM = 1
                if landmarkType[enumerate.value] is not None: 
                    landmarkX = landmarkType[enumerate.value].x
                    landmarkY = landmarkType[enumerate.value].y
                    landmarkZ = landmarkType[enumerate.value].z
                    landmarkPresent = 1
                else: # cordinate is normalizes to be in range 0..1 so we can use -1 to signify lack of presence in model
                    landmarkX = -999
                    landmarkY = -999
                    landmarkZ = -999
                    landmarkPresent = 0
                
                #.name is the name of the enum For example it is "NUM" in the enum NUM = 1
                landmark = {enumerate.name:{'x':landmarkX,'y':landmarkY,'z':landmarkZ,'present':landmarkPresent}}
                    
                formatedLandmarkType.update(landmark)
                
        else: #landmark type wasn't found so set all landmarks in type to not present and default
            landmarkTypeIsPresent = 0
            formatedLandmarkType.update({'present':landmarkTypeIsPresent})
            
            for enumerate in enum:
                landmarkX = -999
                landmarkY = -999
                landmarkZ = -999
                landmarkPresent = 0
                
                #.name is the name of the enum For example it is "NUM" in the enum NUM = 1
                landmark = {enumerate.name:{'x':landmarkX,'y':landmarkY,'z':landmarkZ,'present':landmarkPresent}}
                    
                formatedLandmarkType.update(landmark)
                
        formatedLandmarks.update({landmarkTypeName:formatedLandmarkType})
        formatedLandmarkType = dict() #clear dict for next itration
    
    return formatedLandmarks

# %%
def initialize_FrameModelOutput_Data() -> dict[str, list]:
    """
    This Method initializes data dict with correct keys needed for append_FrameModelOutput_Data()
    """
    data:dict = dict()    
    
    poseEnum = mp.solutions.pose.PoseLandmark #enum/name-values of pose landmarks in mediaPipe
    handEnum = mp.solutions.hands.HandLandmark #enum/name-values of hand landmarks in mediaPipe
    
    #tuple for what we want to include in our data
    landmarkTypesWithEnums = (("poseLandmarks",poseEnum),("poseWorldLandmarks",poseEnum),
                                ("leftHandLandmarks",handEnum),("rightHandLandmarks",handEnum))
    
    #define landmarks we don't wan
    enumsFilter:list[str] = ['LEFT_HIP','RIGHT_HIP','LEFT_KNEE','RIGHT_KNEE','LEFT_ANKLE','RIGHT_ANKLE',
                                'LEFT_HEEL','RIGHT_HEEL','LEFT_FOOT_INDEX','RIGHT_FOOT_INDEX']
                
    for pair in landmarkTypesWithEnums:
        landmarkTypeName, enum = pair
        
        data.update({landmarkTypeName + '_present': []}) #1 for present 0 for not present
        
        for enumerate in enum:
            
            if enumerate.name not in enumsFilter:
            
                #.name is the name of the enum For example it is "NUM" in the enum NUM = 1
                data.update({landmarkTypeName + "_" + enumerate.name + '_x': []})
                data.update({landmarkTypeName + "_" + enumerate.name + '_y': []})
                data.update({landmarkTypeName + "_" + enumerate.name + '_z': []})
                data.update({landmarkTypeName + "_" + enumerate.name + '_present': []})
                                        
    return data

# %%
def append_FrameModelOutput_Data(modelOutput, data:dict[str, list]) -> dict[str,list]:  
    """
    This method expects data to be a dictionary initialized by initialize_FrameModelOutput_Data()
    """        
    
    poseLandmarks = modelOutput.pose_landmarks.landmark if modelOutput.pose_landmarks is not None else None
    poseWorldLandmarks = modelOutput.pose_world_landmarks.landmark if modelOutput.pose_world_landmarks is not None else None
    leftHandLandmarks = modelOutput.left_hand_landmarks.landmark if modelOutput.left_hand_landmarks is not None else None
    rightHandLandmarks = modelOutput.right_hand_landmarks.landmark if modelOutput.right_hand_landmarks is not None else None
    
    poseEnum = mp.solutions.pose.PoseLandmark #enum/name-values of pose landmarks in mediaPipe
    handEnum = mp.solutions.hands.HandLandmark #enum/name-values of hand landmarks in mediaPipe
    
    #tuple for what we want to include in our data
    landmarkTypesWithEnums = (("poseLandmarks", poseLandmarks,poseEnum),("poseWorldLandmarks",poseWorldLandmarks,poseEnum),
                                ("leftHandLandmarks",leftHandLandmarks,handEnum),("rightHandLandmarks",rightHandLandmarks,handEnum))
    
    #define landmarks we don't wan
    enumsFilter:list[str] = ['LEFT_HIP','RIGHT_HIP','LEFT_KNEE','RIGHT_KNEE','LEFT_ANKLE','RIGHT_ANKLE',
                                'LEFT_HEEL','RIGHT_HEEL','LEFT_FOOT_INDEX','RIGHT_FOOT_INDEX']
    
    #logic for turning data we want into a unified customly formated dictionary, instead of an unnamed tuple
    for pair in landmarkTypesWithEnums:
        landmarkTypeName,landmarkType, enum = pair
        
        if landmarkType is not None: # have to check to make sure landmark was defined
            key = landmarkTypeName + '_present'
            data.update({key: data[key] + [1]}) #1 for present 0 for not present
            
            for enumerate in enum:
                
                if enumerate.name not in enumsFilter:
                    
                    #.value is the value of the enum For example it is 1 in the enum NUM = 1
                    if landmarkType[enumerate.value] is not None: 
                        landmarkX = landmarkType[enumerate.value].x
                        landmarkY = landmarkType[enumerate.value].y
                        landmarkZ = landmarkType[enumerate.value].z
                        landmarkPresent = 1
                    else: # cordinate is normalizes to be in range 0..1 so we can use -1 to signify lack of presence in model
                        landmarkX = -999
                        landmarkY = -999
                        landmarkZ = -999
                        landmarkPresent = 0
                    
                    #.name is the name of the enum For example it is "NUM" in the enum NUM = 1
                    key = landmarkTypeName + "_" + enumerate.name + '_x'
                    data.update({key: data[key] + [landmarkX]})
                    key = landmarkTypeName + "_" + enumerate.name + '_y'
                    data.update({key: data[key] + [landmarkY]})
                    key = landmarkTypeName + "_" + enumerate.name + '_z'
                    data.update({key: data[key] + [landmarkZ]})
                    key = landmarkTypeName + "_" + enumerate.name + '_present'
                    data.update({key: data[key] + [landmarkPresent]})
                                
        else: #landmark type wasn't found so set all landmarks in type to not present and default
            key = landmarkTypeName + '_present'
            data.update({key: data[key] + [0]}) #1 for present 0 for not present
            
            for enumerate in enum:
                if enumerate.name not in enumsFilter:
                    landmarkX = -999
                    landmarkY = -999
                    landmarkZ = -999
                    landmarkPresent = 0
                    
                    key = landmarkTypeName + "_" + enumerate.name + '_x'
                    data.update({key: data[key] + [landmarkX]})
                    key = landmarkTypeName + "_" + enumerate.name + '_y'
                    data.update({key: data[key] + [landmarkY]})
                    key = landmarkTypeName + "_" + enumerate.name + '_z'
                    data.update({key: data[key] + [landmarkZ]})
                    key = landmarkTypeName + "_" + enumerate.name + '_present'
                    data.update({key: data[key] + [landmarkPresent]})
                                        
    return data

# %%
def old_processVideo(cap, capType='video', showVideo=False, controllable=False)->dict:
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
    

# %%
def old_processVideoFeed(deviceNum=0, showVideo:bool=False):
    cap = cv2.VideoCapture(deviceNum) #0 is for default video
    
    old_processVideo(cap, 'feed', showVideo)
    
    cap.release()
    cv2.destroyAllWindows()
            
    cap.release()

# %%
def processVideoFeed(deviceNum=0, showVideo:bool=True, controllable=True):
    cap = cv2.VideoCapture(deviceNum) #0 is for default video
    
    data:dict = initialize_FrameModelOutput_Data()
    # Setup mediapipe holistic solution instance
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            #get input
            success, frame = cap.read()
            
            
            #error handling
            if not success:
                # If loading a video, use 'break' if using live feed, use 'continue'.                
                continue
            
            
            modelOutput, image = getLandmarkOutput(frame, holistic) 
            # tmp = modelOutputToLandmarkDict(modelOutput) #This line is just to see output laid out in a nicer dict, it's useful for trouble shooting
            append_FrameModelOutput_Data(modelOutput, data)
            
            
            if showVideo:
                showFrame(image,modelOutput)


            ##############
            #Control Logic
            ##############
            if controllable:
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
    
    cap.release()
    cv2.destroyAllWindows()
            
    cap.release()

# %% [markdown]
# # PreProcessing stuff

# %%
def old_processVideoFile(videoPath:str, showVideo:bool=False) -> dict:
    cap = cv2.VideoCapture(videoPath)
    #?can you change the capture rate to be higher(this would help process mp4s faster, instead of the default 30 fps)
    
    data = old_processVideo(cap, 'video',showVideo)
            
    cap.release()
    cv2.destroyAllWindows()
    return data

# %%
def processVideoFile(videoPath:str, showVideo:bool=False, controllable=False) -> dict:
    cap = cv2.VideoCapture(videoPath)
    #?can you change the capture rate to be higher(this would help process mp4s faster, instead of the default 30 fps)
    
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
                break
            
            modelOutput, image = getLandmarkOutput(frame, holistic) 
            # tmp = modelOutputToLandmarkDict(modelOutput) #This line is just to see output laid out in a nicer dict, it's useful for trouble shooting
            append_FrameModelOutput_Data(modelOutput, data)
            
            
            if showVideo:
                showFrame(image,modelOutput)


            ##############
            #Control Logic
            ##############
            if controllable:
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
            
    cap.release()
    cv2.destroyAllWindows()
    return data

# %%
def filterDF(df:pd.DataFrame, filterCSVPath)->pd.DataFrame:
    filterDF = pd.read_csv(filterCSVPath)
    filterList = filterDF['words'].tolist()
    
    regXList = map(lambda x : f"^({x})((0|1|2|3|4|5|6|7|8|9)*)$",filterList) #capture word at the start of string then have wildcard number regex to the end
    regX = str.join('|', regXList) #combine the regular expressions into master regX
    
    df = df[df['Gloss'].str.contains(regX, case=False, na=False)] #use regX to filter Gloss
    
    df = df.reset_index(drop=True)#reset index after filter and drop the old index
    
    return df

# %%
def old_makePreProcessedData(glossCSVPath,videoFolderPath,outputCSVFilePath,filterCSVPath=None):
    df = pd.read_csv(glossCSVPath)
    #df = df.head() #using only head as a proof of concept
    df = df[['Video file','Gloss']] #we only need file name and label
        
    if filterCSVPath is not None:
        df = filterDF(df, filterCSVPath)

    tmp = df.apply(lambda row: processVideoFile(videoFolderPath + row['Video file']), axis=1, result_type='expand')#result_type='expand' unrolls the dictionaries from processVideoFile into a list of data frames, which allows us to join them later so we don't just get one big column, we get several columns
    df = df.join(tmp,how='left') # we need to join the sign names with their respective data

    df.to_csv(outputCSVFilePath)

# %%
def makePreProcessedData(glossCSVPath,videoFolderPath,outputCSVFilePath,filterCSVPath=None, chunkSize = 10, startIndex=0, stopIndex:int|None=None):
    df = pd.read_csv(glossCSVPath)
    #df = df.head() #using only head as a proof of concept
    df = df[['Video file','Gloss']] #we only need file name and label
        
    if filterCSVPath is not None:
        df = filterDF(df, filterCSVPath)

    numRows = len(df)
    
    if stopIndex is None:
        stopIndex = numRows
    
    stopIndex += 1
    for i in range(startIndex, stopIndex, chunkSize):
        if i+chunkSize > stopIndex:
            chunkSize = stopIndex - i
        
        dfChunk = df[i:i+chunkSize].copy(deep=True)#using a deep copy so we don't write to main df and cause exsesive use of memory
        
        # The following block of old code takes up to much memory during code execution
        tmp = dfChunk.apply(lambda row: processVideoFile(videoFolderPath + row['Video file']), axis=1, result_type='expand')#result_type='expand' unrolls the dictionaries from processVideoFile into a list of data frames, which allows us to join them later so we don't just get one big column, we get several columns
        dfChunk = dfChunk.join(tmp,how='left') # we need to join the sign names with their respective data

        # we need to use w for first chunk to clear out old stuff, and header needs to be written also
        if i == 0:
            writeMode = 'w'
            writeHeader = True
        else:
            writeMode = 'a'
            writeHeader = False
            
            
        dfChunk.to_csv(outputCSVFilePath,mode=writeMode, header=writeHeader)
        print(f"Processed videos {i} to {i+chunkSize-1}")
    print(f"finished processing videos to {outputCSVFilePath}")

# %%
if __name__ == '__main__':
    makePreProcessedData('../docs/val_glossary.csv','../src/frontend/recordings/val/',
                    '../src/training_data/christian_data/christian_val.csv', '../docs/Christian_Key_ASL.csv')

# %% [markdown]
# # make processed dataframe for front end

# %%
def makePaddingFlagColumn(row, newLen):
    numFrames = len(row.iloc[0])
    lenDifference = newLen - numFrames
    assert lenDifference >= 0, f'Video longer than {newLen} frames' 
    return ([0]*numFrames) + ([1] * lenDifference)

# %%
def paddToLen(ls, newLen):
    lenDifference = newLen - len(ls)
    assert lenDifference >= 0, f'Video longer than {newLen} frames' 
    return ls + ([-999] * (lenDifference))

# %%
def makePreProcessedDataForFrontEndWithPadding(filePaths:list[str], paddingLen)->pd.DataFrame:
    df = pd.DataFrame({'Video file':filePaths})
    df = df.apply(lambda row: processVideoFile(row['Video file']), axis=1, result_type='expand')#result_type='expand' unrolls the dictionaries from processVideoFile into a list of data frames, which allows us to join them later so we don't just get one big column, we get several columns
    paddingColumn = df.apply(lambda row: makePaddingFlagColumn(row, paddingLen),axis=1)
    df = df.map(lambda cell: paddToLen(cell, paddingLen))
    df.insert(0,'padding',paddingColumn)
    return df

# %%
def addingPaddingToPreProcessedData(df:pd.DataFrame, paddingLen:int)->pd.DataFrame:
    paddingColumn = df.apply(lambda row: makePaddingFlagColumn(row, paddingLen),axis=1)
    df = df.map(lambda cell: paddToLen(cell, paddingLen))
    df.insert(0,'padding',paddingColumn)
    return df


