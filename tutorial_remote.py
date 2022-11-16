import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import os
import time
import av


class AusarVision():
    
    def __init__(self, isStatic=False, modelComplexity=1, minDetectionConfidence=0.7, minTrackingConfidence=0.7):
        """
        Constructor for AusarVision object.
        Args:
            isStatic: whether the object should recognize static images (True) or video sequnces (False). Default: False.
            modelComplexity: complexity of landmark detection model. The higher the value, the better the quality,
                             but also the more time it takes to perform detection. Values: 0, 1, 2. Default: 1.
            minDetectionConfidence: check out min_detection_confidence for mediapipe.solutions.pose.Pose(args).
            minTrackingConfidence: check out min_tracking_confidence for mediapipe.solutions.pose.Pose(args).
        Returns:
            None   
        """
        
        # Pose detection tools initialization.
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=isStatic,
                                 model_complexity=modelComplexity, 
                                 min_detection_confidence=minDetectionConfidence, 
                                 min_tracking_confidence=minTrackingConfidence)
        
        # Detailed hands detection tools initialization.
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=isStatic, 
                                    max_num_hands=2, 
                                    min_detection_confidence=minDetectionConfidence)
        # ADD DOCUMENTTATION!!!!!!!!!!!!!!!!!
        self.trainParams = {
            'fps': 24,
            'bgImagePath':'white.png',
            'fullscreenWidth': 1920,
            'fullscreenHeight': 1000,
            'framesToRestMode': 30,
            'restRadius': 0.07,
            'restXCenter': 0.2,
            'restYCenter': 0.3,
            'framesToExit': 30,
            'exitRadius': 0.07,
            'exitXCenter': 0.8,
            'exitYCenter': 0.3,
            'modes': {0:'Practice', 1:'Rest'},
            'mode': 0,
            'fileName': '', 
            'fileDir': 'exercises',
            'filePath': '',
            'poses': 0,
            'weights': 0,
            'tutorVideoPath': '',
            'angleFileName': '',
            'angleFileDir': 'exercises',
            'angleFilePath': '',
            'angleMatrix': 0,
            'angleWeights': 0,
            'curIndex': 0,
            'curPose': 0,
            'timeStart': 0,
            'repCnt': 0,
            'flag_start': 0,
            'videoPath': '',
            'outputVideoPath': '',
            'videoUnderSkeleton': True,
            'displaySkeleton': True,
            'debugMode': False,
            'camera_video': 0,
            'tutor_video': 0,
            'video_writer': 0,
            'timeFrameTemplate': 0,
            'timeFrame': 0,
            'tutorFrame': 0
        }
        return

    
    def basePose(self, image):
        """
        This function performs pose detection on a frame from the video/single image.

        Args:
            image: Copy of the input image with a prominent person whose pose landmarks needs to be detected. 
                  (default: cv2.imread(path))

        Returns:
            image: Copy of the input image with the detected pose landmarks drawn.
            landmarks: A list of detected landmarks converted into their original scale.
        """
    
        # Set the function name for error report.
        functionName = 'detectPose'
        # Convert the image from BGR into RGB format.
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Perform the Pose Detection.
        results = self.pose.process(imageRGB)
        # Retrieve the height and width of the input image.
        height, width, _ = image.shape
        # Initialize a list to store the detected landmarks.
        landmarks = []
        # Set the pose_landmarks value. 
        pose_landmarks = results.pose_landmarks 
        if results.pose_landmarks:
            # Iterate over the detected landmarks.
            for landmark in results.pose_landmarks.landmark:
                # Append the landmark into the list.
                landmarks.append([round(landmark.x, 3), round(landmark.y, 3)])
        # Return the output image and the found landmarks.
        return image, landmarks

    
    def calculateAngle(self, landmark1, landmark2, landmark3):
        """
        This function calculates angle between three different landmarks.
        Args:
            landmark1: The first landmark containing the x,y and z coordinates.
            landmark2: The second landmark containing the x,y and z coordinates.
            landmark3: The third landmark containing the x,y and z coordinates.
        Returns:
            angle: The calculated angle (in radian) between the three landmarks.

        """
        
        # Set the 'too small' value.
        eps = 0.00000001
        # Get the required landmarks coordinates.
        x1, y1 = landmark1
        x2, y2 = landmark2
        x3, y3 = landmark3
        # Calculate the difference between landmark1 and landmark2.
        diff12 = np.array([x1-x2, y1-y2])
        # Calculate the difference between landmark3 and landmark2.
        diff32 = np.array([x3-x2, y3-y2])
        # Check if diff vectors are too small.
        if np.linalg.norm(diff12) * np.linalg.norm(diff32) < eps:
            # Set the angle value to zero.
            angle = 0
        # Otherwise, all vectors are not equal to zero.
        else:
            # Calculate the scalar product of diff12 and diff32.
            scp = np.sum(diff12 * diff32)
            # Calculate the cosinus between diff12 and diff32.
            cosinus = scp / (np.linalg.norm(diff12) * np.linalg.norm(diff32))
            # Check if cosinus is > 1 due to computer calculation error.
            if cosinus > 1:
                # Set the angle to 0.
                angle = 0
            # Check if cosinus is < -1 due to computer calculation error.
            elif cosinus < -1:
                # Set the angle to 180 degrees.
                angle = np.pi
            # In any other case perform default calculation.
            else:
                # Calculate the angle in degrees between the three points.
                angle = round(np.arccos(cosinus), 3)
        # Return the calculated angle.
        return angle
    
    
    def calculateAngleBetweenVectors(self, landmark1, landmark2, landmark3, landmark4):
        """
        This function calculates angle between vectors (landmark1, landmark2) and (landmark3, landmark4)
        """
        
        v1 = np.array(landmark2) - np.array(landmark1)
        landmark_1 = np.array(landmark3) + v1
        return self.calculateAngle(landmark_1, landmark3, landmark4)
    
    
    def getAngles(self, landmarks):
        """
        This function calculates all angles between landmarks in the frame.
        Args:
            landmarks: Detected landmarks of the person in the image.
        Returns:
            angles: numpy.array of 12 calculated angles. Their numbers are defined below.
                angles[0]: left_elbow_angle,
                angles[1]: right_elbow_angle,
                angles[2]: left_shoulder_angle,
                angles[3]: right_shoulder_angle,
                angles[4]: left_hip_angle,
                angles[5]: right_hip_angle,
                angles[6]: left_knee_angle,
                angles[7]: right_knee_angle,
                angles[8]: left_tree_angle,
                angles[9]: right_tree_angle

        """

        # Set the function name for error report.
        functionName = 'getAngles'

        # Initialize the angles array.
        angles = np.zeros(12)

        # Calculate the required angles.
        #----------------------------------------------------------------------------------------------------------------

        # Get the angle between the left shoulder, elbow and wrist points. 
        angles[0] = self.calculateAngle(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                          landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                          landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value])

        # Get the angle between the right shoulder, elbow and wrist points. 
        angles[1] = self.calculateAngle(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                           landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                           landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value])

        # Get the angle between the left elbow, shoulder and hip points. 
        angles[2] = self.calculateAngle(landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                             landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                             landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value])

        # Get the angle between the right hip, shoulder and elbow points. 
        angles[3] = self.calculateAngle(landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value],
                                              landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                              landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value])

        # Get the angle between the left shoulder, hip, and knee points. 
        angles[4] = self.calculateAngle(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                         landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value],
                                         landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value])

        # Get the angle between the right shoulder, hip, and knee points 
        angles[5] = self.calculateAngle(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                          landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value])

        # Get the angle between the left hip, knee and ankle points. 
        angles[6] = self.calculateAngle(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value],
                                         landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value],
                                         landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value])

        # Get the angle between the right hip, knee and ankle points 
        angles[7] = self.calculateAngle(landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                          landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value])

        # Get the angle between the left knee, left hip, and right hip points. 
        angles[8] = self.calculateAngle(landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value],
                                         landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value],
                                         landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value])

        # Get the angle between the right knee, right hip, and left hip points.  
        angles[9] = self.calculateAngle(landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                          landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value])

        leftWrist = np.array(landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value])
        rightWrist = np.array(landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value])
        leftElbow = np.array(landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value])
        rightElbow = np.array(landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value])
        leftShoulder = np.array(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value])
        rightShoulder = np.array(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value])

        angles[10] = self.calculateAngleBetweenVectors(leftElbow, leftWrist, rightElbow, rightWrist)
        angles[11] = self.calculateAngleBetweenVectors(leftShoulder, leftElbow, rightShoulder, rightElbow)

        # Return required values.
        return angles
    
    
    def detect(self, image, mode='pose'):
        """
        This function performs pose detection and extends landmark array, adding manually calculatetd landmarks to it.
        Args:
            image: RGB image (default way to obtain: cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)).
            mode: type of found landmarks. Currently only 'pose' mode is implemented.
        Returns:
            image: Copy of the input image with the detected pose landmarks drawn.
            landmarks: 2D np.ndarray object with extended landmarks.
        """
        
        if mode == 'pose':
            image, landmarks = self.basePose(image)
            if landmarks == []:
                return image, None
            landmarks = np.array(landmarks)
            nose = landmarks[0]
#             leftEyeInner = landmarks[1]
            leftEye = landmarks[2]
#             leftEyeOuter = landmarks[3]
#             rightEyeInner = landmarks[4]
            rightEye = landmarks[5]
#             rightEyeOuter = landmarks[6]
            leftEar = landmarks[7]
            rightEar = landmarks[8]
            mouthLeft = landmarks[9]
            mouthRight = landmarks[10]
            leftShoulder = landmarks[11]
            rightShoulder = landmarks[12]
#             leftElbow = landmarks[13]
#             rightElbow = landmarks[14]
#             leftWrist = landmarks[15]
#             rightWrist = landmarks[16]
#             leftPinky = landmarks[17]
#             rightPinky = landmarks[18]
#             leftIndex = landmarks[19]
#             rightIndex = landmarks[20]
#             leftThumb = landmarks[21]
#             rightThumb = landmarks[22]
#             leftHip = landmarks[23]
#             rightHip = landmarks[24]
#             leftKnee = landmarks[25]
#             rightKnee = landmarks[26]
#             leftAnkle = landmarks[27]
#             rightAnkle = landmarks[28]
#             leftHeel = landmarks[29]
#             rightHeel = landmarks[30]
#             leftFootIndex = landmarks[31]
#             rightFootIndex = landmarks[32]
            leftThirdShoulder = leftShoulder * 2/3 + rightShoulder * 1/3
            rightThirdShoulder = leftShoulder * 1/3 + rightShoulder * 2/3
            middleShoulder = (leftShoulder + rightShoulder) / 2
            leftNeck = (leftThirdShoulder + leftEar) / 2
            rightNeck = (rightThirdShoulder + rightEar) / 2
            neck = (leftNeck + rightNeck) / 2
            middleEye = (leftEye + rightEye) / 2
            mouthMiddle = (mouthLeft + mouthRight) / 2
            chin = mouthMiddle + (mouthMiddle - nose)
            tophead = middleEye + 1.3 * (middleEye - mouthMiddle)
            landmarks = np.round(np.append(landmarks, [leftThirdShoulder, rightThirdShoulder, middleShoulder, 
                                                       leftNeck, rightNeck, neck, middleEye, mouthMiddle, 
                                                       chin, tophead]).reshape(-1, 2), 3)
            
            return image, landmarks
    
    
    def drawLandmarks(self, image, landmarks, color=(0,0,255), radius=3):
        """
        This function draws landmarks on the image given.
        Args:
            image: RGB image (default way to obtain: cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)).
            landmarks: iterable object. Contains copy of2D coordinates of landmarks. Both coordinates have float values from 0 to 1,
                       where (0,0) represents the top-left corner and (1,1) represents the bottom-right corner.
            color: RGB color to draw landmarks. Default: blue.
            radius: radius of landmark circles on the image. Default: 3
        Returns:
            image: RGB image with landmarks on it.
        """
        
        # Get the BGR version of the image.
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # Get the image size.
        height, width, _ = image.shape
        landmarks[:, 0] *= width
        landmarks[:, 1] *= height
        landmarks = landmarks.astype(int)
        for landmark in landmarks:
            # Draw a landmark circle on the image.
            cv2.circle(image, landmark, radius, color, thickness=-1)
        # Transform image back to RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    
    def drawConnections(self, image, landmarks, mode='pose', circleColor=(0,0,255), radius=5, 
                        lineColor=(255,255,255), lineThickness=3):
        """
        This fucntion draws landmarks with connections between them, according to the mode selected.
        Args:
            image: RGB image (default way to obtain: cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)).
            landmarks: iterable object. Contains copy of 2D coordinates of landmarks. Both coordinates have float values from 0 to 1,
                       where (0,0) represents the top-left corner and (1,1) represents the bottom-right corner.
            mode: the connection type mode. Currently only 'pose' is implemented.
            circleColor: landmark circle BGR color. Default: blue.
            radius: landmark circle radius. Default: 5.
            lineColor: connection line RGB color. Default: white.
            lineThickness: connection line thickness. Default: 3.
        Returns:
            image: RGB image with connected landmarks on it.
        """
        
#         # Get the BGR version of the image.
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # Get the image size.
        height, width, _ = image.shape
        landmarks = np.array(landmarks)
        for landmark in landmarks:
            # Transform float values into pixels, using image size.
            landmark[0] = int(landmark[0] * width)
            landmark[1] = int(landmark[1] * height)
            # Draw a landmark circle on the image.
            cv2.circle(image, landmark, radius, circleColor, thickness=-1)
        
        if mode == 'pose':
            list1 = list(range(11, 19))
            list2 = list(range(23, 31))
            list3 = list(range(1, 3))
            list4 = list(range(3, 6))
            list5 = [11, 23]
            list1.extend(list2)
            list3.extend(list4)
            list3.extend(list5)
            for i in list1:
                cv2.line(image, landmarks[i], landmarks[i+2], lineColor, thickness=lineThickness)
            for i in list3:
                cv2.line(image, landmarks[i], landmarks[i+1], lineColor, thickness=lineThickness)
            cv2.line(image, landmarks[15], landmarks[19], lineColor, thickness=lineThickness)
            cv2.line(image, landmarks[15], landmarks[21], lineColor, thickness=lineThickness)
            cv2.line(image, landmarks[16], landmarks[20], lineColor, thickness=lineThickness)
            cv2.line(image, landmarks[16], landmarks[22], lineColor, thickness=lineThickness)
            cv2.line(image, landmarks[27], landmarks[31], lineColor, thickness=lineThickness)
            cv2.line(image, landmarks[28], landmarks[32], lineColor, thickness=lineThickness)
            cv2.line(image, landmarks[11], landmarks[23], lineColor, thickness=lineThickness)
            cv2.line(image, landmarks[12], landmarks[24], lineColor, thickness=lineThickness)
            cv2.line(image, landmarks[0], landmarks[1], lineColor, thickness=lineThickness)
            cv2.line(image, landmarks[0], landmarks[4], lineColor, thickness=lineThickness)
            cv2.line(image, landmarks[3], landmarks[7], lineColor, thickness=lineThickness)
            cv2.line(image, landmarks[6], landmarks[8], lineColor, thickness=lineThickness)
            
#             tophead = landmarks[42]
#             chin = landmarks[41]
#             nose = landmarks[0]
            
#             centerCoordinates = nose
#             axesLength = np.array([np.linalg.norm(tophead - nose), np.linalg.norm(leftEar-rightEar)/2]).astype(int)
#             centerCoordinates = ((tophead + chin) / 2).astype(int)
#             axesLength = np.array([np.linalg.norm(tophead - chin)/2, np.linalg.norm(tophead - chin)/(1+np.sqrt(5))]).astype(int)
#             mainAxis = tophead - nose
#             angle = np.arctan2(mainAxis[1], mainAxis[0]) * 180 / np.pi
#             startAngle = 0
#             endAngle = 360
                
#             cv2.ellipse(image, centerCoordinates, axesLength, angle, startAngle, endAngle, lineColor, lineThickness)  
            
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    
    def getVectors(self, landmarks, mode='pose'):
        """
        This function calculates vectors between the connected landmarks, according to mode.
        Then the function forms an array of vectors between connected landmarks.
        Args:
            landmarks: iterable object, which contains 2D coordinates of landmarks.
            mode: landmark connection mode. Currently only 'pose' is implemented.
        Returns:
            vectors: 2D np.ndarray object with vectors between connected landmarks.
        """
        
        if mode == 'pose':
            size = 39
            vectors = np.zeros([size, 2])
            if landmarks is None:
                return None
            landmarks = np.array(landmarks)
            nose = landmarks[0]
            leftEyeInner = landmarks[1]
            leftEye = landmarks[2]
            leftEyeOuter = landmarks[3]
            rightEyeInner = landmarks[4]
            rightEye = landmarks[5]
            rightEyeOuter = landmarks[6]
            leftEar = landmarks[7]
            rightEar = landmarks[8]
            mouthLeft = landmarks[9]
            mouthRight = landmarks[10]
            leftShoulder = landmarks[11]
            rightShoulder = landmarks[12]
            leftElbow = landmarks[13]
            rightElbow = landmarks[14]
            leftWrist = landmarks[15]
            rightWrist = landmarks[16]
            leftPinky = landmarks[17]
            rightPinky = landmarks[18]
            leftIndex = landmarks[19]
            rightIndex = landmarks[20]
            leftThumb = landmarks[21]
            rightThumb = landmarks[22]
            leftHip = landmarks[23]
            rightHip = landmarks[24]
            leftKnee = landmarks[25]
            rightKnee = landmarks[26]
            leftAnkle = landmarks[27]
            rightAnkle = landmarks[28]
            leftHeel = landmarks[29]
            rightHeel = landmarks[30]
            leftFootIndex = landmarks[31]
            rightFootIndex = landmarks[32]
            leftThirdShoulder = landmarks[33]
            rightThirdShoulder = landmarks[34]
            middleShoulder = landmarks[35]
            leftNeck = landmarks[36]
            rightNeck = landmarks[37]
            neck = landmarks[38]
            middleEye = landmarks[39]
            mouthMiddle = landmarks[40]
            chin = landmarks[41]
            tophead = landmarks[42]
            

            vectors[0] = leftShoulder - rightShoulder
            vectors[1] = leftElbow - leftShoulder
            vectors[2] = rightElbow - rightShoulder
            vectors[3] = leftWrist - leftElbow
            vectors[4] = rightWrist - rightElbow
            vectors[5] = leftPinky - leftWrist
            vectors[6] = rightPinky - rightWrist
            vectors[7] = leftIndex - leftPinky
            vectors[8] = rightIndex - rightPinky
            vectors[9] = leftWrist - leftIndex
            vectors[10] = rightWrist - rightIndex
            vectors[11] = leftThumb - leftWrist
            vectors[12] = rightThumb - rightWrist
            
            vectors[13] = leftHip - leftShoulder
            vectors[14] = rightHip - rightShoulder
            vectors[15] = leftHip - rightHip
            vectors[16] = leftKnee - leftHip
            vectors[17] = rightKnee - rightHip
            vectors[18] = leftAnkle - leftKnee
            vectors[19] = rightAnkle - rightKnee
            vectors[20] = leftHeel - leftAnkle
            vectors[21] = rightHeel - rightAnkle
            vectors[22] = leftFootIndex - leftHeel
            vectors[23] = rightFootIndex - rightHeel
            vectors[24] = leftAnkle - leftFootIndex
            vectors[25] = rightAnkle - rightFootIndex
            
            vectors[26] = leftEyeInner - nose
            vectors[27] = rightEyeInner - nose
            vectors[28] = leftEye - leftEyeInner
            vectors[29] = rightEye - rightEyeInner
            vectors[30] = leftEyeOuter - leftEye
            vectors[31] = rightEyeOuter - rightEye
            vectors[32] = leftEar - leftEyeOuter
            vectors[33] = rightEar - rightEyeOuter
            vectors[34] = mouthLeft - mouthRight
            
            vectors[35] = leftEar - leftShoulder
            vectors[36] = rightEar - rightShoulder
            vectors[37] = mouthLeft - nose
            vectors[38] = mouthRight - nose
            
        return vectors
    
    
    def drawFromVectors(self, image, landmarks, mode='pose', k=1, kx=1, ky=1, kmatrix=None, lsh = None,
                        circleColor=(255,0,0), radius=3, 
                        drawLines=True, lineColor=(255,255,255), lineThickness=3):
        """
        This function draws a human skeleton from vectors obtained from getVectors function.
        The drawing mode must be exactly the same as in getVectors.
        Args:
            image: RGB image to draw landmaks in.
            landmarks: iterable object, which contains copy of 2D coordinates of landmarks.
            mode: landmark connection mode. Currently only 'pose' is implemented.
            k: stretching coefficient for ALL landmark vectors. Default: 1.
            kx: x-axis stretching coefficient for ALL landmark vectors. Default:1.
            ky: y-axis stretching coefficient for ALL landmark vectors. Default:1.
            kmatrix: column matrix to apply strecthcing vectorwise. If None, no additional stretching applied. Default: None.
            lsh: left shoulder coordinates to syncronize 2 persons. If None, it will be retrieved from landmarks.
            circleColor: landmark circle BGR color. Default: blue.
            radius: landmark circle radius. Default: 3.
            drawLines: whether to draw skeleton lines or not. Default: True.
            lineColor: connection line RGB color. Default: white.
            lineThickness: connection line thickness. Default: 3.
        Returns:
            image: RGB image with connected landmarks on it.
        """
        
        # Obtain vectors form getVectors.
        vectors = self.getVectors(landmarks, mode)
        if vectors is None:
            return image, None
        
        if mode == 'pose':
            height, width, _ = image.shape
            landmarks = np.array(landmarks)
            if lsh is None:
                leftShoulder = landmarks[11]
            else:
                leftShoulder = lsh
            leftShoulder = (leftShoulder * np.array([width, height])).astype(int)
            vectors *= k
            vectors[:, 0] *= width * kx
            vectors[:, 1] *= height * ky
            if kmatrix is not None:
                vectors *= kmatrix.reshape(-1, 1)
            vectors = vectors.astype(int)
            
            rightShoulder = leftShoulder - vectors[0]
            middleShoulder = ((leftShoulder + rightShoulder) / 2).astype(int)
            leftThirdShoulder = (leftShoulder * 2/3 + rightShoulder * 1/3).astype(int)
            rightThirdShoulder = (leftShoulder * 1/3 + rightShoulder * 2/3).astype(int)
            leftElbow = leftShoulder + vectors[1]
            rightElbow = rightShoulder + vectors[2]
            leftWrist = leftElbow + vectors[3]
            rightWrist = rightElbow + vectors[4]
            leftPinky = leftWrist + vectors[5]
            rightPinky = rightWrist + vectors[6]
            leftIndex = leftPinky + vectors[7]
            rightIndex = rightPinky + vectors[8]
            leftThumb = leftWrist + vectors[11]
            rightThumb = rightWrist + vectors[12]
            
            leftHip = leftShoulder + vectors[13]
            rightHip = rightShoulder + vectors[14]
            leftKnee = leftHip + vectors[16]
            rightKnee = rightHip + vectors[17]
            leftAnkle = leftKnee + vectors[18]
            rightAnkle = rightKnee + vectors[19]
            leftHeel = leftAnkle + vectors[20]
            rightHeel = rightAnkle + vectors[21]
            leftFootIndex = leftHeel + vectors[22]
            rightFootIndex = rightHeel + vectors[23]
            
            leftEar = leftShoulder + vectors[35]
            rightEar = rightShoulder + vectors[36]
            leftEyeOuter = leftEar - vectors[32]
            rightEyeOuter = rightEar - vectors[33]
            leftEye = leftEyeOuter - vectors[30]
            rightEye = rightEyeOuter - vectors[31]
            middleEye = ((leftEye + rightEye) / 2).astype(int)
            leftEyeInner = leftEye - vectors[28]
            rightEyeInner = rightEye - vectors[29]
            nose = leftEyeInner - vectors[26]
            mouthLeft = nose + vectors[37]
            mouthRight = nose + vectors[38]
            mouthMiddle = ((mouthLeft + mouthRight) / 2).astype(int)
            chin = mouthMiddle + (mouthMiddle - nose)
            tophead = (middleEye + 1.3 * (middleEye - mouthMiddle)).astype(int)
            leftNeck = ((leftThirdShoulder + leftEar) / 2).astype(int)
            rightNeck = ((rightThirdShoulder + rightEar) / 2).astype(int)
            neck = ((leftNeck + rightNeck) / 2).astype(int)
            
            leftNeckEnd = ((leftNeck + leftEar) / 2).astype(int)
            rightNeckEnd = ((rightNeck + rightEar) / 2).astype(int)
            
            if drawLines:
#                 cv2.line(image, leftShoulder, rightShoulder, lineColor, thickness=lineThickness)
                cv2.line(image, leftNeck, leftNeckEnd, lineColor, thickness=lineThickness)
                cv2.line(image, rightNeck, rightNeckEnd, lineColor, thickness=lineThickness)
                
                cv2.line(image, leftShoulder, leftThirdShoulder, lineColor, thickness=lineThickness)
                cv2.line(image, rightShoulder, rightThirdShoulder, lineColor, thickness=lineThickness)
                cv2.line(image, leftShoulder, leftElbow, lineColor, thickness=lineThickness)
                cv2.line(image, rightShoulder, rightElbow, lineColor, thickness=lineThickness)
                cv2.line(image, leftElbow, leftWrist, lineColor, thickness=lineThickness)
                cv2.line(image, rightElbow, rightWrist, lineColor, thickness=lineThickness)
                cv2.line(image, leftWrist, leftPinky, lineColor, thickness=lineThickness)
                cv2.line(image, rightWrist, rightPinky, lineColor, thickness=lineThickness)
                cv2.line(image, leftPinky, leftIndex, lineColor, thickness=lineThickness)
                cv2.line(image, rightPinky, rightIndex, lineColor, thickness=lineThickness)
                cv2.line(image, leftIndex, leftWrist, lineColor, thickness=lineThickness)
                cv2.line(image, rightIndex, rightWrist, lineColor, thickness=lineThickness)
                cv2.line(image, leftWrist, leftThumb, lineColor, thickness=lineThickness)
                cv2.line(image, rightWrist, rightThumb, lineColor, thickness=lineThickness)

                cv2.line(image, leftShoulder, leftHip, lineColor, thickness=lineThickness)
                cv2.line(image, rightShoulder, rightHip, lineColor, thickness=lineThickness)
                cv2.line(image, leftHip, rightHip, lineColor, thickness=lineThickness)
                cv2.line(image, leftHip, leftKnee, lineColor, thickness=lineThickness)
                cv2.line(image, rightHip, rightKnee, lineColor, thickness=lineThickness)
                cv2.line(image, leftKnee, leftAnkle, lineColor, thickness=lineThickness)
                cv2.line(image, rightKnee, rightAnkle, lineColor, thickness=lineThickness)
                cv2.line(image, leftAnkle, leftHeel, lineColor, thickness=lineThickness)
                cv2.line(image, rightAnkle, rightHeel, lineColor, thickness=lineThickness)
                cv2.line(image, leftHeel, leftFootIndex, lineColor, thickness=lineThickness)
                cv2.line(image, rightHeel, rightFootIndex, lineColor, thickness=lineThickness)
                cv2.line(image, leftFootIndex, leftAnkle, lineColor, thickness=lineThickness)
                cv2.line(image, rightFootIndex, rightAnkle, lineColor, thickness=lineThickness)

#                 centerCoordinates = nose
#                 axesLength = np.array([np.linalg.norm(tophead - nose), np.linalg.norm(leftEar-rightEar)/2]).astype(int)
                centerCoordinates = ((tophead + neck) / 2).astype(int)
                axesLength = np.array([np.linalg.norm(tophead - neck)/2, np.linalg.norm(tophead - neck)/(1+np.sqrt(5))]).astype(int)
                mainAxis = tophead - neck
                angle = np.arctan2(mainAxis[1], mainAxis[0]) * 180 / np.pi
                startAngle = 0
                endAngle = 360
                
                cv2.ellipse(image, centerCoordinates, axesLength, angle, startAngle, endAngle, lineColor, lineThickness)  
#                 cv2.line(image, leftEar, leftEyeOuter, lineColor, thickness=lineThickness)
#                 cv2.line(image, rightEar, rightEyeOuter, lineColor, thickness=lineThickness)
                cv2.line(image, leftEyeOuter, leftEye, lineColor, thickness=lineThickness)
                cv2.line(image, rightEyeOuter, rightEye, lineColor, thickness=lineThickness)
                cv2.line(image, leftEye, leftEyeInner, lineColor, thickness=lineThickness)
                cv2.line(image, rightEye, rightEyeInner, lineColor, thickness=lineThickness)
#                 cv2.line(image, leftEye, middleEye, lineColor, thickness=lineThickness)
#                 cv2.line(image, rightEye, middleEye, lineColor, thickness=lineThickness)
#                 cv2.line(image, leftEyeInner, nose, lineColor, thickness=lineThickness)
#                 cv2.line(image, rightEyeInner, nose, lineColor, thickness=lineThickness)
#                 cv2.line(image, nose, mouthLeft, lineColor, thickness=lineThickness)
#                 cv2.line(image, nose, mouthRight, lineColor, thickness=lineThickness)
                cv2.line(image, mouthLeft, mouthRight, lineColor, thickness=lineThickness)
                cv2.line(image, leftThirdShoulder, leftNeck, lineColor, thickness=lineThickness)
                cv2.line(image, rightThirdShoulder, rightNeck, lineColor, thickness=lineThickness)
#                 cv2.line(image, leftNeck, leftEar, lineColor, thickness=lineThickness)
#                 cv2.line(image, rightNeck, rightEar, lineColor, thickness=lineThickness)
#                 cv2.line(image, leftNeck, neck, lineColor, thickness=lineThickness)
#                 cv2.line(image, rightNeck, neck, lineColor, thickness=lineThickness)
#                 cv2.line(image, leftEar, tophead, lineColor, thickness=lineThickness)
#                 cv2.line(image, rightEar, tophead, lineColor, thickness=lineThickness)
            
            cv2.circle(image, leftShoulder, radius, circleColor, thickness=-1)
            cv2.circle(image, rightShoulder, radius, circleColor, thickness=-1)
            cv2.circle(image, leftThirdShoulder, radius, circleColor, thickness=-1)
            cv2.circle(image, rightThirdShoulder, radius, circleColor, thickness=-1)
            cv2.circle(image, middleShoulder, radius, circleColor, thickness=-1)
            cv2.circle(image, leftElbow, radius, circleColor, thickness=-1)
            cv2.circle(image, rightElbow, radius, circleColor, thickness=-1)
            cv2.circle(image, leftWrist, radius, circleColor, thickness=-1)
            cv2.circle(image, rightWrist, radius, circleColor, thickness=-1)
            cv2.circle(image, leftPinky, radius, circleColor, thickness=-1)
            cv2.circle(image, rightPinky, radius, circleColor, thickness=-1)
            cv2.circle(image, leftIndex, radius, circleColor, thickness=-1)
            cv2.circle(image, rightIndex, radius, circleColor, thickness=-1)
            cv2.circle(image, leftThumb, radius, circleColor, thickness=-1)
            cv2.circle(image, rightThumb, radius, circleColor, thickness=-1)
            
            cv2.circle(image, leftHip, radius, circleColor, thickness=-1)
            cv2.circle(image, rightHip, radius, circleColor, thickness=-1)
            cv2.circle(image, leftKnee, radius, circleColor, thickness=-1)
            cv2.circle(image, rightKnee, radius, circleColor, thickness=-1)
            cv2.circle(image, leftAnkle, radius, circleColor, thickness=-1)
            cv2.circle(image, rightAnkle, radius, circleColor, thickness=-1)
            cv2.circle(image, leftHeel, radius, circleColor, thickness=-1)
            cv2.circle(image, rightHeel, radius, circleColor, thickness=-1)
            cv2.circle(image, leftFootIndex, radius, circleColor, thickness=-1)
            cv2.circle(image, rightFootIndex, radius, circleColor, thickness=-1)
            
            cv2.circle(image, leftEar, radius, circleColor, thickness=-1)
            cv2.circle(image, rightEar, radius, circleColor, thickness=-1)
            cv2.circle(image, leftEyeOuter, radius, circleColor, thickness=-1)
            cv2.circle(image, rightEyeOuter, radius, circleColor, thickness=-1)
            cv2.circle(image, leftEye, radius, circleColor, thickness=-1)
            cv2.circle(image, rightEye, radius, circleColor, thickness=-1)
            cv2.circle(image, middleEye, radius, circleColor, thickness=-1)
            cv2.circle(image, leftEyeInner, radius, circleColor, thickness=-1)
            cv2.circle(image, rightEyeInner, radius, circleColor, thickness=-1)
            cv2.circle(image, mouthLeft, radius, circleColor, thickness=-1)
            cv2.circle(image, mouthRight, radius, circleColor, thickness=-1)
            cv2.circle(image, nose, radius, circleColor, thickness=-1)
            cv2.circle(image, leftNeck, radius, circleColor, thickness=-1)
            cv2.circle(image, rightNeck, radius, circleColor, thickness=-1)
            cv2.circle(image, neck, radius, circleColor, thickness=-1)
            cv2.circle(image, tophead, radius, circleColor, thickness=-1)
            cv2.circle(image, chin, radius, circleColor, thickness=-1)
            
            landmarks = np.zeros([43, 2])
            landmarks[42] = tophead
            landmarks[41] = chin
            landmarks[40] = mouthMiddle
            landmarks[39] = middleEye
            landmarks[38] = neck
            landmarks[37] = rightNeck
            landmarks[36] = leftNeck
            landmarks[35] = middleShoulder
            landmarks[34] = rightThirdShoulder
            landmarks[33] = leftThirdShoulder
            landmarks[32] = rightFootIndex
            landmarks[31] = leftFootIndex
            landmarks[30] = rightHeel
            landmarks[29] = leftHeel
            landmarks[28] = rightAnkle
            landmarks[27] = leftAnkle
            landmarks[26] = rightKnee
            landmarks[25] = leftKnee
            landmarks[24] = rightHip
            landmarks[23] = leftHip
            landmarks[22] = rightThumb
            landmarks[21] = leftThumb
            landmarks[20] = rightIndex
            landmarks[19] = leftIndex
            landmarks[18] = rightPinky
            landmarks[17] = leftPinky
            landmarks[16] = rightWrist
            landmarks[15] = leftWrist
            landmarks[14] = rightElbow
            landmarks[13] = leftElbow
            landmarks[12] = rightShoulder
            landmarks[11] = leftShoulder
            landmarks[10] = mouthRight
            landmarks[9] = mouthLeft
            landmarks[8] = rightEar
            landmarks[7] = leftEar
            landmarks[6] = rightEyeOuter
            landmarks[5] = rightEye
            landmarks[4] = rightEyeInner
            landmarks[3] = leftEyeOuter
            landmarks[2] = leftEye
            landmarks[1] = leftEyeInner
            landmarks[0] = nose
            
            landmarks[:, 0] /= width
            landmarks[:, 1] /= height
        return image, landmarks
    
    
    def fit(self, exerciseName, videoPath=0, frameInterval=0):
        fileName = exerciseName + os.extsep + 'csv'
        angleFileName = exerciseName + '_angles' + os.extsep + 'csv'
        fileDir = 'exercises'
        filePath = os.path.join(fileDir, fileName)
        angleFilePath = os.path.join(fileDir, angleFileName)
        flag_newfile = 1
#         if fileName in os.listdir(fileDir):
#             flag_newfile = 0
        file = open(filePath, 'w')
        angleFile = open(angleFilePath, 'w')
        currentTime = 0
        previousTime = 0
        currentFrame = 0
        previousFrame = 0
        # Initialize the VideoCapture object to read from the webcam/stored video.
        camera_video = cv2.VideoCapture(videoPath)
        # Iterate until the webcam is accessed successfully.
        while camera_video.isOpened():
            # Read a frame.
            ok, frame = camera_video.read()
            # Check if frame is not read properly.
            if not ok:
                # Breal the loop.
                break
            # Perform Pose landmark detection.
            frame, landmarks = self.detect(frame)
            if landmarks is not None:
                currentFrame += 1
                if previousFrame and currentFrame - previousFrame > frameInterval:
                    previousFrame = currentFrame
                    angles = self.getAngles(landmarks)
                    string = ','.join(landmarks.ravel().astype(str)) + '\n'
                    angleString = ','.join(angles.ravel().astype(str)) + '\n'
                    file.write(string)
                    angleFile.write(angleString)
                elif previousFrame == 0:
                    previousFrame = currentFrame
                    angles = self.getAngles(landmarks)
                    string = ','.join(landmarks.ravel().astype(str)) + '\n'
                    angleString = ','.join(angles.ravel().astype(str)) + '\n'
                    if flag_newfile:
                        labels = ','.join(np.arange(landmarks.ravel().shape[0]).astype(str))+'\n'
                        angleLabels = ','.join(np.arange(angles.ravel().shape[0]).astype(str))+'\n'
                        strWeights = ','.join(np.ones(landmarks.ravel().shape[0]).astype(str)) + '\n'
                        strAngleWeights = ','.join(np.ones(angles.ravel().shape[0]).astype(str)) + '\n'
#                         strFrameInterval = ','.join(np.repeat(frameInterval, landmarks.ravel().shape[0]).astype(int).astype(str)) + '\n'
#                         strAngleFrameInterval = ','.join(np.repeat(frameInterval, angles.ravel().shape[0]).astype(int).astype(str)) + '\n'
                        file.write(labels)
                        angleFile.write(angleLabels)
#                         file.write(strFrameInterval)
#                         angleFile.write(strAngleFrameInterval)
                        file.write(strWeights)
                        angleFile.write(strAngleWeights)
                        file.write(string)
                        angleFile.write(angleString)
        file.close()
        angleFile.close()
        camera_video.release()


    def processFrame(self, frame):
        # ADD DOCUMENTATION!!!!!!!
        frame = frame.to_ndarray(format="bgr24")
        # frame = np.array(frame)
        print(f"FRAME SHAPE: {frame.shape}")
        # Perform Pose landmark detection.
        frame, landmarks = self.detect(frame)
        # Get frame width and height.
        height, width, _ = frame.shape
        # Draw exit-zone and rest-zone circles on the frame.
        cv2.circle(frame,(int(self.trainParams['exitXCenter']*width),int(self.trainParams['exitYCenter']*height)),int(min(width, height)*self.trainParams['exitRadius']),(0,0,255),1)
        cv2.circle(frame,(int(self.trainParams['restXCenter']*width),int(self.trainParams['restYCenter']*height)),int(min(width, height)*self.trainParams['restRadius']),(0,255,0),1)
        # Some additional stuff to do with the first frame.
        if self.trainParams['flag_start']:
            # Form the background.
            tutorFrameMatrix = np.ones([height, width, 3]) * 255 # white bg
            cv2.imwrite(self.trainParams['bgImagePath'], tutorFrameMatrix)
            # Set the time-frame background to the one we have just formed.
            self.trainParams['timeFrameTemplate'] = cv2.imread(self.trainParams['bgImagePath'])
            # Set the background for training without tutorial video.
            if not self.trainParams['videoUnderSkeleton']:
                self.trainParams['tutorFrame'] = cv2.imread(self.trainParams['bgImagePath'])
            # Otherwise get the first frame from the tutorial.
            else:
                _, self.trainParams['tutorFrame'] = self.trainParams['tutor_video'].read()
            # Get the correct values of landmark coordinates.
            trueLandmarks = self.trainParams['poses'][self.trainParams['curIndex']].reshape(-1, 2)
            # Draw the landmarks.
            try:
                self.trainParams['tutorFrame'], _ = self.drawFromVectors(self.trainParams['tutorFrame'], trueLandmarks.copy(), radius=4)
            except Exception:
                trueLandmarks = self.trainParams['poses'][self.trainParams['curIndex']].reshape(-1, 2)
                self.trainParams['tutorFrame'], _ = self.drawFromVectors(self.trainParams['tutorFrame'], trueLandmarks.copy(), radius=4)
            # Flip the frame.
            self.trainParams['tutorFrame'] = cv2.flip(self.trainParams['tutorFrame'], 1)
            # Mark down, that it's not the first frame already.
            self.trainParams['flag_start'] = 0
            
        # Check if landmarks are spotted.
        if landmarks is not None:
            # Check if left wrist is in exit zone.
            if np.linalg.norm(landmarks[15] - np.array([self.trainParams['exitXCenter'], self.trainParams['exitYCenter']])) <= self.trainParams['exitRadius'] \
            or np.linalg.norm(landmarks[17] - np.array([self.trainParams['exitXCenter'], self.trainParams['exitYCenter']])) <= self.trainParams['exitRadius'] \
            or np.linalg.norm(landmarks[19] - np.array([self.trainParams['exitXCenter'], self.trainParams['exitYCenter']])) <= self.trainParams['exitRadius'] \
            or np.linalg.norm(landmarks[21] - np.array([self.trainParams['exitXCenter'], self.trainParams['exitYCenter']])) <= self.trainParams['exitRadius'] :
                # Decrease the amount of frames to exit.
                self.trainParams['framesToExit'] -= 1
                # Draw filled red exit circle.
                cv2.circle(frame,(int(self.trainParams['exitXCenter']*width),int(self.trainParams['exitYCenter']*height)),int(min(width, height)*self.trainParams['exitRadius']),(0,0,255),-1)
            # Otherwise set framesToExit value to default: 30.
            else:
                self.trainParams['framesToExit'] = 30
            # Check if right wrist is in rest zone.
            if np.linalg.norm(landmarks[16] - np.array([self.trainParams['restXCenter'], self.trainParams['restYCenter']])) <= self.trainParams['restRadius']:
                # Decrease the amount of frames before entering/exiting the rest mode.
                self.trainParams['framesToRestMode'] -= 1
                # Draw filled green rest circle.
                cv2.circle(frame,(int(self.trainParams['restXCenter']*width),int(self.trainParams['restYCenter']*height)),int(min(width, height)*self.trainParams['restRadius']),(0,255,0),-1)
                # Check if there are no frames till rest mode.
                if self.trainParams['framesToRestMode'] == 0:
                    # Change the mode.
                    self.trainParams['mode'] = (self.trainParams['mode'] + 1) % 2
                    # Check if it's rest mode, i.e. mode == 1.
                    if self.trainParams['mode']:
                        # Restart the exercise.
                        self.trainParams['curIndex'] = 1
                        if self.trainParams['videoUnderSkeleton']:
                            self.trainParams['tutor_video'].release()
                            self.trainParams['tutor_video'] = cv2.VideoCapture(self.trainParams['tutorVideoPath'])
                        self.trainParams['flag_start'] = 1
                    # Set the framesToRestMode parameter to default: 30.
                    self.trainParams['framesToRestMode'] = 30
            # Otherwise (right wrist outside of the rest circle) set the framesToRestMode parameter to default: 30.
            else:
                self.trainParams['framesToRestMode'] = 30
            # Check if we are in practice mode, i.e. mode == 0.
            if not self.trainParams['mode']:
                # Get the true landmarks' coordinates.
                trueLandmarks = self.trainParams['poses'][self.trainParams['curIndex']].reshape(-1, 2)
                # Get vectors between true landmarks.
                vectors = self.getVectors(trueLandmarks)
                # Calculate true inter-shoulder distance
                interShoulderDist = np.linalg.norm(vectors[0])
                # Calculate current inter-shoulder distance.
                curInterShoulderDist = np.linalg.norm(landmarks[11] - landmarks[12])
                # Form a coefficient k for self.drawFromVectors method.
                try:
                    # Check if true distance is too small (to get rid of ZeroDivision).
                    if np.abs(interShoulderDist) < 0.001:
                        raise ValueError
                    k = curInterShoulderDist / interShoulderDist
                except Exception:
                    # If we are here, then true distance is too small. Let's set k=1 for this case.
                    k = 1
                # Get the coordinates of current position of left shoulder for self.drawFromVectors method.
                lsh = landmarks[11]
                # Draw new (proportional) true landmarks on a separate frame.
                # For now we will need only the landmarks, not the frame with them drawn.
                _, newTrueLandmarks = self.drawFromVectors(frame.copy(), trueLandmarks.copy(), k=k, lsh=lsh)
                # Calculate the weightened distance between current landmarks and proportional true landmarks.
                diff = (landmarks - newTrueLandmarks) * self.trainParams['weights'].reshape(-1, 2)
                # Get true angles' values for current frame. 
                trueAngles = self.trainParams['angleMatrix'][self.trainParams['curIndex']]
                # Get current angles' values from the user pose.
                angles = self.getAngles(landmarks)
                # Calculate the weightened difference between true angles and user's angles.
                anglesDiff = (angles - trueAngles) * self.trainParams['angleWeights']
                # Calculate the norm of landmarks and angles differences.
                norm = np.linalg.norm(diff) + np.linalg.norm(anglesDiff)
                # Check if the norm is in yellow/green zone.
                if norm < 0.5:
                    # Check if the norm is in green zone.
                    if norm < 0.25:
                        # Set color to green.
                        curColor = (0,255,0)
                    # Otherwise set color to yellow.
                    else:
                        curColor = (255,255,0)
                    # Go to the next pose (increment pose index).
                    self.trainParams['curIndex'] += 1
                    # Check if user has completed 1 iteration of the exercise.
                    if self.trainParams['curIndex'] >= self.trainParams['poses'].shape[0]:
                        # Set parameters to the beginning of the new iteration.
                        self.trainParams['curIndex'] = 1
                        self.trainParams['weights'] = self.trainParams['poses'][0]
                        self.trainParams['angleWeights'] = self.trainParams['angleMatrix'][0]
                        # Increment the repeats' counter.
                        self.trainParams['repCnt'] += 1
                        # Restart the tutorial video reader if required.
                        if self.trainParams['videoUnderSkeleton']:
                            self.trainParams['tutor_video'].release()
                            self.trainParams['tutor_video'] = cv2.VideoCapture(self.trainParams['tutorVideoPath'])
                    # Check if we have to adjust the weights.
                    if np.max(self.trainParams['poses'][self.trainParams['curIndex']]) <= -1:
                        self.trainParams['weights'] = self.trainParams['poses'][self.trainParams['curIndex']+1]
                        self.trainParams['angleWeights'] = self.trainParams['angleMatrix'][self.trainParams['curIndex']+1]
                        self.trainParams['curIndex'] += 2
                    # Get next tutorial frame.
                    if self.trainParams['videoUnderSkeleton']:
                        _, self.trainParams['tutorFrame'] = self.trainParams['tutor_video'].read()
                    try:
                        self.trainParams['tutorFrame'], _ = self.drawFromVectors(self.trainParams['tutorFrame'], trueLandmarks.copy(), radius=4)
                    except Exception:
                        trueLandmarks = self.trainParams['poses'][self.trainParams['curIndex']].reshape(-1, 2)
                        self.trainParams['tutorFrame'], _ = self.drawFromVectors(self.trainParams['tutorFrame'], trueLandmarks.copy(), radius=4)
                    # Flip the tutorial frame.
                    self.trainParams['tutorFrame'] = cv2.flip(self.trainParams['tutorFrame'], 1)
                # Otherwise we are in red zone.
                else:
                    # Set color to red.
                    curColor = (255,0,0)
                # Draw landmarks on the frame with current color.
                frame = self.drawLandmarks(frame, landmarks.copy(), color=curColor)
                # If we are not in green zone, display the true skeleton if required.
                if curColor != (0, 255, 0) and self.trainParams['displaySkeleton']:
                    frame, _ = self.drawFromVectors(frame, trueLandmarks.copy(), k=k, lsh=lsh)
        # Flip the frame.
        frame = cv2.flip(frame, 1)
        # Get the current time.
        self.trainParams['timeFrame'] = self.trainParams['timeFrameTemplate'].copy()
        # Calculate the execution time in minutes/seconds.
        timeSec = int(time.time() - self.trainParams['timeStart']) % 60
        timeMin = int(time.time() - self.trainParams['timeStart']) // 60
        # Make some format adjustments, if seconds are between 0 and 9: from s to 0s (Example: 2-> 02).
        if timeSec < 10:
            timeSec = f'0{timeSec}'
        # Put some information about current progress and execution time into the frame.
        cv2.putText(self.trainParams['timeFrame'], f'Time: {timeMin}:{timeSec}', (self.trainParams['timeFrame'].shape[1]//3,self.trainParams['timeFrame'].shape[0]//3), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        cv2.putText(self.trainParams['timeFrame'], f"Repeats: {self.trainParams['repCnt']}", (self.trainParams['timeFrame'].shape[1]//3,2*self.trainParams['timeFrame'].shape[0]//3), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        cv2.putText(self.trainParams['timeFrame'], f"Mode: {self.trainParams['modes'][self.trainParams['mode']]}", (self.trainParams['timeFrame'].shape[1]//3,3*self.trainParams['timeFrame'].shape[0]//4), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        # Resize all 3 (time frame, user frame and tutor frame) frames, so that to concat it properly later.
        frame = cv2.resize(frame, (self.trainParams['fullscreenWidth']//3, self.trainParams['fullscreenHeight']))
        self.trainParams['tutorFrame'] = cv2.resize(self.trainParams['tutorFrame'], (self.trainParams['fullscreenWidth']//3, self.trainParams['fullscreenHeight']))
        self.trainParams['timeFrame'] = cv2.resize(self.trainParams['timeFrame'], (self.trainParams['fullscreenWidth']//3, self.trainParams['fullscreenHeight']))
        # If debug mode is activated, put some debug information into the user frame and tutor frame.
        if self.trainParams['debugMode']:
            cv2.putText(self.trainParams['tutorFrame'], f"Pose {self.trainParams['curIndex']}", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            try:
                cv2.putText(frame, f'Norm: {norm}', (10,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)
            except Exception:
                pass
        # Generate a supeFrame by horizontal concatenation of time frame, user frame and tutor frame.
        superFrame = cv2.hconcat([self.trainParams['timeFrame'], frame, self.trainParams['tutorFrame']])
        # # Get RGB superFrame.
        # superFrame = cv2.cvtColor(superFrame, cv2.COLOR_BGR2RGB)
        print("SUPER FRAME GENERATED")
        return av.VideoFrame.from_ndarray(superFrame, format="bgr24")


    def setTrainParams(self, exerciseName, videoPath=0, outputVideoPath=None, videoUnderSkeleton=True, displaySkeleton=True, debugMode=False):
        # ADD DOCUMENTATION!!!!!
        self.trainParams['fileName'] = exerciseName + os.extsep + 'csv'
        self.trainParams['fileDir'] = 'exercises'
        self.trainParams['filePath'] = os.path.join(self.trainParams['fileDir'], self.trainParams['fileName'])
        self.trainParams['poses'] = np.array(pd.read_csv(self.trainParams['filePath']))
        self.trainParams['weights'] = self.trainParams['poses'][0]
        self.trainParams['tutorVideoPath'] = os.path.join('media', 'video', exerciseName, exerciseName + '_tutorial' + os.extsep + 'mp4')
        self.trainParams['angleFileName'] = exerciseName + '_angles' + os.extsep + 'csv'
        self.trainParams['angleFileDir']= 'exercises'
        self.trainParams['angleFilePath'] = os.path.join(self.trainParams['angleFileDir'], self.trainParams['angleFileName'])
        self.trainParams['angleMatrix'] = np.array(pd.read_csv(self.trainParams['angleFilePath']))#[1:]
        self.trainParams['angleWeights'] = self.trainParams['angleMatrix'][0]
        self.trainParams['curIndex'] = 1
        self.trainParams['curPose'] = self.trainParams['poses'][self.trainParams['curIndex']]
        self.trainParams['timeStart'] = time.time()
        self.trainParams['repCnt'] = 0
        self.trainParams['flag_start'] = 1
        self.trainParams['videoPath'] = videoPath
        self.trainParams['outputVideoPath'] = outputVideoPath
        self.trainParams['videoUnderSkeleton'] = videoUnderSkeleton
        self.trainParams['displaySkeleton'] = displaySkeleton
        self.trainParams['debugMode'] = debugMode
        # Initialize video capture object and video writer object (if required).
        self.trainParams['camera_video'] = cv2.VideoCapture(self.trainParams['videoPath'])
        if self.trainParams['outputVideoPath'] is not None:
            self.trainParams['video_writer'] = cv2.VideoWriter(
                self.trainParams['outputVideoPath'], 
                cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), 
                self.fps, 
                (self.trainParams['fullscreenWidth'], self.trainParams['fullscreenHeight'])
            )
        if videoUnderSkeleton:
            self.trainParams['tutor_video'] = cv2.VideoCapture(self.trainParams['tutorVideoPath'])
        return


    def train(self, exerciseName, videoPath=0, outputVideoPath=None, videoUnderSkeleton=True, displaySkeleton=True, debugMode=False):
        # ADD DOCUMENTATION!!!!!
        # Set the training parameters.
        self.setTrainParams(exerciseName, videoPath, outputVideoPath, videoUnderSkeleton, displaySkeleton, debugMode)
        cv2.namedWindow('Practice', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Practice', self.trainParams['fullscreenWidth'], self.trainParams['fullscreenHeight'])
        cv2.moveWindow('Practice', 0, 0)
        # Iterate until the video is not over.
        while self.trainParams['camera_video'].isOpened():
            # Read a frame.
            ok, frame = self.trainParams['camera_video'].read()
            # Check if frame is not read properly.
            if not ok:
                # Breal the loop.
                break
            # Check if users wants to exit.
            if self.trainParams['framesToExit'] == 0:
                break
            # Process the frame.
            superFrame = self.processFrame(frame) # RGB superFrame
            # Convert the frame from RGB to BGR to display it with cv2.
            superFrame = cv2.cvtColor(superFrame, cv2.COLOR_RGB2BGR)
            # Display the frame and write it into the file (if required).
            cv2.imshow('Practice', superFrame)
            if self.trainParams['outputVideoPath'] is not None:
                self.trainParams['video_writer'].write(superFrame)
            # Wait for a key to be pressed.
            k = cv2.waitKey(1)
            # If it's an 'esc' key, break the loop.
            if k == 27:
                break
        # Release all captures/writers and destroy all windows. 
        self.trainParams['camera_video'].release()
        if self.trainParams['videoUnderSkeleton']:
            self.trainParams['tutor_video'].release()
        if self.trainParams['outputVideoPath'] is not None:
            self.trainParams['video_writer'].release()
        cv2.destroyAllWindows()
        return


# class VideoProcessor(VideoProcessorBase):
# 	def __init__(self, exerciseName):
# 		self.exerciseName = exerciseName
# 		self.vision = AusarVision()
# 		self.vision.setTrainParams(exerciseName)

# 	def transform(self, frame):
# 		return self.vision.processFrame(frame)

exerciseName = 'squat'
vision = AusarVision()
vision.setTrainParams(exerciseName)

webrtc_streamer(key="example", video_frame_callback=vision.processFrame)