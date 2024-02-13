import cv2
import mediapipe as mp
import numpy as np

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    # Vectors in 3D space
    vector1 = a - b
    vector2 = c - b
    
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    
    # Cosine of the angle
    cos_theta = dot_product / (norm_vector1 * norm_vector2)
    
    # Ensure cos_theta is within [-1, 1] to avoid numerical errors
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # Angle in radians
    radians = np.arccos(cos_theta)
    
    # Convert radians to degrees
    angle = np.degrees(radians)
    
    return angle

def forehand_checking(video_input):
    # Set up media pipe
    mp_drawing = mp.solutions.drawing_utils
    # Import pose estimation model
    mp_pose = mp.solutions.pose

    # VIDEO FEED
    #cap = cv2.VideoCapture(0)
    cap = video_input
    # Get the original width and height of the video
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))



    # Set up mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            # Read a frame from the video
            # ret -> return variable
            # frame -> image from cap
            ret, frame = cap.read()

            if not ret:
                print("End of video.")
                break  # Exit the loop when the video is finished.
            
            # Detect stuff and render
            
            # Recolor image
            # default of opencv is rgb, that's why we recolor
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # set the image to not be modified and save memory
            image.flags.writeable = False
            
            # Make detection
            results = pose.process(image) # get detection of pose
            
            # Recolor back to BGR for opencv
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark
                shoulder_right = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                wrist_right = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                
                shoulder_left = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow_left = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist_left = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                
                # calculate angle
                angle_right = calculate_angle(shoulder_right, elbow_right, wrist_right)
                angle_left = calculate_angle(shoulder_left, elbow_left, wrist_left)
                
                #angle_right_3d = calculate_angle_3D(shoulder_right, elbow_right, wrist_right)
                #angle_left_3d = calculate_angle_3D(shoulder_left, elbow_left, wrist_left)

                # display value
                #cv2.putText(image, str(angle_right_3d), [400, 100], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
                # display value
                cv2.putText(image, str(round(angle_right, 2)), 
                            tuple(np.multiply(elbow_right, [desired_width, desired_height]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
                
                #mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
                # elbow, [original_width, original_height] -> normalized coordinates
            except Exception as e:
                print(f"Error: {e}")
                pass
            
            # Render detections
            # draw_landamarks -> utils to draw in image
            # results.pose_landmarks -> coordenates
            # mp_pose.POSE_CONNECTIONS -> how body points are conencted inside model
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Update desired height and width before resizing
            desired_height, desired_width = original_height, original_width

            # Resize the image
            image = cv2.resize(image, (desired_width, desired_height))

            # Display the resized image
            cv2.imshow('UltiThrow', image)
            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break

    # Release the video capture object and close the window
    #cap.release()
    #cv2.destroyAllWindows()