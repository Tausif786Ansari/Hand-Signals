import cv2
import mediapipe as mp
import numpy as np

#----------------Function to Draw Bounding Box----------------
def get_hand_bbox(result, H, W):
    x_val = [int(lm.x * W) for lm in result.landmark]
    y_val = [int(lm.y * H) for lm in result.landmark]
    x_min = max(0, min(x_val) - 20)
    y_min = max(0, min(y_val) - 20)
    x_max = min(W, max(x_val) + 20)
    y_max = min(H, max(y_val) + 20)
    return (x_min, y_min, x_max, y_max)

#-----------------Function To Draw on Frame-------------------
def drawSign(frame, x_min, y_min, x_max, y_max, text):
    cv2.rectangle(frame, (x_min, y_min - 40), (x_max, y_min), (0, 255, 0), -1, cv2.LINE_AA)
    cv2.putText(frame, text, (x_min + 2, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

#---------------Function to Calculate Point-------------
def get_pt(result, H, W, idx):
    return (int(result.landmark[idx].x * W), int(result.landmark[idx].y * H))

#----------------Function to Get All Points-------
def get_all_pt(result, H, W):
    thx1, thy1 = get_pt(result, H, W, 1)  # Lower thumb
    thx, thy = get_pt(result, H, W, 4)    # Thumb tip
    indx1, indy1 = get_pt(result, H, W, 5)  # Lower index
    indx, indy = get_pt(result, H, W, 8)    # Index tip
    midx1, midy1 = get_pt(result, H, W, 9)  # Lower middle
    midx, midy = get_pt(result, H, W, 12)   # Middle tip
    rngx1, rngy1 = get_pt(result, H, W, 13) # Lower ring
    rngx, rngy = get_pt(result, H, W, 16)   # Ring tip
    pnkx1, pnky1 = get_pt(result, H, W, 17) # Lower pinky
    pnkx, pnky = get_pt(result, H, W, 20)   # Pinky tip
    pamx, pamy = get_pt(result, H, W, 0)    # Palm (wrist)
    Lower_thumb = np.array((thx1, thy1))
    thumb = np.array((thx, thy))
    Lower_index = np.array((indx1, indy1))
    index = np.array((indx, indy))
    Lower_middle = np.array((midx1, midy1))
    middle = np.array((midx, midy))
    Lower_ring = np.array((rngx1, rngy1))  # Fixed: "ring" instead of "ringe"
    ring = np.array((rngx, rngy))          # Fixed: "ring" instead of "ringe"
    Lower_pinky = np.array((pnkx1, pnky1))
    pinky = np.array((pnkx, pnky))
    palm = np.array((pamx, pamy))
    return (thumb, index, middle, ring, pinky, palm, Lower_thumb, Lower_index, Lower_middle, Lower_ring, Lower_pinky)

#---------------Function for OK Sign-------------------------
def get_ok(result, H, W):
    thumb, index, middle, ring, pinky = get_all_pt(result, H, W)[:5]
    dst_btw_thum_ind = np.linalg.norm(thumb - index)
    dst_btw_thum_mid = np.linalg.norm(thumb - middle)
    dst_btw_thum_rng = np.linalg.norm(thumb - ring)  # Fixed: "ring" instead of "ringe"
    dst_btw_thum_pnk = np.linalg.norm(thumb - pinky)
    if dst_btw_thum_ind <= 30 and dst_btw_thum_mid >= 150 and dst_btw_thum_rng >= 150 and dst_btw_thum_pnk >= 150:
        return True
    return False

#--------------Function for Peace Sign----------------------
def get_peace(result, H, W):
    thumb, index, middle, ring, pinky, palm = get_all_pt(result, H, W)[:6]
    
    # Calculate distances from each finger tip to palm
    d_thumb = np.linalg.norm(thumb - palm)
    d_index = np.linalg.norm(index - palm)
    d_middle = np.linalg.norm(middle - palm)
    d_ring = np.linalg.norm(ring - palm)
    d_pinky = np.linalg.norm(pinky - palm)
    
    # Distance between index and middle tips (should be apart for peace)
    d_index_middle = np.linalg.norm(index - middle)
    
    # Thresholds tuned for ~800x600 frame; adjust if needed
    if (d_thumb < 120 and d_ring < 120 and d_pinky < 120 and  
        d_index > 180 and d_middle > 180 and d_index_middle > 50):                    
        return True
    return False

#---------------------Function for HangLoose Sign-------------
def get_HangLoose(result,H,W):
    thumb, index, middle, ring, pinky, palm = get_all_pt(result, H, W)[:6]
    
    # Calculate distances from each finger tip to palm
    d_thumb = np.linalg.norm(thumb - palm)
    d_index = np.linalg.norm(index - palm)
    d_middle = np.linalg.norm(middle - palm)
    d_ring = np.linalg.norm(ring - palm)
    d_pinky = np.linalg.norm(pinky - palm)
    
    # Thresholds tuned for ~800x600 frame; adjust if needed
    if (d_thumb > 150 and d_pinky > 150 and
        d_index < 120 and d_middle < 120 and d_ring < 120):
        return True
    return False

#--------------Function for Looser Sign------------------
def get_Loser(result,H,W):
    thumb, index, middle, ring, pinky, palm = get_all_pt(result, H, W)[:6]
    
    # Calculate distances from each finger tip to palm
    d_thumb = np.linalg.norm(thumb - palm)
    d_index = np.linalg.norm(index - palm)
    d_middle = np.linalg.norm(middle - palm)
    d_ring = np.linalg.norm(ring - palm)
    d_pinky = np.linalg.norm(pinky - palm)
    
    # Thresholds tuned for ~800x600 frame; adjust if needed
    if (d_thumb > 150 and d_index > 150 and
        d_pinky < 110 and d_middle < 110 and d_ring < 110):
        return True
    return False

#--------------------Function for Rock Sign----------------
def get_Rock(result,H,W):
    thumb, index, middle, ring, pinky, palm = get_all_pt(result, H, W)[:6]
    
    # Calculate distances from each finger tip to palm
    d_thumb = np.linalg.norm(thumb - palm)
    d_index = np.linalg.norm(index - palm)
    d_middle = np.linalg.norm(middle - palm)
    d_ring = np.linalg.norm(ring - palm)
    d_pinky = np.linalg.norm(pinky - palm)
    
    d_index_pinky = np.linalg.norm(index-pinky)
    
    # Thresholds tuned for ~800x600 frame; adjust if needed
    if (d_pinky > 150 and d_index > 150 and d_index_pinky > 110 and
        d_thumb < 130 and d_middle < 120 and d_ring < 120):
        return True
    return False

#--------------------Function for Bang Bang sign----------
def get_Bang(result,H,W):
    thumb, index, middle, ring, pinky, palm = get_all_pt(result, H, W)[:6]
    
    # Calculate distances from each finger tip to palm
    d_thumb = np.linalg.norm(thumb - palm)
    d_index = np.linalg.norm(index - palm)
    d_middle = np.linalg.norm(middle - palm)
    d_ring = np.linalg.norm(ring - palm)
    d_pinky = np.linalg.norm(pinky - palm)
    
    # Thresholds tuned for ~800x600 frame; adjust if needed
    if (d_thumb > 150 and d_index > 150 and d_middle > 150 and
        d_pinky < 120 and d_ring < 120):
        return True
    return False

#--------------------Function for Good Job----------------
def get_GoodJob(result,H,W):
    thumb, index, middle, ring, pinky, palm = get_all_pt(result, H, W)[:6]
    
    # Calculate distances from each finger tip to palm
    d_thumb = np.linalg.norm(thumb - palm)
    d_index = np.linalg.norm(index - palm)
    d_middle = np.linalg.norm(middle - palm)
    d_ring = np.linalg.norm(ring - palm)
    d_pinky = np.linalg.norm(pinky - palm)
    
    # Thresholds tuned for ~800x600 frame; adjust if needed
    if (d_thumb > 150 and d_index < 110 and
        d_pinky < 110 and d_middle < 110 and d_ring < 110):
        return True
    return False

#-------------Module of Hand and Drawing--------------
Hand_Module = mp.solutions.hands
Draw_Module = mp.solutions.drawing_utils
Cap = cv2.VideoCapture(1)

#-------------Calling Hand Class----------------------
with Hand_Module.Hands(static_image_mode=False, max_num_hands=1, model_complexity=1, min_detection_confidence=0.7, min_tracking_confidence=0.7) as Hand_obj:
    while Cap.isOpened():
        ret, frame = Cap.read()
        if not ret:
            break
        frame = cv2.resize(frame,None,fx=1.5,fy=1.5)
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result_obj = Hand_obj.process(rgb_frame)
        if result_obj.multi_hand_landmarks:
            result = result_obj.multi_hand_landmarks[0]
            Draw_Module.draw_landmarks(frame, result, Hand_Module.HAND_CONNECTIONS,
                                       Draw_Module.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                       Draw_Module.DrawingSpec(color=(255, 0, 0), thickness=2))
            
            #----------------Draw Bounding Box------------------
            H, W, _ = frame.shape
            x_min, y_min, x_max, y_max = get_hand_bbox(result, H, W)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2, cv2.LINE_AA)
            
            #------------------Checking which sign------------
            if get_ok(result, H, W):
                drawSign(frame, x_min, y_min, x_max, y_max, 'OK')
            if get_peace(result, H, W):
                drawSign(frame, x_min, y_min, x_max, y_max, 'PEACE')
            if get_HangLoose(result,H,W):
                drawSign(frame, x_min, y_min, x_max, y_max, 'HANG LOOSE')
            if get_Loser(result,H,W):
                drawSign(frame, x_min, y_min, x_max, y_max, 'LOSER')
            if get_GoodJob(result,H,W):
                drawSign(frame, x_min, y_min, x_max, y_max, 'GOOD JOB')
            if get_Rock(result,H,W):
                drawSign(frame, x_min, y_min, x_max, y_max, 'ROCK')
            if get_Bang(result,H,W):
                drawSign(frame, x_min, y_min, x_max, y_max, 'BANG BANG')
                
        cv2.imshow("Hand Drawing", frame)
        if cv2.waitKey(1) == ord('q'):
            break
Cap.release()
cv2.destroyAllWindows()