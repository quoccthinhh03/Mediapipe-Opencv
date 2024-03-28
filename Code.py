import cv2
import mediapipe as mp
import os

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(0)

FolderPath = "Fingers"
lst=os.listdir(FolderPath)
lst_2=[]
for i in lst:
    #print(i)
    image=cv2.imread(f"{FolderPath}/{i}")
    #qprint(f"{FolderPath}/{i}")
    lst_2.append(image)
    #print(lst_2[10].shape)

while True:
    
    success, img = cap.read()
    if not success:
        break
    img=cv2.flip(img,1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    count_left = 0
    count_right = 0
    if results.multi_hand_landmarks: # neu phat hien ra ban tay
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness): # duyet vong lap nhan dien cac moc ban tay , va nhan dien ban tay trai va phai
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS) # phat hien ban tay thi noi cac toa do lai
            #print(hand_landmarks,handedness) # hand_landmarks cac toa do x,y va z | handedness : nhan dien tay trai hay tay phai , co 3 thong so la  {index, score, label}
            my_hand = []  # Tao 1 list de luu cac toa do
            for idx, lm in enumerate(hand_landmarks.landmark):
                #print(idx,lm) # idx la moc ban tay co 21 moc | lm la toa do x,y, z
                h, w, _ = img.shape # lay kich thuoc cao,rong, kenh màu từ cam
                my_hand.append([int(lm.x * w), int(lm.y * h)])
                #print(my_hand) # in ra toa do so nguyen của cac ngon tay
            if handedness.classification[0].label == "Left": # kiem tra neu la tay trai thi
                #ngon cai
                if my_hand[4][0] > my_hand[3][0]:
                    count_left +=1
                #4 ngon con lai    
                if my_hand[8][1] < my_hand[6][1]:
                    count_left  += 1
                if my_hand[12][1] < my_hand[10][1]:
                    count_left  += 1
                if my_hand[16][1] < my_hand[14][1]:
                    count_left  += 1
                if my_hand[20][1] < my_hand[18][1]:
                    count_left  += 1
            if handedness.classification[0].index == 1: # kiem tra neu la tay trai thi
                #ngon cai
                if my_hand[4][0] < my_hand[3][0]:
                    count_right += 1
                #4 ngon con lai     
                if my_hand[8][1] < my_hand[6][1]:
                    count_right += 1
                if my_hand[12][1] < my_hand[10][1]:
                    count_right += 1
                if my_hand[16][1] < my_hand[14][1]:
                    count_right += 1
                if my_hand[20][1] < my_hand[18][1]:
                    count_right += 1
        # In ra man hinh
        cv2.putText(img, str(count_left+count_right)+" ngon tay", (30, 390), cv2.FONT_HERSHEY_PLAIN, 6, (255, 0, 0), 5)
        h, w, c = lst_2[(count_left+count_right) - 1].shape
        img[0:h, 0:w] = lst_2[(count_left+count_right) - 1]
    cv2.imshow('Nhan dien ban tay', img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
cap.release() # Giải phóng cam
cv2.destroyAllWindows() # Đóng tất cả cửa sổ
