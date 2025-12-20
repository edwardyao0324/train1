import cv2
import mediapipe as mp
import math

mp_drawing = mp.solutions.drawing_utils  # 用於繪製手部關鍵點
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands  # mediapipe 手部偵測模組

# 根據兩點的座標，計算向量的夾角
def vector_2d_angle(v1, v2):
    v1_x = v1[0]
    v1_y = v1[1]
    v2_x = v2[0]
    v2_y = v2[1]
    try:
        angle_ = math.degrees(math.acos((v1_x * v2_x + v1_y * v2_y) / (((v1_x ** 2 + v1_y ** 2) ** 0.5) * ((v2_x ** 2 + v2_y ** 2) ** 0.5))))
    except:
        angle_ = 180  # 當計算出錯時，將角度設為 180
    return angle_

# 根據傳入的 21 個關鍵點座標，計算手指的角度
def hand_angle(hand_):
    angle_list = []
    # 計算大拇指角度
    angle_ = vector_2d_angle(
            ((int(hand_[0][0]) - int(hand_[2][0])), (int(hand_[0][1]) - int(hand_[2][1]))),
            ((int(hand_[3][0]) - int(hand_[4][0])), (int(hand_[3][1]) - int(hand_[4][1])))
        )
    angle_list.append(angle_)
    # 計算食指角度
    angle_ = vector_2d_angle(
            ((int(hand_[0][0]) - int(hand_[6][0])), (int(hand_[0][1]) - int(hand_[6][1]))),
            ((int(hand_[7][0]) - int(hand_[8][0])), (int(hand_[7][1]) - int(hand_[8][1])))
        )
    angle_list.append(angle_)
    # 計算中指角度
    angle_ = vector_2d_angle(
            ((int(hand_[0][0]) - int(hand_[10][0])), (int(hand_[0][1]) - int(hand_[10][1]))),
            ((int(hand_[11][0]) - int(hand_[12][0])), (int(hand_[11][1]) - int(hand_[12][1])))
        )
    angle_list.append(angle_)
    # 計算無名指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0]) - int(hand_[14][0])), (int(hand_[0][1]) - int(hand_[14][1]))),
        ((int(hand_[15][0]) - int(hand_[16][0])), (int(hand_[15][1]) - int(hand_[16][1])))
        )
    angle_list.append(angle_)
    # 計算小拇指角度
    angle_ = vector_2d_angle(
            ((int(hand_[0][0]) - int(hand_[18][0])), (int(hand_[0][1]) - int(hand_[18][1]))),
            ((int(hand_[19][0]) - int(hand_[20][0])), (int(hand_[19][1]) - int(hand_[20][1])))
        )
    angle_list.append(angle_)
    return angle_list

# 根據手指角度的列表內容，返回對應的手勢名稱
def hand_pos(finger_angle):
    f1 = finger_angle[0]  # 大拇指角度
    f2 = finger_angle[1]  # 食指角度
    f3 = finger_angle[2]  # 中指角度
    f4 = finger_angle[3]  # 無名指角度
    f5 = finger_angle[4]  # 小拇指角度

    # 小於 50 表示手指伸直，大於等於 50 表示手指捲縮
    if f1 < 50 and f2 >= 50 and f3 >= 50 and f4 >= 50 and f5 >= 50:
        return 'good'
    elif f1 >= 50 and f2 >= 50 and f3 < 50 and f4 >= 50 and f5 >= 50:
        return 'no!!!'
    elif f1 < 50 and f2 < 50 and f3 >= 50 and f4 >= 50 and f5 < 50:
        return 'ROCK!'
    elif f1 >= 50 and f2 >= 50 and f3 >= 50 and f4 >= 50 and f5 >= 50:
        return '0'
    elif f1 >= 50 and f2 >= 50 and f3 >= 50 and f4 >= 50 and f5 < 50:
        return 'pink'
    elif f1 >= 50 and f2 < 50 and f3 >= 50 and f4 >= 50 and f5 >= 50:
        return '1'
    elif f1 >= 50 and f2 < 50 and f3 < 50 and f4 >= 50 and f5 >= 50:
        return '2'
    elif f1 >= 50 and f2 >= 50 and f3 < 50 and f4 < 50 and f5 < 50:
        return 'ok'
    elif f1 < 50 and f2 >= 50 and f3 < 50 and f4 < 50 and f5 < 50:
        return 'ok'
    elif f1 >= 50 and f2 < 50 and f3 < 50 and f4 < 50 and f5 > 50:
        return '3'
    elif f1 >= 50 and f2 < 50 and f3 < 50 and f4 < 50 and f5 < 50:
        return '4'
    elif f1 < 50 and f2 < 50 and f3 < 50 and f4 < 50 and f5 < 50:
        return '5'
    elif f1 < 50 and f2 >= 50 and f3 >= 50 and f4 >= 50 and f5 < 50:
        return '6'
    elif f1 < 50 and f2 < 50 and f3 >= 50 and f4 >= 50 and f5 >= 50:
        return '7'
    elif f1 < 50 and f2 < 50 and f3 < 50 and f4 >= 50 and f5 >= 50:
        return '8'
    elif f1 < 50 and f2 < 50 and f3 < 50 and f4 < 50 and f5 >= 50:
        return '9'
    else:
        return ''

cap = cv2.VideoCapture(0)  # 啟用攝影機
fontFace = cv2.FONT_HERSHEY_SIMPLEX  # 設定字型
lineType = cv2.LINE_AA  # 設定字型邊框

# 啟用 Mediapipe 進行手部偵測
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    if not cap.isOpened():
        print("無法開啟攝影機")
        exit()
    w, h = 700, 500  # 設定影像尺寸
    while True:
        ret, img = cap.read()  # 從攝影機讀取影像
        img = cv2.resize(img, (w, h))  # 縮小影像尺寸以提高處理效率
        if not ret:
            print("無法接收影像")
            break
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 轉換為 RGB 格式
        results = hands.process(img2)  # 處理影像以進行手部偵測
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                finger_points = []  # 用來存儲手指節點座標的列表
                fx = []  # 存儲所有 x 座標
                fy = []  # 存儲所有 y 座標
                for i in hand_landmarks.landmark:
                    # 計算手指每個節點的座標並保存
                    x = i.x * w  # 計算 x 座標
                    y = i.y * h  # 計算 y 座標
                    finger_points.append((x, y))
                    fx.append(int(x))  # 記錄 x 座標
                    fy.append(int(y))  # 記錄 y 座標
                if finger_points:
                    finger_angle = hand_angle(finger_points)  # 計算手指角度
                    text = hand_pos(finger_angle)  # 根據手指角度得到手勢名稱
                    if text == 'no!!!':
                        x_max = max(fx)  # 取得最大 x 座標
                        y_max = max(fy)  # 取得最大 y 座標
                        x_min = min(fx) - 10  # 取得最小 x 座標並向外擴展
                        y_min = min(fy) - 10  # 取得最小 y 座標並向外擴展
                        if x_max > w: x_max = w  # 確保最大值不超過邊界
                        if y_max > h: y_max = h  # 確保最大值不超過邊界
                        if x_min < 0: x_min = 0  # 確保最小值不小於 0
                        if y_min < 0: y_min = 0  # 確保最小值不小於 0
                        mosaic_w = x_max - x_min  # 計算馬賽克區域的寬度
                        mosaic_h = y_max - y_min  # 計算馬賽克區域的高度
                        mosaic = img[y_min:y_max, x_min:x_max]  # 截取馬賽克區域
                        mosaic = cv2.resize(mosaic, (8, 8), interpolation=cv2.INTER_LINEAR)  # 縮小馬賽克區域
                        mosaic = cv2.resize(mosaic, (mosaic_w, mosaic_h), interpolation=cv2.INTER_NEAREST)  # 放大回原始大小
                        img[y_min:y_max, x_min:x_max] = mosaic  # 將馬賽克區域放回原圖
                    else:
                        cv2.putText(img, text, (30, 120), fontFace, 5, (255, 255, 255), 10, lineType)  # 顯示手勢名稱

        cv2.imshow('Joyous Hand_pos', img)  # 顯示處理後的影像
        if cv2.waitKey(5) == ord('q'):  # 按 'q' 鍵退出
            break

cap.release()  # 釋放攝影機
cv2.destroyAllWindows()  # 關閉所有 OpenCV 視窗