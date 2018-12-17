import numpy as np
import cv2
import matplotlib.pyplot as plt
import keras

from keras.models import load_model


# Thông số cho cửa sổ tách bàn tay
x0 = 400
y0 = 100
height = 200
width = 200

# Chương trình con tách vùng bàn tay
# Sau đó chuyển ảnh có chứa bàn tay sang ảnh nhị phân
def binaryMask(frame, x0, y0, width, height):
    cv2.rectangle(frame, (x0,y0),(x0+width,y0+height),(0,255,0),1)
    roi = frame[y0:y0+height, x0:x0+width]
    fgmask = bgModel.apply(roi,learningRate=0)
    res = cv2.bitwise_and(roi, roi, mask=fgmask)
    cv2.imshow('res',res)
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    ret, thresh = cv2.threshold(blur,20, 255, cv2.THRESH_BINARY)
    return thresh

# Load model CNN đã training từ file train.py
model = load_model('cnn_model.h5')
print("Loaded model from disk")

# Biến nhớ cho biết đã lấy background hay chưa ?
isBgCaptured = 0   

cap = cv2.VideoCapture(0)
while(1):
    ret,frame = cap.read()
    frame = cv2.flip(frame, 1)  
    

    # Khi chưa tách nền, isBgCaptured = 0 --> không làm gì cả
    # Nếu đã tách nền --> xử lý cửa sổ được tách
    if isBgCaptured == 1:

        # Gọi hàm tách bàn tay, đưa về nhị phân:
        binary = binaryMask(frame,x0,y0,width,height)
        binary = cv2.flip(binary,1)
        cv2.imshow('binary',binary)

        # Resize ảnh về đúng chuẩn với đầu vào của mạng CNN
        binary = cv2.resize(binary,(28,28))
        binary = np.asarray(binary)
        binary = binary.reshape(-1,28,28,1)

        # Nhận diện cử chỉ bằng mạng CNN
        y = model.predict(binary)
        y = y.flatten()
        
        # Do ngõ ra của mạng có sử dụng hàm softmax
        # nên kết quả sẽ là phần tử có xác suất cao nhất
        i = np.argmax(y)
        objects = ['PUNCH', 'HAND', 'NONE', 'ONE', 'TWO']
        text = objects[i]

        # Ghi kết quả lên màn hình
        cv2.putText(frame, text, (400, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), lineType=cv2.LINE_AA) 

        # Vẽ đồ thị cột biểu thị xác suất của từng ngõ ra sau khi qua hàm softmax 
        y_pos = np.arange(len(objects))
        plt.bar(y_pos, y, align='center', alpha=0.5)
        plt.xticks(y_pos, objects)
        plt.draw()
        plt.pause(0.001)
        plt.cla()
    cv2.imshow('frame',frame)

    k = cv2.waitKey(25)
    if k == 27:
        break
    elif k == ord('b'):  # Ấn 'b' để lấy background
        bgModel = cv2.createBackgroundSubtractorMOG2(0, 100)
        isBgCaptured = 1
        print( '!!!Background Captured!!!')
    elif k == ord('r'):  # Ấn 'r' để reset background
        bgModel = None
        isBgCaptured = 0
        print ('!!!Reset BackGround!!!')
   
cap.release()
cv2.destroyAllWindows()