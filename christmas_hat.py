import cv2
import random


def face_detect(img, cname):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_hist = cv2.equalizeHist(img_gray)
    face_cascade = cv2.CascadeClassifier(cname)
    faces = face_cascade.detectMultiScale(img_hist)
    return faces


def christmas_hat(img, cname='data/haarcascade_frontalface_alt.xml'):
    
    faces = face_detect(img, cname)
    
    hats = [cv2.imread(f'img/hats/hat_{i+1}.png', -1) for i in range(3)]
    for face in faces:
        hat = random.choice(hats) 
        scale = face[3] / hat.shape[0] * 2  
        hat = cv2.resize(hat, (0, 0), fx=scale, fy=scale) 
        x_offset = int(face[0] + face[2] / 2 - hat.shape[1] / 2) 
        y_offset = int(face[1] - hat.shape[0] / 2)
        
        x1 = max(x_offset, 0)
        x2 = min(x_offset + hat.shape[1], img.shape[1])
        y1 = max(y_offset, 0)
        y2 = min(y_offset + hat.shape[0], img.shape[0])
        hat_x1 = max(0, -x_offset)
        hat_x2 = hat_x1 + x2 - x1
        hat_y1 = max(0, -y_offset)
        hat_y2 = hat_y1 + y2 - y1
        
        alpha_h = hat[hat_y1:hat_y2, hat_x1:hat_x2, 3] / 255
        alpha = 1 - alpha_h
        
        for c in range(3):
            img[y1:y2, x1:x2, c] = alpha_h * hat[hat_y1:hat_y2, hat_x1:hat_x2, c] + alpha * img[y1:y2, x1:x2, c]
        return img
        


def main(fname):
    img = cv2.imread(fname)
    christmas_hat(img)
    cv2.imwrite(f'img/result/{fname.split("/")[-1]}', img) 

if __name__ == "__main__":
    main('img/test/test_1.jpg')
