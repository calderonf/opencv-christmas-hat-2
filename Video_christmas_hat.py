"""
import pip
package_names=['opencv-python', 'opencv-contrib-python'] #packages to install
pip.main(['install'] + package_names + ['--upgrade'])
"""
import cv2
import random
import time
import platform
import math
from datetime import datetime
show_params = False
CAMERANUM = 1#2
FullScreen = True  
font = cv2.FONT_HERSHEY_SIMPLEX
color = (90, 90, 255)
color2 = (0, 255, 204)

FONT_SCALE = 2e-3  # Adjust for larger font size in all images
THICKNESS_SCALE = 1e-3  # Adjust for larger thickness in all images
TEXT_Y_OFFSET_SCALE = 1e-2  # Adjust for larger Y-offset of text and bounding box

def face_detect(img, cname):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_hist = cv2.equalizeHist(img_gray)
    face_cascade = cv2.CascadeClassifier(cname)
    faces = face_cascade.detectMultiScale(img_hist)
    return faces


def christmas_hat(img, cname='data/haarcascade_frontalface_alt.xml'):
    faces = face_detect(img, cname)
    hats = [cv2.imread(f'img/hats/Hat_puj_{i + 1}.png', -1) for i in [0]]
    for face in faces:
        hat = random.choice(hats)
        scale = face[3] / hat.shape[0]
        hat = cv2.resize(hat, (0, 0), fx=scale, fy=scale)
        x_offset = int(face[0] + (face[2] / 2) - (hat.shape[1] / 2))#+ (face[2] * 0.1)
        y_offset = int(face[1] - hat.shape[0]/1.5)

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


def put_frame(img):
    frame = cv2.imread('img/frames/frame_2.png', -1)
    x_offset = 160
    y_offset = 70
    tam = (frame.shape[1] - x_offset * 2, frame.shape[0] - y_offset * 2)

    img2 = cv2.resize(img, tam)

    x1 = x_offset
    x2 = img2.shape[1] + x_offset
    y1 = y_offset
    y2 = img2.shape[0] + y_offset

    alpha_h = frame[:, :, 3] / 255
    alpha = 1 - alpha_h
    #print("x1=", x1,"x2=", x2,"y1=", y1,"y2=", y2)
    #print("tam alpha", alpha.shape)
    #print("tam img2", img2.shape)
    #print("tam frame", frame.shape)
    for c in range(3):
        frame[y1:y2, x1:x2, c] = alpha[y1:y2, x1:x2] * img2[:, :, c] + alpha_h[y1:y2, x1:x2] * frame[y1:y2, x1:x2, c]
    return frame


def main():
    cap = cv2.VideoCapture(CAMERANUM)
    if platform.system() == "Linux":
        cap = cv2.VideoCapture(CAMERANUM)
    elif platform.system() == 'Windows':
        cap = cv2.VideoCapture(cv2.CAP_DSHOW + CAMERANUM)
    else:
        print("Unsupported OS")
        exit()
    if not cap.isOpened():
        print("ERROR! Unable to open camera")
    else:
        if FullScreen:
            cv2.namedWindow('Video', cv2.WINDOW_FREERATIO)
            cv2.setWindowProperty('Video', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.namedWindow("Video")
        prev_frame_time = time.time()
        time.sleep(0.03)
        new_frame_time = time.time()
        contarcuadros=0
        temporizador=False
        while True:
            _status, im = cap.read()
            if im is None:
                print("hardware error")
                break
            img = cv2.flip(im, +1)
            if show_params:
                new_frame_time = time.time()
                fps = 1 / (new_frame_time - prev_frame_time)
                prev_frame_time = new_frame_time
                fps = str(int(fps))
                cv2.putText(img, "FPS: {}".format(fps), (15, 80), font, 1.0, color)
            vis = christmas_hat(img)
            framed = put_frame(vis)
            
            if temporizador and contarcuadros>1:
                height, width, _ = framed.shape
                if int(contarcuadros/10)==0:
                    cv2.putText(framed, "Di TranSIStor", (int(framed.shape[1]/2)-170, int(framed.shape[0]/2)),
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                    fontScale=min(width, height) * FONT_SCALE,
                    thickness=math.ceil(min(width, height) * THICKNESS_SCALE),
                    color=color2)
                else:
                    cv2.putText(framed, "{}".format(int(contarcuadros/10)), (int(framed.shape[1]/2), int(framed.shape[0]/2)),
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                    fontScale=min(width, height) * FONT_SCALE,
                    thickness=math.ceil(min(width, height) * THICKNESS_SCALE),
                    color=color)

            cv2.imshow("Video", framed)
            k = cv2.waitKey(1)
            
            if k == 27:
                cv2.destroyAllWindows()
                break
            if k == 32:
                #print("espacio detectado")
                contarcuadros=39
                temporizador=True
            
            if contarcuadros <= 0 and temporizador:
                temporizador=False
                today = datetime.now()
                iso_date = str(today.isoformat()).replace("-","").replace(".","").replace(":","")
                filename='C:/Users/calderonf/Dropbox/navidad_2022_electronica/'+str(iso_date)+'.jpg'
                #print("Guardando en ",filename)
                cv2.imwrite(filename,framed)
                cv2.imshow("Video", cv2.bitwise_not(framed))
                k = cv2.waitKey(1)
            contarcuadros = contarcuadros-1


if __name__ == "__main__":
    main()
