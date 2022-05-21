import cv2
from regex import B

capture = cv2.VideoCapture('http://172.30.1.47:8080/video')

while True:
    _, frame = capture.read()
    cv2.imshow('livestream', frame)

    if cv2.waitKey(1) == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
