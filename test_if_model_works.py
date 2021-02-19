import cv2
import numpy as np
import os
import tensorflow as tf


def main():

    model = tf.keras.models.load_model('model/model_1.h5')

    capture = cv2.VideoCapture(0)

    while capture.isOpened():
        ret, frame = capture.read()

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eyes = cv2.CascadeClassifier('haarcascade_eye.xml').detectMultiScale(
            gray_frame, scaleFactor=1.2, minNeighbors=5)

        for (x, y, h, w) in eyes[:2]:
            roi = gray_frame[y: y + h, x: x + w]

            height, width = roi.shape
            roi = cv2.resize(roi, (64, 64), interpolation=cv2.INTER_AREA)
            roi = np.reshape(roi, (64, 64, 1))

            y_ratio = roi.shape[0] / height
            x_ratio = roi.shape[1] / width

            result = model.predict(np.array([roi]))[0]

            pupil = (int(round(result[0] / x_ratio) + x), int(round(result[1] / y_ratio) + y))
            cv2.circle(frame, pupil, 2, (255, 0, 0), 2)

        cv2.imshow("Video", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
