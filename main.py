from autopilot.seeker import Seeker, Point
from autopilot.controller import Controller
import cv2
import numpy as np
import mss
import statistics

delta_sliding_window_size = 5
delta_sliding_window = []

def grab_screenshot():
    """
    Делаем скриншот
    :return:
    """
    with mss.mss() as m:
        return np.array(m.grab(m.monitors[2]))


cropbox = [Point(200,0), Point(1920, 1080)]
s = Seeker(cropbox, roi=(0.22, 0.8, 0.4, 0.55, 0.6, 0.55, 0.78, 0.8))
c = Controller("autopilot test", 46611, local_host='192.168.1.5', remote_host='192.168.1.8')
c.run()


while True:
    try:
        frame = grab_screenshot()  # получаем скриншот
        image, delta = s.process_frame(frame, show_data=True)  # получаем кадр с данными, отклонение от середины
        # Показываем как оно все есть
        cv2.namedWindow('frame', cv2.WINDOW_KEEPRATIO)
        cv2.imshow('frame', image)
        cv2.resizeWindow('frame', int(1280), int(720))
        # Усредняем дельту, что бы при резких изменениях ситуации руль не прыгал из стороны в сторону
        if delta is not None:
            if len(delta_sliding_window) == delta_sliding_window_size:
                delta_sliding_window.pop(0)

            delta_sliding_window.append(delta)
            c.steering = 0.5 + sum(delta_sliding_window) / len(delta_sliding_window) / 600
        else:
            if len(delta_sliding_window) == delta_sliding_window_size:
                delta_sliding_window.pop(0)

            delta_sliding_window.append(0)
            c.steering = 0.5
        if cv2.waitKey(1) == ord('q'):
            break
    except KeyboardInterrupt:
        break

cv2.destroyAllWindows()
del(c)
