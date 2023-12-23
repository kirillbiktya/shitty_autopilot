from typing import List, Tuple, Union, Any
import cv2
import numpy as np
import statistics

SLIDING_LINE_WINDOW = 5  # интервал усреднения положения линий
DELTA_IF_ONE_OF_LINES_WASNT_DETECTED = 100

LEFT_LINE_DETECTED_COLOR = (0, 255, 0)
RIGHT_LINE_DETECTED_COLOR = (0, 0, 255)
FILTERED_OUT_LINES_DETECTED_COLOR = (255, 255, 255)
LEFT_LINE_BORDER_COLOR = (255, 255, 0)
RIGHT_LINE_BORDER_COLOR = (0, 255, 255)
ROI_BORDERS_COLOR = (255, 0, 0)
CAR_DIRECTION_LINE = (110, 0, 180)
GUESSED_LANE_CENTER_LINE = (180, 0, 110)


class Point:
    """
    Класс, определяющий точку. Имеет как отдельные координаты x, y, так и кортеж координат.
    """
    def __init__(self, x, y, none=False):
        self.x = int(x)  # округление втупую: отбрасываем дробную часть float-а, получаем int
        self.y = int(y)  # округление втупую: отбрасываем дробную часть float-а, получаем int

    def __sub__(self, other):
        """
        Перегрузка оператора вычитания
        :param other:
        :return:
        """
        return Point(self.x - other.x, self.y - other.y)

    def __repr__(self):
        """
        Просто, что бы красиво отображать объекты функцией print()
        :return:
        """
        return "Point({}, {})".format(self.x, self.y)

    @property
    def pt(self):
        """
        Кортеж с координатами точки
        :return:
        """
        return self.x, self.y


class Line:
    """
    Класс, определяющий линию. Имеет две точки, производную
    """
    def __init__(self, x1, y1, x2, y2):
        """
        Для операций с линиями необходимо, что бы был определенный порядок начала-конца линии. Выбор начала обусловлен
        координатой x1
        :param x1:
        :param y1:
        :param x2:
        :param y2:
        """
        if x1 < x2:
            self.x1 = int(x1)
            self.x2 = int(x2)
            self.y1 = int(y1)
            self.y2 = int(y2)
        else:
            self.x2 = int(x1)
            self.x1 = int(x2)
            self.y2 = int(y1)
            self.y1 = int(y2)

    @property
    def slope(self):
        """
        Производная линии
        :return:
        """
        try:
            return (self.y2 - self.y1) / (self.x2 - self.x1)
        except ZeroDivisionError:
            return 0

    @property
    def pt1(self):
        """
        Начало линии (кортеж с координатами точки)
        :return:
        """
        return self.x1, self.y1

    @property
    def pt2(self):
        """
        Конец линии (кортеж с координатами точки)
        :return:
        """
        return self.x2, self.y2


class Seeker:
    """
    Класс, определяющий границы полосы движения. Основан на алгоритмах
    https://ru.wikipedia.org/wiki/%D0%9E%D0%BF%D0%B5%D1%80%D0%B0%D1%82%D0%BE%D1%80_%D0%9A%D1%8D%D0%BD%D0%BD%D0%B8
    https://ru.wikipedia.org/wiki/%D0%9F%D1%80%D0%B5%D0%BE%D0%B1%D1%80%D0%B0%D0%B7%D0%BE%D0%B2%D0%B0%D0%BD%D0%B8%D0%B5_%D0%A5%D0%B0%D1%84%D0%B0
    """
    def __init__(
            self,
            image_cropbox: List[Point],
            roi: Tuple[float] = (0.2, 0.95, 0.45, 0.6, 0.55, 0.6, 0.8, 0.95),
            rho: int = 2,
            theta: float = np.pi / 180,
            threshold: int = 15,
            min_line_length: int = 40,
            max_line_gap: int = 25
    ):
        """

        :param image_cropbox: если необходимо обрезать изображение
        :param roi: зона интереса (часть изображения, в которой будут определяться линии, задается коэффициентами от
            ширины и высоты изображения)
        :param rho: параметр алгоритма Хафа, точность определения точек
        :param theta: параметр алгоритма Хафа, угол тета
        :param threshold: параметр алгоритма Хафа, уровень чувствительности
        :param min_line_length: параметр алгоритма Хафа, минимальная длина линии
        :param max_line_gap: параметр алгоритма Хафа, максимальный разрыв между сегментами линии
        """
        self._crop_vertices: List[Point] = image_cropbox
        self._original_image_size: Point = Point(0, 0)
        self._roi_vertices: List[Point] = self._create_roi(*self._get_cropped_image_size().pt, *roi)
        self._rho: int = rho
        self._theta: float = theta
        self._threshold: int = threshold
        self._min_line_length: int = min_line_length
        self._max_line_gap: int = max_line_gap

        self._left_sliding_line = []
        self._right_sliding_line = []

    def _crop_image(self, image):
        """
        Обрезаем изображение по кропбоксу
        :param image:
        :return:
        """
        height, width = image.shape[:2]
        self._original_image_size = Point(width, height)
        return image[
               self._crop_vertices[0].y:self._crop_vertices[1].y,
               self._crop_vertices[0].x:self._crop_vertices[1].x
               ]

    def _get_cropped_image_size(self):
        return self._crop_vertices[1] - self._crop_vertices[0]

    @staticmethod
    def _create_roi(image_width, image_height, x1, y1, x2, y2, x3, y3, x4, y4):
        """
        Возвращает массив точек, указывающих на вершины четырехугольной зоны интереса
        :param image_width:
        :param image_height:
        :param x1:
        :param y1:
        :param x2:
        :param y2:
        :param x3:
        :param y3:
        :param x4:
        :param y4:
        :return:
        """
        return [
                Point(image_width * x1, image_height * y1),
                Point(image_width * x2, image_height * y2),
                Point(image_width * x3, image_height * y3),
                Point(image_width * x4, image_height * y4),
            ]

    def _get_roi(self):
        return [x.pt for x in self._roi_vertices]

    def _resize_line_image(self, line_image):
        """
        Дополняет изображение с линиями до размеров исходного кадра
        :param line_image:
        :return:
        """
        im = np.zeros((self._original_image_size.y, self._original_image_size.x, 4))
        im[self._crop_vertices[0].y:self._crop_vertices[1].y,
        self._crop_vertices[0].x:self._crop_vertices[1].x] = line_image
        return im

    @staticmethod
    def _average_line(line_list):
        """
        Возвращает усредненную линию, полученную из массива линий
        :param line_list:
        :return:
        """
        return Line(
                statistics.mean([l.x1 for l in line_list]),
                statistics.mean([l.y1 for l in line_list]),
                statistics.mean([l.x2 for l in line_list]),
                statistics.mean([l.y2 for l in line_list])
            )

    def _calc_average_line(self, xs, ys, is_left) -> Union[Line, None]:
        """
        Вычисляет усредненную линию, используя скользящее среднее каждой координаты
        :param xs:
        :param ys:
        :param is_left:
        :return:
        """
        try:
            mean_x = statistics.mean(xs)
            mean_y = statistics.mean(ys)
            xs_gt_mean = list(filter(lambda x: x > mean_x, xs))
            ys_gt_mean = list(filter(lambda y: y > mean_y, ys))
            xs_lt_mean = list(filter(lambda x: x < mean_x, xs))
            ys_lt_mean = list(filter(lambda y: y < mean_y, ys))
            if is_left:
                if len(self._left_sliding_line) == SLIDING_LINE_WINDOW:
                    self._left_sliding_line.pop(0)
                self._left_sliding_line.append(
                    Line(
                        statistics.mean(xs_gt_mean),
                        statistics.mean(ys_lt_mean),
                        statistics.mean(xs_lt_mean),
                        statistics.mean(ys_gt_mean)
                    )
                )
                line = self._average_line(self._left_sliding_line)
            else:
                if len(self._right_sliding_line) == SLIDING_LINE_WINDOW:
                    self._right_sliding_line.pop(0)
                self._right_sliding_line.append(
                    Line(
                        statistics.mean(xs_lt_mean),
                        statistics.mean(ys_lt_mean),
                        statistics.mean(xs_gt_mean),
                        statistics.mean(ys_gt_mean)
                    )
                )
                line = self._average_line(self._right_sliding_line)
            return line
        except statistics.StatisticsError:
            return None

    def _detect_lines(self, image) -> Tuple[Any, Union[Line, None], Union[Line, None]]:
        """
        Основная функция, в которой происходит поиск линий и все вычисления
        :param image:
        :return:
        """
        image = self._crop_image(image)  # обрезаем изображение
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # так надо
        blur = cv2.GaussianBlur(gray, (5, 5), 0)  # применяем гауссово размытие
        edges = cv2.Canny(blur, 50, 150)  # Canny edges detection
        mask_color = 255
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, np.array([self._get_roi()], dtype=np.int32), mask_color)
        masked_edges = cv2.bitwise_and(edges, mask)
        lines = cv2.HoughLinesP(masked_edges, rho=self._rho, theta=self._theta, threshold=self._threshold,
                                minLineLength=self._min_line_length, maxLineGap=self._max_line_gap)  # алгоритм Хафа

        line_image = np.zeros_like(image)  # Пустое изображение для визуализации

        # Накладываем зону интереса
        cv2.line(line_image, self._roi_vertices[0].pt, self._roi_vertices[1].pt, ROI_BORDERS_COLOR, 1)
        cv2.line(line_image, self._roi_vertices[1].pt, self._roi_vertices[2].pt, ROI_BORDERS_COLOR, 1)
        cv2.line(line_image, self._roi_vertices[2].pt, self._roi_vertices[3].pt, ROI_BORDERS_COLOR, 1)
        cv2.line(line_image, self._roi_vertices[3].pt, self._roi_vertices[0].pt, ROI_BORDERS_COLOR, 1)

        if lines is not None:
            left_lines = []
            right_lines = []

            for line in lines:
                _l = Line(*line[0])

                if abs(_l.slope) > 0.55:  # отсекаем горизонтальные и малонаклоненные линии
                    center = (self._crop_vertices[1] - self._crop_vertices[0]).x / 2
                    if _l.slope > 0:  # если производная больше нуля, значит, линия должна быть похожа на правую линию разметки
                        if _l.x1 > center and _l.x2 > center:  # но если линия находится левее центра изображения - она нас не интересует
                            cv2.line(line_image, _l.pt1, _l.pt2, RIGHT_LINE_DETECTED_COLOR, 1)  # рисуем линию
                            right_lines.append(_l)
                    else:
                        if _l.x1 < center and _l.x2 < center:
                            cv2.line(line_image, _l.pt1, _l.pt2, LEFT_LINE_DETECTED_COLOR, 1)
                            left_lines.append(_l)
                else:
                    # cv2.line(line_image, _l.pt1, _l.pt2, FILTERED_OUT_LINES_DETECTED_COLOR, 1)
                    pass

            left_line = None
            right_line = None

            # Вычисляем усредненные левую и правую линии, отражающие найденную разметку на дороге
            if len(left_lines) > 0:
                left_xs = [x.x1 for x in left_lines]
                left_xs.extend([x.x2 for x in left_lines])
                left_ys = [x.y1 for x in left_lines]
                left_ys.extend([x.y2 for x in left_lines])
                left_line = self._calc_average_line(left_xs, left_ys, True)
                if left_line is not None:
                    cv2.line(line_image, left_line.pt1, left_line.pt2, LEFT_LINE_BORDER_COLOR, 3)
            if len(right_lines) > 0:
                right_xs = [x.x1 for x in right_lines]
                right_xs.extend([x.x2 for x in right_lines])
                right_ys = [x.y1 for x in right_lines]
                right_ys.extend([x.y2 for x in right_lines])
                right_line = self._calc_average_line(right_xs, right_ys, False)
                if right_line is not None:
                    cv2.line(line_image, right_line.pt1, right_line.pt2, RIGHT_LINE_BORDER_COLOR, 3)

            return line_image, left_line, right_line

        return line_image, None, None

    def _direction_overlay(self, delta, line_image):
        """
        Две линии, указывающие реальное направление движения (всегда вперед), и желаемое
        :param delta: отклонение от центра изображения
        :param line_image:
        :return:
        """
        overlay = np.zeros_like(line_image)
        car_direction_line = Line(
            self._original_image_size.x / 2,
            self._original_image_size.y,
            self._original_image_size.x / 2,
            self._original_image_size.y / 2
        )
        cv2.line(overlay, car_direction_line.pt1, car_direction_line.pt2, CAR_DIRECTION_LINE, 2)
        if delta is not None:
            guessed_drive_line_center = Line(
                self._original_image_size.x / 2,
                self._original_image_size.y,
                self._original_image_size.x / 2 - delta,
                self._original_image_size.y / 2
            )
            cv2.line(overlay, guessed_drive_line_center.pt1, guessed_drive_line_center.pt2, GUESSED_LANE_CENTER_LINE, 2)
        return overlay

    def process_frame(self, image, show_data=False):
        """
        Основной метод, с которым взаимодействует пользователь
        :param image:
        :param show_data:
        :return:
        """
        line_image, left_line, right_line = self._detect_lines(image)
        delta = None
        if left_line is not None and right_line is not None:  # если определены обе линии, считаем средний x между их средними координатами x
            average_x_of_left = int((left_line.x1 + left_line.x2) / 2)
            average_x_of_right = int((right_line.x1 + right_line.x2) / 2)
            guessed_center = int((average_x_of_right + average_x_of_left) / 2)
            # guessed_center = int((left_line.x1 + right_line.x2) / 2)
            frame_center = line_image.shape[:2][1] / 2
            delta = frame_center - guessed_center
        else:  # иначе, применяем ЭМПИРИЧЕСКУЮ величину отклонения
            if left_line is not None and right_line is None:
                delta = -DELTA_IF_ONE_OF_LINES_WASNT_DETECTED
            elif right_line is not None and left_line is None:
                delta = DELTA_IF_ONE_OF_LINES_WASNT_DETECTED
            else:
                delta = None
        if show_data:  # рисовать нам данные поверх кадра?
            directions_image = self._direction_overlay(delta, line_image)
            _image = cv2.addWeighted(directions_image, 1, line_image, 1, 0, dtype=0)
            final_image = image # cv2.addWeighted(image, 0.6, _image, 1, 0, dtype=0)
            box = final_image[self._crop_vertices[0].y:self._crop_vertices[1].y, self._crop_vertices[0].x:self._crop_vertices[1].x]
            final_image = cv2.addWeighted(box, 1, _image, 1, 0, dtype=0)
        else:
            final_image = image
        return final_image, delta
