import sys
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import cv2 as cv
import matplotlib.pyplot as plt
import warnings

import image_processing as img_pr


class Ui_MainWindow:
    def __init__(self):
        self.original_img = None
        self.edges = None
        self.lines = None
        self.triangles = None
        self.img_after_segmetation = None
        self.circles = None
        self.labels = None

    def setupUi(self, MainWindow):
        self.window = MainWindow

        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1380, 844)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(0, 0, 271, 821))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.verticalLayout.setContentsMargins(0, 5, 0, 0)
        self.verticalLayout.setSpacing(7)
        self.verticalLayout.setObjectName("verticalLayout")
        self.load_btn = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.load_btn.setObjectName("load_btn")
        self.verticalLayout.addWidget(self.load_btn)
        self.line1 = QtWidgets.QFrame(self.verticalLayoutWidget)
        self.line1.setFrameShadow(QtWidgets.QFrame.Plain)
        self.line1.setLineWidth(3)
        self.line1.setFrameShape(QtWidgets.QFrame.HLine)
        self.line1.setObjectName("line1")
        self.verticalLayout.addWidget(self.line1)
        self.canny_btn = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.canny_btn.setObjectName("canny_btn")
        self.verticalLayout.addWidget(self.canny_btn)
        self.canny_thresh1_spin = QtWidgets.QSpinBox(self.verticalLayoutWidget)
        self.canny_thresh1_spin.setEnabled(True)
        self.canny_thresh1_spin.setAlignment(QtCore.Qt.AlignCenter)
        self.canny_thresh1_spin.setSpecialValueText("")
        self.canny_thresh1_spin.setMaximum(500)
        self.canny_thresh1_spin.setProperty("value", 100)
        self.canny_thresh1_spin.setObjectName("canny_thresh1_spin")
        self.verticalLayout.addWidget(self.canny_thresh1_spin)
        self.canny_thresh2_spin = QtWidgets.QSpinBox(self.verticalLayoutWidget)
        self.canny_thresh2_spin.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.canny_thresh2_spin.setAlignment(QtCore.Qt.AlignCenter)
        self.canny_thresh2_spin.setMinimum(0)
        self.canny_thresh2_spin.setMaximum(500)
        self.canny_thresh2_spin.setProperty("value", 200)
        self.canny_thresh2_spin.setObjectName("canny_thresh2_spin")
        self.verticalLayout.addWidget(self.canny_thresh2_spin)
        self.line3 = QtWidgets.QFrame(self.verticalLayoutWidget)
        self.line3.setFrameShadow(QtWidgets.QFrame.Plain)
        self.line3.setLineWidth(3)
        self.line3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line3.setObjectName("line3")
        self.verticalLayout.addWidget(self.line3)
        self.hough_lines_btn = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.hough_lines_btn.setObjectName("hough_lines_btn")
        self.verticalLayout.addWidget(self.hough_lines_btn)
        self.hough_lines_votes_spin = QtWidgets.QSpinBox(self.verticalLayoutWidget)
        self.hough_lines_votes_spin.setAlignment(QtCore.Qt.AlignCenter)
        self.hough_lines_votes_spin.setMinimum(1)
        self.hough_lines_votes_spin.setMaximum(500)
        self.hough_lines_votes_spin.setProperty("value", 80)
        self.hough_lines_votes_spin.setObjectName("hough_lines_votes_spin")
        self.verticalLayout.addWidget(self.hough_lines_votes_spin)
        self.line4_2 = QtWidgets.QFrame(self.verticalLayoutWidget)
        self.line4_2.setFrameShadow(QtWidgets.QFrame.Plain)
        self.line4_2.setLineWidth(3)
        self.line4_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line4_2.setObjectName("line4_2")
        self.verticalLayout.addWidget(self.line4_2)
        self.segment_btn = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.segment_btn.setObjectName("segment_btn")
        self.verticalLayout.addWidget(self.segment_btn)
        self.line4 = QtWidgets.QFrame(self.verticalLayoutWidget)
        self.line4.setFrameShadow(QtWidgets.QFrame.Plain)
        self.line4.setLineWidth(3)
        self.line4.setFrameShape(QtWidgets.QFrame.HLine)
        self.line4.setObjectName("line4")
        self.verticalLayout.addWidget(self.line4)
        self.hough_circles_btn = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.hough_circles_btn.setObjectName("hough_circles_btn")
        self.verticalLayout.addWidget(self.hough_circles_btn)
        self.hough_circles_min_dist_spin = QtWidgets.QSpinBox(self.verticalLayoutWidget)
        self.hough_circles_min_dist_spin.setAlignment(QtCore.Qt.AlignCenter)
        self.hough_circles_min_dist_spin.setMinimum(0)
        self.hough_circles_min_dist_spin.setMaximum(150)
        self.hough_circles_min_dist_spin.setProperty("value", 6)
        self.hough_circles_min_dist_spin.setObjectName("hough_circles_min_dist_spin")
        self.verticalLayout.addWidget(self.hough_circles_min_dist_spin)
        self.hough_circles_votes_spin = QtWidgets.QSpinBox(self.verticalLayoutWidget)
        self.hough_circles_votes_spin.setAlignment(QtCore.Qt.AlignCenter)
        self.hough_circles_votes_spin.setMinimum(1)
        self.hough_circles_votes_spin.setMaximum(500)
        self.hough_circles_votes_spin.setProperty("value", 4)
        self.hough_circles_votes_spin.setObjectName("hough_circles_votes_spin")
        self.verticalLayout.addWidget(self.hough_circles_votes_spin)
        self.hough_circles_min_radius_spin = QtWidgets.QSpinBox(self.verticalLayoutWidget)
        self.hough_circles_min_radius_spin.setAlignment(QtCore.Qt.AlignCenter)
        self.hough_circles_min_radius_spin.setObjectName("hough_circles_min_radius_spin")
        self.verticalLayout.addWidget(self.hough_circles_min_radius_spin)
        self.hough_circles_max_radius_spin = QtWidgets.QSpinBox(self.verticalLayoutWidget)
        self.hough_circles_max_radius_spin.setAlignment(QtCore.Qt.AlignCenter)
        self.hough_circles_max_radius_spin.setMinimum(1)
        self.hough_circles_max_radius_spin.setProperty("value", 5)
        self.hough_circles_max_radius_spin.setObjectName("hough_circles_max_radius_spin")
        self.verticalLayout.addWidget(self.hough_circles_max_radius_spin)
        self.classify_indentation_spin = QtWidgets.QSpinBox(self.verticalLayoutWidget)
        self.classify_indentation_spin.setAlignment(QtCore.Qt.AlignCenter)
        self.classify_indentation_spin.setProperty("value", 3)
        self.classify_indentation_spin.setObjectName("classify_indentation_spin")
        self.verticalLayout.addWidget(self.classify_indentation_spin)
        self.classify_n_ignore_points_spin = QtWidgets.QSpinBox(self.verticalLayoutWidget)
        self.classify_n_ignore_points_spin.setAlignment(QtCore.Qt.AlignCenter)
        self.classify_n_ignore_points_spin.setProperty("value", 1)
        self.classify_n_ignore_points_spin.setObjectName("classify_n_ignore_points_spin")
        self.verticalLayout.addWidget(self.classify_n_ignore_points_spin)
        self.classify_btn = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.classify_btn.setObjectName("classify_btn")
        self.verticalLayout.addWidget(self.classify_btn)
        self.v_line_user_input = QtWidgets.QFrame(self.centralwidget)
        self.v_line_user_input.setGeometry(QtCore.QRect(261, 2, 20, 821))
        self.v_line_user_input.setFrameShadow(QtWidgets.QFrame.Plain)
        self.v_line_user_input.setLineWidth(3)
        self.v_line_user_input.setFrameShape(QtWidgets.QFrame.VLine)
        self.v_line_user_input.setObjectName("v_line_user_input")
        self.h_line_user_input = QtWidgets.QFrame(self.centralwidget)
        self.h_line_user_input.setGeometry(QtCore.QRect(0, 813, 271, 16))
        self.h_line_user_input.setFrameShadow(QtWidgets.QFrame.Plain)
        self.h_line_user_input.setLineWidth(3)
        self.h_line_user_input.setMidLineWidth(0)
        self.h_line_user_input.setFrameShape(QtWidgets.QFrame.HLine)
        self.h_line_user_input.setObjectName("h_line_user_input")
        self.image_label = QtWidgets.QLabel(self.centralwidget)
        self.image_label.setGeometry(QtCore.QRect(275, 12, 1094, 820))
        self.image_label.setText("")
        self.image_label.setPixmap(QtGui.QPixmap("pictures/3.jpg"))
        self.image_label.setScaledContents(True)
        self.image_label.setObjectName("image_label")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.load_btn.clicked.connect(self.load_img)
        self.canny_btn.clicked.connect(self.canny)
        self.hough_lines_btn.clicked.connect(self.hough_lines)
        self.segment_btn.clicked.connect(self.make_segmentation)
        self.hough_circles_btn.clicked.connect(self.hough_circles_and_classify)
        self.classify_btn.clicked.connect(self.get_res)

        MainWindow.setWindowIcon(QIcon('etc/icon.png'))
        self.apply_stylesheet()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Тримино"))
        self.load_btn.setText(_translate("MainWindow", "Загрузить изображение"))
        self.canny_btn.setText(_translate("MainWindow", "Применить оператор Кэнни"))
        self.canny_thresh1_spin.setPrefix(_translate("MainWindow", "Порог 1: "))
        self.canny_thresh2_spin.setPrefix(_translate("MainWindow", "Порог 2: "))
        self.hough_lines_btn.setText(_translate("MainWindow", "Применить преобразование\nХафа для прямых"))
        self.hough_lines_votes_spin.setPrefix(_translate("MainWindow", "Порог голосов: "))
        self.segment_btn.setText(_translate("MainWindow", "Сегментировать фишки"))
        self.hough_circles_btn.setText(_translate("MainWindow", "Применить преобразование\nХафа для окружностей"))
        self.hough_circles_min_dist_spin.setPrefix(_translate("MainWindow", "Минимальное расстояние: "))
        self.hough_circles_votes_spin.setPrefix(_translate("MainWindow", "Порог голосов: "))
        self.hough_circles_min_radius_spin.setPrefix(_translate("MainWindow", "Минимальный радиус: "))
        self.hough_circles_max_radius_spin.setPrefix(_translate("MainWindow", "Максимальный радиус: "))
        self.classify_indentation_spin.setPrefix(_translate("MainWindow", "Отступ от сторон: "))
        self.classify_n_ignore_points_spin.setPrefix(_translate("MainWindow", "Порог игнорирования: "))
        self.classify_btn.setText(_translate("MainWindow", "Классифицировать фишки"))

    def _make_error_window(self, text="Произошла неизвестная ошибка"):
        error_mes = QMessageBox()
        error_mes.setWindowTitle("Ошибка")
        error_mes.setText(text)
        error_mes.setIcon(QMessageBox.Warning)
        error_mes.setStandardButtons(QMessageBox.Ok)
        error_mes.setWindowIcon(QIcon('etc/icon.png'))
        error_mes.exec_()

    def draw_img(self, img):
        is_black_and_white = (len(img.shape) == 2) 
        if is_black_and_white:
            img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)

        height, width, _ = img.shape
        bytesPerLine = 3 * width
        q_img = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(pixmap)

    def load_img(self):
        try:
            file_dialog = QFileDialog()
            file_path, _ = file_dialog.getOpenFileName(self.window,
                                                    "Выберите изображение", "", "Images (*.png *.jpg *.bmp)")
            if file_path:
                self.original_img = plt.imread(file_path)
                self.draw_img(self.original_img)

            self.edges = None
            self.lines = None
            self.triangles = None
            self.img_after_segmetation = None
            self.circles = None
            self.labels = None

        except:
            self._make_error_window()

    def canny(self):
        try:
            if self.original_img is None:
                self._make_error_window("Сначала нужно загрузить изображение.")
                return
            thresh1 = self.canny_thresh1_spin.value()
            thresh2 = self.canny_thresh2_spin.value()
            self.edges = cv.Canny(self.original_img, thresh1, thresh2)
            self.draw_img(self.edges)

            self.lines = None
            self.triangles = None
            self.img_after_segmetation = None
            self.circles = None
            self.labels = None
        except:
            self._make_error_window()

    def hough_lines(self):
        try:
            if self.edges is None:
                self._make_error_window("Сначала нужно применить оператор Кэнни.")
                return
            votes_thresh = self.hough_lines_votes_spin.value()
            self.lines = cv.HoughLines(self.edges, 1, np.pi/180 / 2, votes_thresh)

            if self.lines is None:
                self._make_error_window("Подберите другие параметры, прямые отсутствуют.")
                return
            
            # drawing lines
            img = self.original_img.copy()
            for line in self.lines:
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 700 * (-b))
                y1 = int(y0 + 700 * (a))
                x2 = int(x0 - 1100 * (-b))
                y2 = int(y0 - 1100 * (a))
                cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)

            self.draw_img(img)

            self.triangles = None
            self.img_after_segmetation = None
            self.circles = None
            self.labels = None
        except:
            self._make_error_window()

    def make_segmentation(self):
        try:
            if self.lines is None:
                self._make_error_window("Сначала нужно применить преобразование Хафа для прямых.")
                return
            
            gray_image = cv.cvtColor(self.original_img, cv.COLOR_RGB2GRAY)
            self.triangles = []
            for line in self.lines:
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 700 * (-b))
                y1 = int(y0 + 700 * (a))
                x2 = int(x0 - 1100 * (-b))
                y2 = int(y0 - 1100 * (a))

                segs_borders = img_pr.get_segs_borders(x1, y1, x2, y2,
                                                       self.edges,
                                                       min_seg_len=65,
                                                       max_gap=3,
                                                       radius=0,
                                                       max_seg_len=95)
                for p0, p1 in segs_borders:
                    self.triangles.append(img_pr.Triangle(p0, p1, gray_image))

            self.triangles = img_pr.get_unique_triangles(self.triangles)
            self.img_after_segmetation = self.original_img.copy()
            for i, triangle in enumerate(self.triangles):
                direction = triangle.direction
                if direction == "top":
                    color = (255, 0, 0)
                elif direction == "bottom":
                    color = (0, 255, 0)
                elif direction == "left":
                    color = (0, 0, 255)
                elif direction == "right":
                    color = (155,38,182) # Violet
                else:
                    raise RuntimeError
                triangle.draw(self.img_after_segmetation, sides_color=color,
                              center_color=color, main_side_thickness=3)
                cv.putText(self.img_after_segmetation, str(i), org=triangle.center,
                        fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                        color=(255, 255, 255), thickness=2, lineType=cv.LINE_AA)

            self.draw_img(self.img_after_segmetation)

            self.circles = None
            self.labels = None
        except:
            self._make_error_window()

    def hough_circles_and_classify(self):
        try:
            if self.img_after_segmetation is None:
                self._make_error_window("Сначала нужно сегментировать фишки.")
                return
            
            min_dist = self.hough_circles_min_dist_spin.value()
            votes = self.hough_circles_votes_spin.value()
            min_radius = self.hough_circles_min_radius_spin.value()
            max_radius = self.hough_circles_max_radius_spin.value()

            self.circles = cv.HoughCircles(self.edges, cv.HOUGH_GRADIENT, 1, min_dist,
                                           param1=40, param2=votes,
                                           minRadius=min_radius, maxRadius=max_radius)
            
            if self.circles is None:
                self._make_error_window("Подберите другие параметры, окружности отсутствуют.")
                return
            
            self.circles = np.uint16(np.around(self.circles))

            img = self.img_after_segmetation.copy()

            indentation = self.classify_indentation_spin.value()
            n_ignore_points = self.classify_n_ignore_points_spin.value()
            self.labels = img_pr.classify(self.triangles, self.circles, self.original_img, img_pr.ColorClf(),
                                     return_logits=False,
                                     indentation=indentation,
                                     radius=1,
                                     n_ignore_points=n_ignore_points,
                                     draw_circles=True,
                                     img_to_draw=img)

            self.draw_img(img)
        except:
            self._make_error_window()

    def get_res(self):
        try:
            if self.labels is None:
                self._make_error_window("Сначала нужно применить преобразование Хафа для окружностей.")
                return
            
            self.draw_img(self.img_after_segmetation)
            self._make_res_window()
            
        except:
            self._make_error_window()

    def _make_res_window(self):
        try:
            res_mes = QMessageBox()
            res_mes.setWindowTitle("Результат")

            strings = []

            for i, (trianlge, label) in enumerate(zip(self.triangles, self.labels)):
                s = f"Треугольник {i}: {trianlge.center[0]}, {trianlge.center[1]}; {label[0]}, {label[1]}, {label[2]}"
                strings.append(s)

            text = "\n".join(strings)
            res_mes.setText(text)
            res_mes.setIcon(QMessageBox.Information)
            res_mes.setWindowIcon(QIcon('etc/icon.png'))
            res_mes.setStyleSheet(
                "QMessageBox { font-size: 13pt; font-family: Arial; } QPushButton { font-size: 12pt; }"
            )

            # Создаем кнопку "Сохранить в файл"
            save_button = res_mes.addButton("Сохранить в файл", QMessageBox.ActionRole)

            # Стандартная кнопка OK
            res_mes.setStandardButtons(QMessageBox.Ok)

            # Привязываем сигнал к слоту
            save_button.clicked.connect(self._make_save_to_file_window)

            res_mes.exec_()
        except:
            self._make_error_window()

    def _make_save_to_file_window(self):
        try:
            strings = [str(len(self.triangles))]
            for trianlge, label in zip(self.triangles, self.labels):
                s = f"{trianlge.center[0]}, {trianlge.center[1]}; {label[0]}, {label[1]}, {label[2]}"
                strings.append(s)

            text = "\n".join(strings)

            file_path, _ = QFileDialog.getSaveFileName(self.window, "Сохранить результат", "", "Text Files (*.txt)")
            if file_path:
                if not file_path.endswith('.txt'):
                    file_path += '.txt'
                with open(file_path, 'w') as f:
                    f.write(text)

                #QMessageBox.information(self.window, "Сохранение", "Файл успешно сохранен!", QMessageBox.Ok)
                saved_mes = QMessageBox()
                saved_mes.setWindowTitle("Сохранение")
                saved_mes.setText("Файл успешно сохранен!")
                saved_mes.setIcon(QMessageBox.Information)
                saved_mes.setWindowIcon(QIcon('etc/icon.png'))
                saved_mes.setStandardButtons(QMessageBox.Ok)
                saved_mes.exec_()
        except:
            self._make_error_window()

    def apply_stylesheet(self):
        try:
            style_file = "styles.css"
            with open(style_file, "r", encoding="utf-8") as f:
                style = f.read()
                self.window.setStyleSheet(style)
        except:
            warnings.warn("styles error")
            

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
