import sys
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import cv2 as cv
import matplotlib.pyplot as plt
import warnings

import image_processing as img_pr

class ImageLabel(QtWidgets.QLabel):
    def __init__(self, param):
        super().__init__(param)
        self.setMouseTracking(True)
        self.selection_start = None
        self.selection_end = None
        self.selections = []
        self.original_pixmap = None

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.selection_start = event.pos()
            self.selection_end = event.pos()

    def mouseMoveEvent(self, event):
        if self.selection_start is not None:
            self.selection_end = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            if self.selection_start is not None:
                self.selections.append((self.selection_start, self.selection_end))
                self.selection_start = None
                self.selection_end = None
                self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        painter.setPen(QtGui.QPen(QtCore.Qt.red, 2, QtCore.Qt.SolidLine))

        for selection in self.selections:
            start, end = selection
            rect = QtCore.QRect(start, end)
            painter.drawRect(rect.normalized())

        if self.selection_start is not None and self.selection_end is not None:
            rect = QtCore.QRect(self.selection_start, self.selection_end)
            painter.drawRect(rect.normalized())

    def save_selections(self):
        if self.selections and self.original_pixmap:
            for i, selection in enumerate(self.selections):
                selected_rect = self.get_selected_rect_relative_to_original_image(selection)
                selected_pixmap = self.original_pixmap.copy(selected_rect)
                selected_image = selected_pixmap.toImage()

                # Преобразование изображения в монохромное
                selected_image = selected_image.convertToFormat(QImage.Format_Mono)
                
                selected_image.save(f"etc/selected_fragment_{i}.bmp")

        return len(self.selections)

    def get_selected_rect_relative_to_original_image(self, selection):
        selection_rect = QtCore.QRect(selection[0], selection[1]).normalized()
        selected_rect = QtCore.QRect(
            selection_rect.topLeft().x() * self.original_pixmap.width() / self.width(),
            selection_rect.topLeft().y() * self.original_pixmap.height() / self.height(),
            selection_rect.width() * self.original_pixmap.width() / self.width(),
            selection_rect.height() * self.original_pixmap.height() / self.height()
        )

        return selected_rect

    def get_selections_as_arrays(self):
        selections_as_arrays = []
        if self.selections and self.original_pixmap:
            for selection in self.selections:
                selected_rect = self.get_selected_rect_relative_to_original_image(selection)
                selected_pixmap = self.original_pixmap.copy(selected_rect)
                selected_image = selected_pixmap.toImage()
                width = selected_image.width()
                height = selected_image.height()
                ptr = selected_image.bits()
                ptr.setsize(width * height)
                arr = np.array(ptr).reshape(height, width)  # черно-белое изображение
                selections_as_arrays.append(arr)
        return selections_as_arrays
    
    def set_original_pixmap(self, pixmap):
        self.original_pixmap = pixmap

    def clear_selections(self):
        self.selections.clear()
        self.update()


class Ui_MainWindow:
    def __init__(self):
        self.original_img = None
        self.gray_img = None
        self.black_and_white_img = None

    def setupUi(self, MainWindow):
        self.window = MainWindow

        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1547, 838)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(0, 2, 265, 731))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setSpacing(7)
        self.verticalLayout.setObjectName("verticalLayout")
        self.load_btn = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.load_btn.setObjectName("load_btn")
        self.verticalLayout.addWidget(self.load_btn)
        self.line0 = QtWidgets.QFrame(self.verticalLayoutWidget)
        self.line0.setFrameShadow(QtWidgets.QFrame.Plain)
        self.line0.setLineWidth(3)
        self.line0.setFrameShape(QtWidgets.QFrame.HLine)
        self.line0.setObjectName("line0")
        self.verticalLayout.addWidget(self.line0)
        self.get_gray_btn = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.get_gray_btn.setObjectName("get_gray_btn")
        self.verticalLayout.addWidget(self.get_gray_btn)
        self.line1 = QtWidgets.QFrame(self.verticalLayoutWidget)
        self.line1.setFrameShadow(QtWidgets.QFrame.Plain)
        self.line1.setLineWidth(3)
        self.line1.setFrameShape(QtWidgets.QFrame.HLine)
        self.line1.setObjectName("line1")
        self.verticalLayout.addWidget(self.line1)
        self.bin_magic_btn = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.bin_magic_btn.setObjectName("bin_magic_btn")
        self.verticalLayout.addWidget(self.bin_magic_btn)
        self.line2 = QtWidgets.QFrame(self.verticalLayoutWidget)
        self.line2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line2.setObjectName("line2")
        self.verticalLayout.addWidget(self.line2)
        self.bin_thresh_btn = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.bin_thresh_btn.setObjectName("bin_thresh_btn")
        self.verticalLayout.addWidget(self.bin_thresh_btn)
        self.bin_thresh_spin = QtWidgets.QSpinBox(self.verticalLayoutWidget)
        self.bin_thresh_spin.setAlignment(QtCore.Qt.AlignCenter)
        self.bin_thresh_spin.setSpecialValueText("")
        self.bin_thresh_spin.setMaximum(255)
        self.bin_thresh_spin.setProperty("value", 175)
        self.bin_thresh_spin.setObjectName("bin_thresh_spin")
        self.verticalLayout.addWidget(self.bin_thresh_spin)
        self.line3 = QtWidgets.QFrame(self.verticalLayoutWidget)
        self.line3.setFrameShadow(QtWidgets.QFrame.Plain)
        self.line3.setLineWidth(3)
        self.line3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line3.setObjectName("line3")
        self.verticalLayout.addWidget(self.line3)
        self.erosion_btn = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.erosion_btn.setObjectName("erosion_btn")
        self.verticalLayout.addWidget(self.erosion_btn)
        self.dilation_btn = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.dilation_btn.setObjectName("dilation_btn")
        self.verticalLayout.addWidget(self.dilation_btn)
        self.opening_btn = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.opening_btn.setObjectName("opening_btn")
        self.verticalLayout.addWidget(self.opening_btn)
        self.closing_btn = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.closing_btn.setObjectName("closing_btn")
        self.verticalLayout.addWidget(self.closing_btn)
        self.morph_ex_kernel_size_spin = QtWidgets.QSpinBox(self.verticalLayoutWidget)
        self.morph_ex_kernel_size_spin.setAlignment(QtCore.Qt.AlignCenter)
        self.morph_ex_kernel_size_spin.setMinimum(1)
        self.morph_ex_kernel_size_spin.setMaximum(50)
        self.morph_ex_kernel_size_spin.setProperty("value", 3)
        self.morph_ex_kernel_size_spin.setObjectName("morph_ex_kernel_size_spin")
        self.verticalLayout.addWidget(self.morph_ex_kernel_size_spin)
        self.line4 = QtWidgets.QFrame(self.verticalLayoutWidget)
        self.line4.setFrameShadow(QtWidgets.QFrame.Plain)
        self.line4.setLineWidth(3)
        self.line4.setFrameShape(QtWidgets.QFrame.HLine)
        self.line4.setObjectName("line4")
        self.verticalLayout.addWidget(self.line4)
        self.get_res_btn = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.get_res_btn.setObjectName("get_res_btn")
        self.verticalLayout.addWidget(self.get_res_btn)
        self.min_square_spin = QtWidgets.QSpinBox(self.verticalLayoutWidget)
        self.min_square_spin.setAlignment(QtCore.Qt.AlignCenter)
        self.min_square_spin.setMinimum(1)
        self.min_square_spin.setMaximum(300)
        self.min_square_spin.setProperty("value", 20)
        self.min_square_spin.setObjectName("min_square_spin")
        self.verticalLayout.addWidget(self.min_square_spin)
        self.clear_selections_btn = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.clear_selections_btn.setObjectName("clear_selections_btn")
        self.verticalLayout.addWidget(self.clear_selections_btn)
        self.line5 = QtWidgets.QFrame(self.verticalLayoutWidget)
        self.line5.setFrameShadow(QtWidgets.QFrame.Plain)
        self.line5.setLineWidth(3)
        self.line5.setFrameShape(QtWidgets.QFrame.HLine)
        self.line5.setObjectName("line5")
        self.verticalLayout.addWidget(self.line5)
        self.res_line_edit = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.res_line_edit.setFont(font)
        self.res_line_edit.setAlignment(QtCore.Qt.AlignCenter)
        self.res_line_edit.setObjectName("res_line_edit")
        self.verticalLayout.addWidget(self.res_line_edit)
        self.v_line_user_input = QtWidgets.QFrame(self.centralwidget)
        self.v_line_user_input.setGeometry(QtCore.QRect(256, 2, 20, 727))
        self.v_line_user_input.setFrameShadow(QtWidgets.QFrame.Plain)
        self.v_line_user_input.setLineWidth(3)
        self.v_line_user_input.setFrameShape(QtWidgets.QFrame.VLine)
        self.v_line_user_input.setObjectName("v_line_user_input")
        self.h_line_user_input = QtWidgets.QFrame(self.centralwidget)
        self.h_line_user_input.setGeometry(QtCore.QRect(0, 649, 268, 161))
        self.h_line_user_input.setFrameShadow(QtWidgets.QFrame.Plain)
        self.h_line_user_input.setLineWidth(3)
        self.h_line_user_input.setMidLineWidth(0)
        self.h_line_user_input.setFrameShape(QtWidgets.QFrame.HLine)
        self.h_line_user_input.setObjectName("h_line_user_input")

        # using ImageLabel
        self.image_label = ImageLabel(self.centralwidget)
        self.image_label.setGeometry(QtCore.QRect(272, 2, 1271, 831))
        self.image_label.setText("")
        self.image_label.setScaledContents(True)
        self.image_label.setObjectName("image_label")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.load_btn.clicked.connect(self.load_img)
        
        self.get_gray_btn.clicked.connect(self.get_gray_img)

        self.bin_magic_btn.clicked.connect(self.bin_magic)
        self.bin_thresh_btn.clicked.connect(self.bin_thresh)

        self.erosion_btn.clicked.connect(lambda: self.morph_ex("erosion"))
        self.dilation_btn.clicked.connect(lambda: self.morph_ex("dilation"))
        self.opening_btn.clicked.connect(lambda: self.morph_ex("opening"))
        self.closing_btn.clicked.connect(lambda: self.morph_ex("closing"))

        self.clear_selections_btn.clicked.connect(self.image_label.clear_selections)

        self.get_res_btn.clicked.connect(self.get_res)

        MainWindow.setWindowIcon(QIcon('etc/icon.png'))
        self.apply_stylesheet()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Кладбища самолётов"))
        self.load_btn.setText(_translate("MainWindow", "Загрузить изображение"))
        self.get_gray_btn.setText(_translate("MainWindow", "Получить серое изображение"))
        self.bin_magic_btn.setText(_translate("MainWindow", "Бинаризовать, учитывая цвет"))
        self.bin_thresh_btn.setText(_translate("MainWindow", "Бинаризовать с порогом\nи инвертировать"))
        self.bin_thresh_spin.setPrefix(_translate("MainWindow", "Порог: "))
        self.erosion_btn.setText(_translate("MainWindow", "Применить эрозию"))
        self.dilation_btn.setText(_translate("MainWindow", "Применить дилатацию"))
        self.opening_btn.setText(_translate("MainWindow", "Применить открытие"))
        self.closing_btn.setText(_translate("MainWindow", "Применить закрытие"))
        self.morph_ex_kernel_size_spin.setPrefix(_translate("MainWindow", "Размер кв. ядра из единиц: "))
        self.get_res_btn.setText(_translate("MainWindow", "Посчитать число самолетов"))
        self.min_square_spin.setPrefix(_translate("MainWindow", "Минимальная площадь самолета: "))
        self.clear_selections_btn.setText(_translate("MainWindow", "Очистить выделенные рамки"))
        self.res_line_edit.setText(_translate("MainWindow", "Число самолетов: 0"))

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
        if is_black_and_white:
            self.image_label.set_original_pixmap(pixmap)
        self.image_label.setPixmap(pixmap)

    def load_img(self):
        try:
            file_dialog = QFileDialog()
            file_path, _ = file_dialog.getOpenFileName(self.window,
                                                    "Выберите изображение", "", "Images (*.png *.jpg *.bmp)")
            if file_path:
                self.original_img = plt.imread(file_path)
                self.draw_img(self.original_img)

            self.gray_img = None
            self.black_and_white_img = None
        except:
            self._make_error_window()

    def get_gray_img(self):
        try:
            if self.original_img is None:
                self._make_error_window("Сначала нужно загрузить изображение.")
                return
            
            self.gray_img = cv.cvtColor(self.original_img, cv.COLOR_RGB2GRAY)
            self.draw_img(self.gray_img)
            self.black_and_white_img = None
        except:
             self._make_error_window()

    def bin_thresh(self):
        try:
            if self.gray_img is None:
                self._make_error_window("Сначала нужно получить серое изображение.")
                return
            
            thresh = self.bin_thresh_spin.value()
            self.black_and_white_img = img_pr.binarize_thresh(
                self.gray_img, thresh, inverse=True)
            
            self.draw_img(self.black_and_white_img)
        except:
            self._make_error_window()

    def bin_magic(self):
        try:
            if self.original_img is None:
                self._make_error_window("Сначала нужно загрузить изображение.")
                return

            self.black_and_white_img = img_pr.binarize_magic(self.original_img)      
            self.draw_img(self.black_and_white_img)
        except:
            self._make_error_window()

    def morph_ex(self, ex):
        try:
            if self.black_and_white_img is None:
                self._make_error_window("Сначала нужно бинаризовать изображение.")
                return

            kernel_size = self.morph_ex_kernel_size_spin.value()
            self.black_and_white_img = img_pr.make_morph_ex(self.black_and_white_img,
                                        ex, (kernel_size, kernel_size))
            self.draw_img(self.black_and_white_img)
        except:
            self._make_error_window()

    def get_res(self):
        try:
            if self.black_and_white_img is None:
                self._make_error_window("Сначала нужно бинаризовать изображение.")
                return
            
            if not self.image_label.selections: 
                img_with_contours, n_planes = img_pr.get_img_with_contours_and_cnt(
                    self.black_and_white_img, self.min_square_spin.value()
                )
            else:
                n_selections = self.image_label.save_selections()
                n_planes = 0
                #img_with_contours = np.full_like(self.original_img, img_pr.MAX_BRIGHTNESS)
                img_with_contours = cv.cvtColor(self.black_and_white_img,  cv.COLOR_GRAY2RGB)

                for i in range(n_selections):
                    img_with_contours_fragment, n_planes_fragment = img_pr.get_img_with_contours_and_cnt(
                        None, self.min_square_spin.value(), img_name=f"selected_fragment_{i}"
                    )

                    # Применяем координаты к фрагменту изображения с контурами
                    selected_rect = self.image_label.get_selected_rect_relative_to_original_image(
                        self.image_label.selections[i]
                    )

                    # Применяем фрагмент к изображению с контурами
                    try:
                        img_with_contours[
                            selected_rect.topLeft().y():selected_rect.bottomRight().y(),
                            selected_rect.topLeft().x():selected_rect.bottomRight().x()] = img_with_contours_fragment[:-1, :-1, :]

                        n_planes += n_planes_fragment
                    except ValueError:
                        warnings.warn(f"""img_with_contours[...] = img_with_contours_fragment warning: {
                            img_with_contours[
                            selected_rect.topLeft().y():selected_rect.bottomRight().y(),
                            selected_rect.topLeft().x():selected_rect.bottomRight().x()].shape,
                            img_with_contours_fragment[:-1, :-1, :].shape
                        }""")
                
            self.draw_img(img_with_contours)
            self.res_line_edit.setText(f"Число самолетов: {n_planes}")
        
        except:
            self._make_error_window()

    def apply_stylesheet(self):
        try:
            style_file = "styles.css"  # Путь к вашему CSS-файлу
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
