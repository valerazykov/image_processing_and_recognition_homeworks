import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import cv2 as cv
import matplotlib.pyplot as plt
import warnings

import image_processing as img_pr


class Ui_MainWindow(object):
    def __init__(self):
        self.original_img = None
        self.gray_img = None
        self.black_white_img = None
        self.black_white_img_after_morph_ex = None
        self.img_with_skeleton = None

    def setupUi(self, MainWindow):
        self.window = MainWindow

        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1305, 775)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(0, 0, 265, 681))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.verticalLayout.setContentsMargins(2, 0, 0, 0)
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
        self.bin_thresh_btn = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.bin_thresh_btn.setObjectName("bin_thresh_btn")
        self.verticalLayout.addWidget(self.bin_thresh_btn)
        self.bin_thresh_spin = QtWidgets.QSpinBox(self.verticalLayoutWidget)
        self.bin_thresh_spin.setAlignment(QtCore.Qt.AlignCenter)
        self.bin_thresh_spin.setSpecialValueText("")
        self.bin_thresh_spin.setMaximum(255)
        self.bin_thresh_spin.setProperty("value", 75)
        self.bin_thresh_spin.setObjectName("bin_thresh_spin")
        self.verticalLayout.addWidget(self.bin_thresh_spin)
        self.line3 = QtWidgets.QFrame(self.verticalLayoutWidget)
        self.line3.setFrameShadow(QtWidgets.QFrame.Plain)
        self.line3.setLineWidth(3)
        self.line3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line3.setObjectName("line3")
        self.verticalLayout.addWidget(self.line3)
        self.closing_btn = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.closing_btn.setObjectName("closing_btn")
        self.verticalLayout.addWidget(self.closing_btn)
        self.morph_ex_kernel_size_spin = QtWidgets.QSpinBox(self.verticalLayoutWidget)
        self.morph_ex_kernel_size_spin.setAlignment(QtCore.Qt.AlignCenter)
        self.morph_ex_kernel_size_spin.setMinimum(1)
        self.morph_ex_kernel_size_spin.setMaximum(50)
        self.morph_ex_kernel_size_spin.setProperty("value", 15)
        self.morph_ex_kernel_size_spin.setObjectName("morph_ex_kernel_size_spin")
        self.verticalLayout.addWidget(self.morph_ex_kernel_size_spin)
        self.line4 = QtWidgets.QFrame(self.verticalLayoutWidget)
        self.line4.setFrameShadow(QtWidgets.QFrame.Plain)
        self.line4.setLineWidth(3)
        self.line4.setFrameShape(QtWidgets.QFrame.HLine)
        self.line4.setObjectName("line4")
        self.verticalLayout.addWidget(self.line4)
        self.get_skel_and_clf_btn = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.get_skel_and_clf_btn.setObjectName("get_skel_and_clf_btn")
        self.verticalLayout.addWidget(self.get_skel_and_clf_btn)
        self.skel_pruning_spin = QtWidgets.QSpinBox(self.verticalLayoutWidget)
        self.skel_pruning_spin.setAlignment(QtCore.Qt.AlignCenter)
        self.skel_pruning_spin.setMinimum(0)
        self.skel_pruning_spin.setMaximum(2000)
        self.skel_pruning_spin.setProperty("value", 600)
        self.skel_pruning_spin.setObjectName("skel_pruning_spin")
        self.verticalLayout.addWidget(self.skel_pruning_spin)
        self.skel_min_dist_spin = QtWidgets.QSpinBox(self.verticalLayoutWidget)
        self.skel_min_dist_spin.setAlignment(QtCore.Qt.AlignCenter)
        self.skel_min_dist_spin.setMaximum(200)
        self.skel_min_dist_spin.setProperty("value", 25)
        self.skel_min_dist_spin.setObjectName("skel_min_dist_spin")
        self.verticalLayout.addWidget(self.skel_min_dist_spin)
        self.line5 = QtWidgets.QFrame(self.verticalLayoutWidget)
        self.line5.setFrameShadow(QtWidgets.QFrame.Plain)
        self.line5.setLineWidth(3)
        self.line5.setFrameShape(QtWidgets.QFrame.HLine)
        self.line5.setObjectName("line5")
        self.verticalLayout.addWidget(self.line5)
        self.degree_vec_line_edit = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.degree_vec_line_edit.setAlignment(QtCore.Qt.AlignCenter)
        self.degree_vec_line_edit.setObjectName("degree_vec_line_edit")
        self.verticalLayout.addWidget(self.degree_vec_line_edit)
        self.res_line_edit = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.res_line_edit.setFont(font)
        self.res_line_edit.setAlignment(QtCore.Qt.AlignCenter)
        self.res_line_edit.setObjectName("res_line_edit")
        self.verticalLayout.addWidget(self.res_line_edit)
        self.h_line_user_input = QtWidgets.QFrame(self.verticalLayoutWidget)
        self.h_line_user_input.setFrameShadow(QtWidgets.QFrame.Plain)
        self.h_line_user_input.setLineWidth(3)
        self.h_line_user_input.setMidLineWidth(0)
        self.h_line_user_input.setFrameShape(QtWidgets.QFrame.HLine)
        self.h_line_user_input.setObjectName("h_line_user_input")
        self.verticalLayout.addWidget(self.h_line_user_input)
        self.v_line_user_input = QtWidgets.QFrame(self.centralwidget)
        self.v_line_user_input.setGeometry(QtCore.QRect(256, 2, 20, 670))
        self.v_line_user_input.setFrameShadow(QtWidgets.QFrame.Plain)
        self.v_line_user_input.setLineWidth(3)
        self.v_line_user_input.setFrameShape(QtWidgets.QFrame.VLine)
        self.v_line_user_input.setObjectName("v_line_user_input")
        self.image_label = QtWidgets.QLabel(self.centralwidget)
        self.image_label.setGeometry(QtCore.QRect(272, 2, 1024, 768))
        self.image_label.setText("")
        #self.image_label.setPixmap(QtGui.QPixmap("pictures/3.jpg"))
        self.image_label.setScaledContents(True)
        self.image_label.setObjectName("image_label")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        MainWindow.setWindowIcon(QIcon('etc/icon.png'))
        self.apply_stylesheet()

        self.load_btn.clicked.connect(self.load_img)
        self.get_gray_btn.clicked.connect(self.get_gray_img)
        self.bin_thresh_btn.clicked.connect(self.bin_thresh)
        self.closing_btn.clicked.connect(self.make_closing)
        self.get_skel_and_clf_btn.clicked.connect(self.get_skel_and_clf)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Скелеты"))
        self.load_btn.setText(_translate("MainWindow", "Загрузить изображение"))
        self.get_gray_btn.setText(_translate("MainWindow", "Получить серое изображение"))
        self.bin_thresh_btn.setText(_translate("MainWindow", "Бинаризовать с порогом"))
        self.bin_thresh_spin.setPrefix(_translate("MainWindow", "Порог: "))
        self.closing_btn.setText(_translate("MainWindow", "Применить закрытие"))
        self.morph_ex_kernel_size_spin.setPrefix(_translate("MainWindow", "Размер кв. ядра из единиц: "))
        self.get_skel_and_clf_btn.setText(_translate("MainWindow", "Получить скелет и классиф-ть"))
        self.skel_pruning_spin.setPrefix(_translate("MainWindow", "Порог стрижки скелета: "))
        self.skel_min_dist_spin.setPrefix(_translate("MainWindow", "Мин. расст. между вершинами: "))
        self.degree_vec_line_edit.setText(_translate("MainWindow", "Вектор степеней вершин: (0, 0, 0, 0, 0)"))
        self.res_line_edit.setText(_translate("MainWindow", "Результат: класс ?"))

    def apply_stylesheet(self):
        try:
            style_file = "styles.css"
            with open(style_file, "r", encoding="utf-8") as f:
                style = f.read()
                self.window.setStyleSheet(style)
        except:
            warnings.warn("styles error")

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

            self.gray_img = None
            self.black_white_img = None
            self.black_white_img_after_morph_ex = None
            self.img_with_skeleton = None

            self.degree_vec_line_edit.setText("Вектор степеней вершин: (0, 0, 0, 0, 0)")
            self.res_line_edit.setText("Результат: класс ?")

        except:
            self._make_error_window()

    def get_gray_img(self):
        try:
            if self.original_img is None:
                self._make_error_window("Сначала нужно загрузить изображение.")
                return
            self.gray_img = cv.cvtColor(self.original_img, cv.COLOR_RGB2GRAY)
            self.draw_img(self.gray_img)

            self.black_white_img = None
            self.black_white_img_after_morph_ex = None
            self.img_with_skeleton = None

        except:
            self._make_error_window()

    def bin_thresh(self):
        try:
            if self.gray_img is None:
                self._make_error_window("Сначала нужно получить серое изображение.")
                return

            thresh = self.bin_thresh_spin.value()
            self.black_white_img = cv.threshold(self.gray_img, thresh, 255, cv.THRESH_BINARY)[1]
            self.draw_img(self.black_white_img)

            self.black_white_img_after_morph_ex = None
            self.img_with_skeleton = None

        except:
            self._make_error_window()

    def make_closing(self):
        try:
            if self.black_white_img is None:
                self._make_error_window("Сначала нужно бинаризовать изображение.")
                return

            kernel_shape = self.morph_ex_kernel_size_spin.value()
            kernel_shape = (kernel_shape, kernel_shape)
            self.black_white_img_after_morph_ex = img_pr.make_morph_ex(self.black_white_img, "closing", kernel_shape)
            self.draw_img(self.black_white_img_after_morph_ex)

            self.img_with_skeleton = None

        except:
            self._make_error_window()

    def get_skel_and_clf(self):
        try:
            if self.black_white_img_after_morph_ex is None:
                self._make_error_window("Сначала нужно применить закрытие.")
                return

            pruning_thresh = self.skel_pruning_spin.value()
            min_dist = self.skel_min_dist_spin.value()

            skel, graph = img_pr.get_skel_and_graph(self.black_white_img_after_morph_ex, pruning_thresh)
            self.img_with_skeleton, nodes = img_pr.get_img_with_skel(self.black_white_img_after_morph_ex, skel, graph, min_dist)
            self.draw_img(self.img_with_skeleton)
            degree_vector, cls = img_pr.get_degree_vector_and_classify(nodes)

            self.degree_vec_line_edit.setText(f"Вектор степеней вершин: {tuple(degree_vector)}")
            self.res_line_edit.setText(f"Результат: класс {cls}")
        except:
            self._make_error_window()



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
