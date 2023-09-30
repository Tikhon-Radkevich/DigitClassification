from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QPainter, QColor, QPen, QPainterPath
from PyQt5.QtCore import Qt

import sys

from predict import DigitClassifier


class DrawingWidget(QWidget):
    def __init__(self, main_win):
        super().__init__()

        self.main_win = main_win
        self.setFixedSize(280, 280)
        self.path = QPainterPath()
        self.pixel_matrix = [[0 for _ in range(28)] for _ in range(28)]

    def paintEvent(self, event):
        painter = QPainter(self)

        # Set the border (circuit)
        border = QPen(QColor("black"), 2, Qt.SolidLine)
        painter.setPen(border)
        painter.drawRect(self.rect())

        # Set the drawing pen
        pen = QPen(QColor("black"), 20, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen)

        if self.path:
            painter.drawPath(self.path)

    def mouseReleaseEvent(self, event):
        # Fill the pixel_matrix based on the drawn path
        for point in self.path.toSubpathPolygons():
            for p in point:
                x = int(p.x() * 28 / 280)
                y = int(p.y() * 28 / 280)
                if 0 <= x <= 27 and 0 <= y <= 27:
                    self.pixel_matrix[y][x] = 255

    def mousePressEvent(self, event):
        self.main_win.clear_predict()
        if event.button() == Qt.LeftButton:
            self.path.moveTo(event.pos())
            self.update()

    def mouseMoveEvent(self, event):
        if self.path:
            self.path.lineTo(event.pos())
            self.update()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.digit_classifier = DigitClassifier("../model/digit_classification_model.h5")
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Digit Classifier")
        self.setGeometry(100, 100, 400, 400)

        # Create the drawing widget
        self.drawing_widget = DrawingWidget(self)

        # Create the "Run" button           
        self.run_button = QPushButton("Run", self)
        self.run_button.clicked.connect(self.run)

        # Create the "Clear" button
        self.clear_button = QPushButton("Clear", self)
        self.clear_button.clicked.connect(self.clear)

        # Create the label for the output text
        self.default_level_text = "Click run to recognize a number"
        self.output_label = QLabel(self.default_level_text, self)
        self.output_label.setWordWrap(True)
        self.output_label.setAlignment(Qt.AlignTop)

        # Add the UI components to the window
        widget = QWidget()
        layout = QVBoxLayout()
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.run_button)
        button_layout.addWidget(self.clear_button)
        layout.addLayout(button_layout)
        layout.addWidget(self.drawing_widget)
        layout.addWidget(self.output_label)
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def run(self):
        # Process the drawing to get the digit
        digit = self.digit_classifier.get_digit(self.drawing_widget.pixel_matrix)
        output_text = f"Digit: {digit}"
        # Update the output label with the output text
        self.output_label.setText(output_text)

    def clear_predict(self):
        self.output_label.setText(self.default_level_text)

    def clear(self):
        # Clear the drawing widget and the pixel matrix
        self.drawing_widget.path = QPainterPath()
        self.drawing_widget.update()
        self.drawing_widget.pixel_matrix = [[0 for _ in range(28)] for _ in range(28)]
        self.clear_predict()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
