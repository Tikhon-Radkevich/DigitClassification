from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, \
    QSizePolicy, QGraphicsScene, QGraphicsView
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QCursor
from PyQt5.QtCore import Qt

import sys

from predict import get_digit


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()
        self.pixel_matrix = [[0 for i in range(28)] for j in range(28)]

        self.is_recognized: bool = False
        self.default_level_text = None
        self.scene = None
        self.view = None
        self.run_button = None
        self.clear_button = None
        self.output_label = None

    def initUI(self):
        # Set the window title
        self.setWindowTitle("Draw and Convert")

        # Set the window size
        self.setGeometry(100, 100, 400, 400)

        # Create the drawing space
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setFixedSize(280, 280)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setStyleSheet("background-color: white;")

        # Create the "Run" button
        self.run_button = QPushButton("Run", self)
        self.run_button.clicked.connect(self.run)

        # Create the "Clear" button
        self.clear_button = QPushButton("Clear", self)
        self.clear_button.clicked.connect(self.clear)

        # Create the label for the output text
        self.is_recognized = False
        self.default_level_text = "Click run to recognize a number"
        self.output_label = QLabel(self.deafult_lavel_text, self)
        self.output_label.setWordWrap(True)
        self.output_label.setAlignment(Qt.AlignTop)

        # Add the UI components to the window
        widget = QWidget()
        layout = QVBoxLayout()
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.run_button)
        button_layout.addWidget(self.clear_button)
        layout.addLayout(button_layout)
        layout.addWidget(self.view)
        layout.addWidget(self.output_label)
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def keyPressEvent(self, event):
        pixel_size = 10
        pos = QCursor()
        x = pos.pos().x()
        x = int(x * 28 / 1920)
        y = pos.pos().y()
        y = int(y * 28 / 1080)
        bias = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        if 1 <= x < 27 and 1 <= y < 27:
            if self.is_recognized:
                self.is_recognized = False
                self.output_label.setText(self.deafult_lavel_text)
            # Draw the pixel on the drawing space and update the pixel matrix
            pen = QPen(QColor("black"), pixel_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            brush = QBrush(QColor("black"), Qt.SolidPattern)
            self.scene.addEllipse(x * pixel_size, y * pixel_size, pixel_size, pixel_size, pen, brush)
            self.pixel_matrix[y][x] = 255
            for x_bias, y_bias in bias:
                self.scene.addEllipse((x + x_bias) * pixel_size, (y + y_bias) * pixel_size, pixel_size, pixel_size, pen,
                                      brush)
                self.pixel_matrix[y + y_bias][x + x_bias] = 255

    def run(self):
        # Call the my_func() function with the pixel matrix as the argument
        digit = get_digit(self.pixel_matrix)
        output_text = f"Digit:  {digit}"
        # Update the output label with the output text
        self.output_label.setText(output_text)
        self.is_recognized = True

    def clear(self):
        # Clear the drawing space and the pixel matrix
        self.scene.clear()
        self.pixel_matrix = [[0 for i in range(28)] for j in range(28)]


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
