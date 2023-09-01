import pyqtgraph as pg
import numpy as np

# Generate data
x = np.linspace(1, 10, 100)
y = np.linspace(1, 10, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))/np.sqrt(X**2 + Y**2)

# Create plot widget
pw = pg.plot(title='Contour Plot')

# Create contour plot item
contour = pg.ImageItem()
pw.addItem(contour)

# Set data for contour plot
contour.setImage(Z)

# Set axis labels
pw.setLabel('bottom', 'X Axis', units='m')
pw.setLabel('left', 'Y Axis', units='m')

# Show plot
pg.QtGui.QGuiApplication.exec_()
