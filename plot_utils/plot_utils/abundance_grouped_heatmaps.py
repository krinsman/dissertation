import matplotlib.ticker as mtick
import numpy as np

def add_ticks(axes_axis):
    axes_axis.set_ticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    axes_axis.set_minor_locator(mtick.FixedLocator([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]))
    axes_axis.set_minor_formatter(mtick.FixedFormatter(['.01%', '.02%', '.05%', '0.1%', '0.2%', '0.5%', '1.0%', '2.0%', '5.0%']))
    
def custom_percent_formatter(x, pos):
    percentage = np.around(100 * x, decimals=6)
    return '{}%'.format(percentage)