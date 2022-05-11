import matplotlib.pyplot as plt
import re

def cool_savefig(title):
    file_title = title.replace(' ', '_')
    file_title = re.sub('[^A-Za-z0-9_]+', '', file_title)
    # turn multiple underscores, e.g. resulting from a ' - ', into one
    file_title = re.sub('_[_]+', '_', file_title)
    file_title = '{}.png'.format(file_title)
    plt.savefig(file_title, bbox_inches = "tight")