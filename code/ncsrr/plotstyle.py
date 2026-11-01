import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager

COL = ['#2b6cb0', '#c53030', '#2f855a', '#4a5568', '#b7791f', '#6b46c1']


def setup():
    for path in ['/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc']:
        try:
            font_manager.fontManager.addfont(path)
        except Exception:
            pass
    plt.rcParams.update({
        'font.sans-serif': ['WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'SimHei', 'DejaVu Sans'],
        'axes.unicode_minus': False,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.titlesize': 10, 'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'legend.fontsize': 8,
    })
