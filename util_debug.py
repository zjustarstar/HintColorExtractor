import numpy as np
import json
import time
import math


def color_bar(colors):
    """
    此函数仅作调试用，用于输出图片以展示颜色列表
    :param colors: 颜色列表 ndarray  e.g. [[0,0,0],[255,255,255]]
    :return: 色条图片 ndarray
    """
    res = np.zeros([100, len(colors) * 100, 3])
    for i, o in enumerate(colors):
        res[:, i * 100:(i + 1) * 100] = colors[i]
    return res


def export_mapping(color_mapping):
    """
    导出 color_mapping, 供调试使用
    :param color_mapping:
    :return:
    """
    out = {}
    for i in color_mapping:
        [a, b, c] = color_mapping[i]
        out[i] = [int(a), int(b), int(c)]
    with open('color_mapping.json', 'w') as fo:
        fo.write(json.dumps(out, indent=2, ensure_ascii=False))
        fo.close()


def import_mapping():
    """
    导入 color_mapping, 供调试使用
    :return:
    """
    color_mapping = {}
    with open('color_mapping.json', 'r') as fi:
        content = json.loads(fi.read())
        for i in content:
            color_mapping[int(i)] = content[i]
        fi.close()
    return color_mapping


def show_run_time(display=True):
    def run(func):
        def decorated(*args, **kwargs):
            tstart = time.time()
            res = func(*args, **kwargs)
            tend = time.time()

            timespan = tend - tstart
            hour = math.floor(timespan / 3600)
            m = math.floor((timespan - hour * 3600) / 60)
            sec = math.floor(timespan - hour * 3600 - m * 60)

            if display:
                print("耗时:%d时%d分%d秒" % (hour, m, sec))
            return res

        return decorated
    return run


