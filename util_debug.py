import numpy as np
import json


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

