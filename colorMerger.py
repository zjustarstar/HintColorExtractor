import cv2
import numpy as np
from sklearn.cluster import KMeans
import os


# 两种颜色是否相近的判断标准
# 返回是否是相近颜色,以及距离值
def isCloseColor(c1, c2, thre):
    d = abs(c1 - c2)
    # 单独在每个通道计算,每个通道的像素值都不能超过阈值;
    return max(d) < thre, sum(d)


def mergeColors_all(colors_map, dist_thre, remove_dup=False):
    '''
    所有距离在dist_thre之间的颜色,都会被合并.合并时,选择亮度大一点的作为最终结果
    :param colors_map: 输入的颜色表
    :param dist_thre: 颜色相似度阈值
    :param remove_dup: 是否删除颜色表中相同的颜色
    :return: 合并后的颜色表
    '''

    for i in range(len(colors_map)-1):
        for j in range(i+1, len(colors_map)):
            # 选择亮度更大一点的作为最终结果
            is_close, _ = isCloseColor(colors_map[i], colors_map[j], dist_thre)
            if is_close:
                if sum(colors_map[i]) > sum(colors_map[j]):
                    colors_map[j] = colors_map[i]
                else:
                    colors_map[i] = colors_map[j]

    if remove_dup:
        out = list(set([tuple(i) for i in colors_map]))
        # 从[()]转为[[]]
        new_colors = []
        for i in range(len(out)):
            new_colors.append(list(out[i]))
        return new_colors
    else:
        return colors_map


def combine_near_color(color_mapping, distance):
    """
    合并 color_mapping 中rgb距离小于distance的颜色
    :param color_mapping:
    :param distance:
    :return:
    """
    for i in color_mapping:
        min_distance = 1000000
        cloest_color = None
        for j in color_mapping:
            if i == j:
                continue
            is_close_color, color_distance = isCloseColor(color_mapping[i], color_mapping[j], distance)
            if is_close_color:
                if min_distance > color_distance:
                    min_distance = color_distance
                    cloest_color = color_mapping[j]
        if cloest_color is not None:
            color_mapping[i] = cloest_color
    return color_mapping




def compute_colors_num(color_mapping):
    """
    此函数用于计算所有的颜色数量 colors_num, 和只存在一次的颜色数量 single_color_num
    纯暴力计算
    :param color_mapping:
    :return:
        colors_num: int
        single_color_num: int
    """
    colors = []
    for i in color_mapping:
        [a, b, c] = color_mapping[i]
        colors.append([str(int(a)), str(int(b)), str(int(c))])

    str_colors = ([','.join(a) for a in colors])
    colors_num = len(list(set(str_colors)))

    single_color_num = 0
    for i in str_colors:
        nc = 0
        for j in str_colors:
            if i == j:
                nc += 1
        if nc == 1:
            single_color_num += 1

    return colors_num, single_color_num


def final_merge(color_mapping, max_color_num=99):
    """
    此函数用于精简 color_mapping, 以使其达到 99 > 颜色数量 > 区域数*6% ,且 单一颜色数量 < 5 的要求
    :param color_mapping:
    :param max_color_num: 最多有几种颜色
    :return: color_mapping:
    """
    # 求区块数量
    block_num = len(color_mapping)
    # 提取 color_mapping 中的 value
    colors = []
    for i in color_mapping:
        colors.append(color_mapping[i])

    # 当颜色数大于最大值时, 将颜色列表输入 Kmeans 聚类, 每个颜色从聚类中选取最相近的颜色更新
    if block_num > max_color_num:
        clusters_num = 99
        kmeans = KMeans(n_clusters=clusters_num, random_state=0).fit(colors)
        classified_colors = np.array(kmeans.cluster_centers_, dtype=int)
        for i in color_mapping:
            min_distance = 1000000
            cloest_color = None
            for j in classified_colors:
                color1 = np.array(color_mapping[i])
                color2 = np.array(j)
                distance = np.sum(np.abs(color1 - color2))
                if distance < min_distance:
                    min_distance = distance
                    cloest_color = j
            color_mapping[i] = cloest_color

    # 计算每种颜色被使用的次数
    c_num = {}
    for i in color_mapping:
        color = ','.join([str(a) for a in color_mapping[i]])
        if color in c_num:
            c_num[color] += 1
        else:
            c_num[color] = 1

    # 提取使用次数为一的颜色
    single_color_mapping = {}
    for color_str in c_num:
        if c_num[color_str] == 1:
            for index in color_mapping:
                if ','.join([str(a) for a in color_mapping[index]]) == color_str:
                    single_color_mapping[index] = color_mapping[index]

    # 如果单一颜色数量大于5, 则作进一步处理
    # 1 设定一个 初始distance , 将颜色集合中rgb通道距离小于此distance的颜色合并
    # 2 计算合并后的单一颜色数量, 若仍大于5, 则增大distance, 重复第1步
    if len(single_color_mapping) > 5:
        short_distance = 30
        n = 0
        while 1:
            n += 1
            cn, scp = compute_colors_num(single_color_mapping)
            if scp <= 5:
                break
            single_color_mapping = combine_near_color(single_color_mapping, short_distance)
            # single_color_mapping = mergeColors(single_color_mapping, short_distance)
            short_distance += 5

        for i in single_color_mapping:
            color_mapping[i] = single_color_mapping[i]
    return color_mapping
