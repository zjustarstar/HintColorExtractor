import cv2
import numpy as np
import random
from sklearn.cluster import KMeans
from tqdm import tqdm
import warnings
import json
import os
import copy

warnings.filterwarnings("ignore")

CLUSTERS_NUM = 2  # Kmeans 函数的聚类数量

# 以下四个参数仅供调试用
if_filter_color = 0  # 若仅调试单独颜色，则赋值为1
filter_color = 19  # 需调试的单独颜色
if_filter_img = 1  # 若仅调试单张图片，则赋值为1
filter_img = '61e698d985bc0c00010f1260'  # 需调试的单张图片id


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


def get_origin_color_mapping(origin_region_img, origin_color_img):
    """
    此函数用于获取初始的 color_mapping ，即 region_img 中的色块序号与应填充颜色的映射

    :param origin_region_img:   原始分块图像
    :param origin_color_img:    原始彩图
    :return: 序号与颜色的映射对象
    e.g {
        1：[0,0,0],
        2: [255,255,255],
        ...
    }
    """
    color_mapping = {}
    # 最大最小区域编号;编号按照b和g通道的数字进行累加
    min_color, max_color = get_color_range(origin_region_img)

    for color_type in tqdm(range(min_color, max_color + 1)):
        # 用于测试
        if if_filter_color and color_type != filter_color:
            continue
        b_type = color_type % 256
        g_type = int(color_type / 256)

        # mask表示当前分割区域
        color_img = np.copy(origin_color_img)
        region_img = np.copy(origin_region_img)
        mask = cv2.inRange(region_img, np.array([b_type, g_type, 0]), np.array([b_type, g_type, 0]))

        # 将需填色部分之外的区域置白, 以便后续进行膨胀操作
        region_img[mask == 0] = [255, 255, 255]

        # 使用膨胀函数将白色区域扩张, 相对地能使填色区域减小, 从而剥离了区域边缘的线稿颜色
        kernel = np.ones((3, 3), np.uint8)
        region_img = cv2.dilate(region_img, kernel, iterations=2)

        # region_img 输出测试
        if if_filter_color:
            cv2.imwrite('output/region.png', region_img)

        # 从彩图中获取需填色的区域的像素色彩列表
        colors = color_img[np.all(region_img - [b_type, g_type, 0] == 0, axis=2)]
        # 若colors数组太大, 则精简像素列表, 提高效率
        colors = colors[::int(colors.shape[0] / 10000 + 1)]

        # 若colors数组太小, 则无需输入Kmeans, 直接处理
        if len(colors) < 2:
            if len(colors):
                color_mapping[color_type] = colors[0]
            else:
                color_mapping[color_type] = [0, 0, 0]
        else:
            # 用Kmeans 处理colors ,将颜色聚类, 获取 CLUSTERS_NUM 种候选颜色
            kmeans = KMeans(n_clusters=CLUSTERS_NUM, random_state=0).fit(colors)

            # 处理聚类结果, 获取已分类颜色列表 classified_colors
            classified_colors = []
            for i in np.array(kmeans.cluster_centers_, dtype=int):
                classified_colors.append(i)
            classified_colors = np.array(classified_colors)

            # classified_colors 输出测试
            if if_filter_color:
                cv2.imwrite('output/color.png', color_bar(classified_colors))

            # 从 classified_colors 中选取占比最大的颜色
            max_range = 0
            max_color = [0, 0, 0]
            colors_img = colors.reshape(colors.shape[0], 1, 3)
            for color in classified_colors:
                # 得分公式简化 score = 255 - | image - color |
                # 颜色在原图中占比越高, 得分越高
                score = 255 - np.average(np.abs((colors_img - color)))
                if if_filter_color:
                    print(color, score)
                if score > max_range:
                    max_range = score
                    max_color = color
            color_mapping[color_type] = max_color
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


def slim(color_mapping):
    """
    此函数用于精简 color_mapping, 以使其达到 99 > 颜色数量 > 区域数*6% ,且 单一颜色数量 < 5 的要求
    :param color_mapping:
    :return: color_mapping:
    """
    # 求区块数量
    block_num = len(color_mapping)
    # 提取 color_mapping 中的 value
    colors = []
    for i in color_mapping:
        colors.append(color_mapping[i])

    # 当颜色数大于99时, 将颜色列表输入 Kmeans 聚类, 每个颜色从聚类中选取最相近的颜色更新
    if block_num > 99:
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
            short_distance += 5

        for i in single_color_mapping:
            color_mapping[i] = single_color_mapping[i]
    return color_mapping


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
            color_distance = np.sum(np.abs(np.array(color_mapping[i]) - np.array(
                color_mapping[j])))
            if color_distance < distance:
                if min_distance > color_distance:
                    min_distance = color_distance
                    cloest_color = color_mapping[j]
        if cloest_color is not None:
            color_mapping[i] = cloest_color
    return color_mapping


def fill(color_mapping, region_img, color_img):
    """
    使用 color_mapping 中的颜色填充, 得到最终图片
    :param color_mapping:
    :param region_img:
    :param color_img:
    :return:
    """
    res = np.copy(color_img)
    min_color, max_color = get_color_range(region_img)
    for color_type in tqdm(range(min_color, max_color + 1)):
        if if_filter_color and color_type != filter_color:
            continue
        b_type = color_type % 256
        g_type = int(color_type / 256)
        mask = cv2.inRange(region_img, np.array([b_type, g_type, 0]), np.array([b_type, g_type, 0]))
        if np.sum(mask / 255) == 0:
            continue
        res[mask > 0] = color_mapping[color_type]
    return res


def get_color_range(img):
    """
    获取 region_img 中的色彩范围
    :param img:
    :return:
    """
    img[np.all(img - [255, 255, 255] == 0, axis=2)] = [0, 0, 0]
    img = np.array(img, dtype=int)
    min_color = (np.min(img[:, :, 0] + img[:, :, 1] * 256))
    max_color = (np.max(img[:, :, 0] + img[:, :, 1] * 256))
    return min_color, max_color


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


def main():
    input_folder = 'example'
    # 遍历输入文件夹
    for name, l, fl in os.walk(input_folder):
        for id in l:
            if if_filter_img and id != filter_img:
                continue
            print('dealing {}'.format(id))
            color_file_name = input_folder + '\\{}\\{}_colored.jpeg'.format(id, id)
            region_file_name = input_folder + '\\{}\\{}_region.png'.format(id, id)
            region_img = cv2.imread(region_file_name)
            color_img = cv2.imread(color_file_name)

            color_mapping = get_origin_color_mapping(region_img, color_img)

            # 下面两行供调试使用, 需多次运行时无需重复运行上一行函数, 只需先导出color_mapping, 后续直接导入即可
            # 导出 color_mapping
            # export_mapping(color_mapping)
            # 导入 color_mapping
            # color_mapping = import_mapping()

            # 区块数量
            block_num = len(color_mapping)
            # 精简颜色映射集合, 合并相近颜色
            color_mapping = slim(color_mapping)
            # 获取颜色数和单个颜色数
            colors_num, single_color_num = compute_colors_num(color_mapping)
            # 生成文件名
            file_name = '{}_{}_{}'.format(block_num, colors_num, single_color_num)
            # 填充彩图
            res_img = fill(color_mapping, region_img, color_img)
            # 输出彩图
            cv2.imwrite('output\\{}_{}.png'.format(id, file_name), res_img)


if __name__ == '__main__':
    main()
