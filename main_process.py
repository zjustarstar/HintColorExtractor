import cv2
import numpy as np
import colorMerger
import paramset
from sklearn.cluster import KMeans
from tqdm import tqdm
import util_debug
import warnings
import os


# warnings.filterwarnings("ignore")

# 以下参数仅供调试用
if_filter_color = 0  # 若仅调试单独颜色，则赋值为1
filter_color = 19  # 需调试的单独颜色


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
            kmeans = KMeans(n_clusters=paramset.COLOR_MAP_CLUSTER_NUM, random_state=0).fit(colors)

            # 处理聚类结果, 获取已分类颜色列表 classified_colors
            classified_colors = []
            for i in np.array(kmeans.cluster_centers_, dtype=int):
                classified_colors.append(i)

            # 如果颜色都可以合并, 则直接合并;
            merged_colors = colorMerger.mergeColors_all(classified_colors, paramset.COLOR_DIST_THRE, True)
            # 只有一种颜色,直接赋值
            if len(merged_colors) == 1:
                color_mapping[color_type] = merged_colors[0]
                continue

            classified_colors = np.array(merged_colors)

            # classified_colors 输出测试
            if if_filter_color:
                cv2.imwrite('output/color.png', util_debug.color_bar(classified_colors))

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


def gen_hint_color_map(color_file_name, region_file_name, params):
    '''
    输入要处理的彩色图及区域图，以及存放结果的输出文件夹
    :param params: 算法参数
    :param color_file_name:
    :param region_file_name: 输入的彩色图和区域图
    :return:
    '''
    print("currently process {0}".format(color_file_name))

    output_folder = params[paramset.OUTPUT_PATH]
    filepath, tempfilename = os.path.split(color_file_name)
    filename, extension = os.path.splitext(tempfilename)
    fileid = filename[0:filename.find('_')]

    region_img = cv2.imread(region_file_name)
    color_img = cv2.imread(color_file_name)
    if region_img is None:
        print("fail to read region image")
        return False
    if color_img is None:
        print("fail to read color image")
        return False

    color_mapping = get_origin_color_mapping(region_img, color_img)

    # 下面两行供调试使用, 需多次运行时无需重复运行上一行函数, 只需先导出color_mapping, 后续直接导入即可
    # 导出 color_mapping
    # export_mapping(color_mapping)
    # 导入 color_mapping
    # color_mapping = import_mapping()

    # 区块数量
    block_num = len(color_mapping)
    # 精简颜色映射集合, 合并相近颜色
    color_mapping = colorMerger.final_merge(color_mapping, params)
    # 获取颜色数和单个颜色数
    colors_num, single_color_num = colorMerger.compute_colors_num(color_mapping)
    # 生成文件名
    add_name = '{}_{}_{}'.format(block_num, colors_num, single_color_num)
    # 填充彩图
    res_img = fill(color_mapping, region_img, color_img)

    # 输出彩图
    # 未指定输出路径，则输出在原文件夹
    if len(output_folder) == 0:
        cv2.imwrite('{}\\{}_{}.png'.format(filepath, fileid, add_name), res_img)
    else:
        cv2.imwrite('{}\\{}_{}.png'.format(output_folder, fileid, add_name), res_img)

    return True



def gen_hint_color_map_async(color_file_names, region_file_names, params, fileid):
    '''
    输入要处理的彩色图及区域图，以及存放结果的输出文件夹。多线程版本
    :param params: 算法参数
    :param color_file_name:
    :param region_file_name: 输入的彩色图列表和区域图列表
    :param fileid: 当前处理的文件index
    :return:
    '''

    color_file_name = color_file_names[fileid]
    region_file_name = region_file_names[fileid]

    return gen_hint_color_map(color_file_name, region_file_name, params)
