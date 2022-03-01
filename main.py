import cv2
import numpy as np
import random
from sklearn.cluster import KMeans
from util import time_test, the_world
from tqdm import tqdm
import warnings
import json
import os
import copy

warnings.filterwarnings("ignore")

NEAR_DISTANCE = 100

if_filter_color = 0
filter_color = 19
if_filter_img = 0
filter_img = '61d410b38c549f0001c3dee4'


def random_color_mapping():
    res = []
    for i in range(256 * 10):
        res.append([
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        ])
    return np.array(res)


def color_bar(colors):
    res = np.zeros([100, len(colors) * 100, 3])
    for i, o in enumerate(colors):
        # print(i, o)
        res[:, i * 100:(i + 1) * 100] = colors[i]
    return res


def get_origin_color_mapping(origin_region_img, origin_color_img):
    color_mapping = {}
    min_color, max_color = get_color_range(origin_region_img)
    # print()
    for color_type in tqdm(range(min_color, max_color + 1)):
        if if_filter_color and color_type != filter_color:
            continue
        b_type = color_type % 256
        g_type = int(color_type / 256)

        color_img = np.copy(origin_color_img)
        region_img = np.copy(origin_region_img)
        mask = cv2.inRange(region_img, np.array([b_type, g_type, 0]), np.array([b_type, g_type, 0]))

        region_img[mask == 0] = [255, 255, 255]

        kernel = np.ones((3, 3), np.uint8)
        region_img = cv2.dilate(region_img, kernel, iterations=2)

        if if_filter_color:
            cv2.imwrite('output/region.png', region_img)

        colors = color_img[np.all(region_img - [b_type, g_type, 0] == 0, axis=2)]
        colors = colors[::int(colors.shape[0] / 10000 + 1)]

        if len(colors) < 2:
            if len(colors):
                color_mapping[color_type] = colors[0]
            else:
                color_mapping[color_type] = [0, 0, 0]
        else:
            kmeans = KMeans(n_clusters=2, random_state=0).fit(colors)
            classed_colors = []
            for i in np.array(kmeans.cluster_centers_, dtype=int):
                # if np.sum(i) < 30:
                #     continue
                classed_colors.append(i)
            classed_colors = np.array(classed_colors)
            if if_filter_color:
                cv2.imwrite('output/color.png', color_bar(classed_colors))
            max_range = 0
            max_color = [0, 0, 0]
            colors_img = colors.reshape(colors.shape[0], 1, 3)
            # print(np.average(colors_img, axis=0))
            for color in classed_colors:
                # number = cv2.inRange(colors_img, color - 10, color + 10)
                # score = (np.sum(number / 200))
                score = 255 - np.average(np.abs((colors_img - color)))

                if if_filter_color:
                    print(color, score)
                if score > max_range:
                    max_range = score
                    max_color = color
            color_mapping[color_type] = max_color
    return color_mapping


def compute_colors_num(color_mapping):
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
    color_mapping = copy.deepcopy(color_mapping)
    block_num = len(color_mapping)
    colors = []
    for i in color_mapping:
        colors.append(color_mapping[i])
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

    c_num = {}
    for i in color_mapping:
        color = ','.join([str(a) for a in color_mapping[i]])
        if color in c_num:
            c_num[color] += 1
        else:
            c_num[color] = 1
    single_color_mapping = {}
    for color_str in (c_num):
        if c_num[color_str] == 1:
            for index in color_mapping:
                if ','.join([str(a) for a in color_mapping[index]]) == color_str:
                    single_color_mapping[index] = color_mapping[index]

    if len(single_color_mapping) > 5:
        short_distance = 30
        n = 0
        while 1:
            n += 1
            cn, scp = compute_colors_num(single_color_mapping)
            if scp <= 5:
                break
            for i in single_color_mapping:
                min_distance = 1000000
                cloest_color = None
                for j in single_color_mapping:
                    if i == j:
                        continue
                    color_distance = np.sum(np.abs(np.array(single_color_mapping[i]) - np.array(
                        single_color_mapping[j])))
                    if color_distance < short_distance:
                        if min_distance > color_distance:
                            min_distance = color_distance
                            cloest_color = single_color_mapping[j]
                if cloest_color is not None:
                    single_color_mapping[i] = cloest_color

            short_distance += 5

        for i in single_color_mapping:
            color_mapping[i] = single_color_mapping[i]
    colors_num, single_color_num = compute_colors_num(color_mapping)
    fname = '{}_{}_{}'.format(block_num, colors_num, single_color_num)
    return color_mapping, fname


def fill(color_mapping, fn, region_img, color_img):
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
    # mask = cv2.inRange(color_img, np.array([0, 0, 0]), np.array([20, 20, 20]))
    # res[mask > 0] = [0, 0, 0]
    cv2.imwrite(fn, res)
    return res


def get_block_num(img):
    img[np.all(img - [255, 255, 255] == 0, axis=2)] = [0, 0, 0]
    img = np.array(img, dtype=int)
    block_num = (np.max(img[:, :, 0] + img[:, :, 1] * 256))
    return block_num


def get_color_range(img):
    img[np.all(img - [255, 255, 255] == 0, axis=2)] = [0, 0, 0]
    img = np.array(img, dtype=int)
    min_color = (np.min(img[:, :, 0] + img[:, :, 1] * 256))
    max_color = (np.max(img[:, :, 0] + img[:, :, 1] * 256))
    return min_color, max_color


def export_mapping(color_mapping):
    out = {}
    for i in color_mapping:
        [a, b, c] = color_mapping[i]
        out[i] = [int(a), int(b), int(c)]
    with open('color_mapping.json', 'w') as fo:
        fo.write(json.dumps(out, indent=2, ensure_ascii=False))
        fo.close()


def import_mapping():
    color_mapping = {}
    with open('color_mapping.json', 'r') as fi:
        content = json.loads(fi.read())
        for i in content:
            color_mapping[int(i)] = content[i]
        fi.close()
    return color_mapping


@time_test(True)
def main():
    for name, l, fl in os.walk('input'):
        for id in l:
            if if_filter_img and id != filter_img:
                continue
            if 'todo' in id:
                continue
            print('dealing {}'.format(id))
            color_file_name = 'input\\{}\\{}_colored.jpeg'.format(id, id)
            regin_file_name = 'input\\{}\\{}_region.png'.format(id, id)
            result_file_name = 'input\\{}\\{}_result.png'.format(id, id)
            region_img = cv2.imread(regin_file_name)
            color_img = cv2.imread(color_file_name)
            result_img = cv2.imread(result_file_name)


            # start_x = 1400
            # start_y = 0
            # region_img = region_img[start_x:start_x + 500, start_y:start_y + 500]
            # color_img = color_img[start_x:start_x + 500, start_y:start_y + 500]
            # start_x = random.randint(0, 2048 - 500)
            # start_y = random.randint(0, 2048 - 500)
            # print(start_x, start_y)
            # start_x = 839
            # start_y = 850

            # cv2.imwrite('output\\{}_region.png'.format(id), region_img)
            # cv2.imwrite('output\\{}_0result.png'.format(id), result_img)
            cv2.imwrite('output\\{}_colored.png'.format(id), color_img)
            #

            color_mapping = get_origin_color_mapping(region_img, color_img)
            export_mapping(color_mapping)
            #
            # color_mapping = import_mapping()

            fname = 'TEST'
            # if not if_filter_color:
            color_mapping, fname = slim(color_mapping)
            #
            # cv2.imwrite('output\\{}_color.png'.format(id), color_bar(classified_colors))
            fill(color_mapping, 'output\\{}_{}.png'.format(id, fname), region_img, color_img)


if __name__ == '__main__':
    main()
