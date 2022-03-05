import main_process as mp
import paramset
from multiprocessing import Pool
import os
import time
from util_debug import show_run_time

# 以下参数仅用于调试
if_filter_img = 0  # 若仅调试单张图片，则赋值为1
filter_img = 'a6270530533e908e56f08713184cca07'  # 需调试的单张图片id


# 单cpu处理
def main_single(input_folder, params):
    # 遍历输入文件夹
    for name, l, fl in os.walk(input_folder):
        for id in l:
            if if_filter_img and id != filter_img:
                continue

            color_file_name = input_folder + '\\{}\\{}_colored.jpeg'.format(id, id)
            region_file_name = input_folder + '\\{}\\{}_region.png'.format(id, id)

            res = mp.gen_hint_color_map(color_file_name, region_file_name, params)
            if not res:
                continue


def main_async(input_folder, params):
    colorFileNames, regionFileNames = [], []
    # 遍历输入文件夹
    for name, l, fl in os.walk(input_folder):
        for id in l:
            if if_filter_img and id != filter_img:
                continue

            color_file_name = input_folder + '\\{}\\{}_colored.jpeg'.format(id, id)
            region_file_name = input_folder + '\\{}\\{}_region.png'.format(id, id)

            colorFileNames.append(color_file_name)
            regionFileNames.append(region_file_name)

    print("you are using async model. cpu count=%d" % os.cpu_count())
    print("Total %d images in the queue" % len(colorFileNames))
    res_list = []  # 结果集
    p = Pool(processes=os.cpu_count() - 1)
    for i in range(len(colorFileNames)):
        res = p.apply_async(mp.gen_hint_color_map_async,
                            (colorFileNames, regionFileNames, params, i))
        res_list.append(res)
        # 必须有这个sleep
        time.sleep(1)
    p.close()
    p.join()


# param是参数集
@show_run_time()
def main(input_folder, params):
    output_folder = params[paramset.OUTPUT_PATH]

    # 如果结果文件夹不为空, 则先创建
    if len(output_folder) > 0:
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

    if params[paramset.ASYNC_MODEL]:
        main_async(input_folder, params)
    else:
        main_single(input_folder, params)


if __name__ == '__main__':
    param = {}
    # 最多颜色数量
    param[paramset.MAX_COLORS_NUM] = 90
    # 仅一个区域的颜色数量
    param[paramset.ONE_REGION_COLORS_NUM] = 5
    # 保存在output文件夹
    param[paramset.OUTPUT_PATH] = 'output'
    # cpu模式，单线程还是多线程
    param[paramset.ASYNC_MODEL] = True

    input_folder = 'example'
    main(input_folder, param)
