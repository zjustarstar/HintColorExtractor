import main_process as mp
import paramset
import os

if_filter_img = 0  # 若仅调试单张图片，则赋值为1
filter_img = '61e698d985bc0c00010f1260'  # 需调试的单张图片id

# 如果output_folder不存在,则直接创建
# param是参数集
def main(input_folder, params, output_folder=''):
    if len(output_folder) > 0:
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

    # 遍历输入文件夹
    for name, l, fl in os.walk(input_folder):
        for id in l:
            if if_filter_img and id != filter_img:
                continue
            print('dealing {}'.format(id))
            color_file_name = input_folder + '\\{}\\{}_colored.jpeg'.format(id, id)
            region_file_name = input_folder + '\\{}\\{}_region.png'.format(id, id)
            mp.gen_hint_color_map(color_file_name, region_file_name, params, output_folder)


if __name__ == '__main__':
    input_folder = 'example2'
    param = {}
    # 最多颜色数量
    param[paramset.MAX_COLORS_NUM] = 90
    # 仅一个区域的颜色数量
    param[paramset.ONE_REGION_COLORS_NUM] = 4

    main(input_folder, param, 'output')
