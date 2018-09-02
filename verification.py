import face_recognition
import cv2
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
import os
from pylab import *

mpl.rcParams['font.sans-serif'] = ['SimHei']

# 全局变量
test_img = '2.jpg'
threshold = 0.5  # 阈值

# 所有路径
project_path = os.path.abspath('.')  # 工程绝对路径
member_path = os.path.join(project_path, '所有成员')  # 成员路径
image_path = os.path.join(project_path, '现场照片')  # 现场照片路径


# 所有成员图片展示
def show_all_members():
    # 获取'所有成员'文件夹下所有文件列表
    dir_list = os.listdir(member_path)
    i = 140
    # 遍历 member_path 文件夹
    for dir in dir_list:
        # 获取单个文件的名称（取'.'前部分内容，如文件名称为 dir = '詹姆斯.jpg'，则 name = '詹姆斯'）
        name = dir.split('.')[0]
        # 获取文件绝对路径
        dir_path = os.path.join(member_path, dir)
        # 打开原图并转换通道
        member_img = PIL.Image.open(dir_path)
        member_img = cv2.cvtColor(np.asarray(member_img), cv2.COLOR_RGB2BGR)
        # 统一图片尺寸
        member_img = cv2.resize(member_img, (600, 600))
        # 图片展示
        i = i + 1
        plt.subplot(i).set_title(name)
        plt.imshow(cv2.cvtColor(member_img, cv2.COLOR_BGR2RGB))
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        plt.axis('off')
    plt.suptitle('人脸库')
    plt.show()


# 人脸库生成函数
def generate_face_database(member_path):
    """
    :param member_path: 人脸文件夹路径
    :return: 成员姓名列表（member_names）、成员人脸解码信息列表（member_face_encodings）
    """
    print('......正在解码人脸库......')
    # 获取'所有成员'文件夹下所有文件列表
    dir_list = os.listdir(member_path)
    # 初始化 成员姓名列表（member_names）、成员人脸解码信息列表（member_face_encodings）
    member_names = []
    member_face_encodings = []
    # 遍历 member_path 文件夹
    for dir in dir_list:
        # 获取单个文件的名称（取'.'前部分内容，如文件名称为 dir = '詹姆斯.jpg'，则 name = '詹姆斯'）
        name = dir.split('.')[0]
        # 获取文件绝对路径
        dir_path = os.path.join(member_path, dir)
        # 加载文件（图片）
        img = face_recognition.load_image_file(dir_path)
        # 获取人脸位置坐标
        face_location = face_recognition.face_locations(img)
        # 获取人脸解码信息（维度为128的行向量）
        face_encoding = face_recognition.face_encodings(img, face_location, num_jitters=5)[0]
        # 逐个加入 成员姓名列表（member_names）
        member_names.append(name)
        # 逐个加入 成员人脸解码信息列表（member_face_encodings）
        member_face_encodings.append(face_encoding)
        print(name, '人脸解码成功')
    print('......人脸库解码完毕......')
    return member_names, member_face_encodings


if __name__ == '__main__':

    # 所有成员图片展示
    show_all_members()
    # 获取测试图片绝对路径
    test_img_path = os.path.join(image_path, test_img)
    # 获取测试图片的解码信息：
    # 成员姓名列表（member_list）、成员人脸解码信息列表（member_face_encodings）
    member_list, member_face_encodings = generate_face_database(member_path)
    # print(member_names, np.array(member_face_encodings).shape)

    # 打开原图并转换通道
    img_to_show = PIL.Image.open(test_img_path)
    img_to_show = cv2.cvtColor(np.asarray(img_to_show), cv2.COLOR_RGB2BGR)
    # 复制原图给origin_img
    origin_img = np.copy(img_to_show)
    # 显示origin_img
    plt.subplot(111).set_title("原始图片")
    plt.imshow(cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.show()

    print('.......开始检测图片.......')
    # 初始化 缺席名单列表（absent_list）、出席名单列表（present_list）、蹭课名单列表（others_list）
    #       蹭课人脸位置（others_face_locations）、蹭课标识符（others_flag）
    absent_list = []
    present_list = []
    others_list = []
    others_face_locations = []
    others_flag = False
    # 所有成员个数
    member_nums = len(member_list)
    # 加载测试图片
    img = face_recognition.load_image_file(test_img_path)
    # 获取测试图片所有人脸位置坐标
    face_locations = face_recognition.face_locations(img)
    # 获取测试图片所有人脸解码信息
    encodings = face_recognition.face_encodings(img, face_locations, num_jitters=2)
    # 出席人数
    present_nums = len(encodings)
    # 取出检测图片中每个人脸的特征向量与所有成员人脸特征向量对比
    for j in range(0, present_nums):
        # 单个从测试图片中检测出来的特征向量
        face_to_compare = encodings[j]
        # 与所有成员的特征向量分别计算欧氏距离
        distances = face_recognition.face_distance(member_face_encodings, face_to_compare)
        print('distances', distances)
        # 取最小欧氏距离
        min_distance = min(distances)
        # 查找最小欧氏距离的序号
        dist_nums = len(distances)
        min_dist_num = 0
        for dis_num in range(0, dist_nums):
            if distances[dis_num] == min_distance:
                min_dist_num = dis_num
                break

        # 设置阈值，当最小阈值大于设定，则将此人加入蹭课名单列表，小于阈值则加入出席名单列表
        # 备注：人脸越相似，欧式距离越小
        if min_distance > threshold:
            others_list.append(j)
            others_face_locations.append(face_locations[j])
        else:
            present_list.append(member_list[min_dist_num])
    # print('present_list', present_list)

    # 求member_list与present_list的差集，得到缺席名单列表（absent_list）
    for member in member_list:
        if member not in present_list:
            absent_list.append(member)
    # 缺席人数
    absent_nums = len(absent_list)
    # 蹭课人数
    others_nums = len(others_list)
    # 结果展示
    print('......图片检测完毕......')
    print('应到人数：', member_nums)
    print('实到人数：', present_nums)
    # 若无缺席
    if absent_nums == 0:
        print('检测结果：全勤')
        # 若有人蹭课
        if not others_nums == 0:
            print('蹭课人数：', others_nums)
            others_flag = True
    # 若有缺席
    else:
        print('检测结果：缺勤')
        # 若无人蹭课
        if others_nums == 0:
            print('缺席人数：', absent_nums)
            print('缺席人员：', absent_list)
        # 若有人蹭课
        else:
            print('蹭课人数：', others_nums)
            print('缺席人数：', absent_nums)
            print('缺席人员：', absent_list)
            others_flag = True

    # 检测图片中人脸个数
    face_num = len(face_locations)
    # 遍历检测图片中每个人脸，并在img_to_show上画框
    for i in range(0, face_num):
        # 人脸框的上下左右
        top = face_locations[i][0]
        right = face_locations[i][1]
        bottom = face_locations[i][2]
        left = face_locations[i][3]

        start = (left, top)
        end = (right, bottom)
        # 框的颜色、厚度
        color = (255, 0, 0)
        thickness = 4
        # 画框
        cv2.rectangle(img_to_show, start, end, color, thickness)

    for k in range(0, others_nums):
        # 人脸框的上下左右
        top = others_face_locations[k][0]
        right = others_face_locations[k][1]
        bottom = others_face_locations[k][2]
        left = others_face_locations[k][3]

        start = (left, top)
        end = (right, bottom)
        # 框的颜色、厚度
        color = (0, 0, 255)
        thickness = 4
        # 画框
        cv2.rectangle(img_to_show, start, end, color, thickness)
    detected_img = img_to_show

    # 显示人脸检测结果
    plt.subplot(111).set_title("检测图片")
    plt.imshow(cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.show()
