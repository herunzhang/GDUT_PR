import cv2
import os
from image_show import Model, get_file_name, get_name_list

# 定义输入图片文件夹路径
input_image_folder = r"C:\Users\Sun_Philip\Desktop\recognition_gender-master\face\rawdata_bmp"
# 定义输出男性图片文件夹路径
output_male_folder = r"C:\Users\Sun_Philip\Desktop\recognition_gender-master\face\sex\male"
# 定义输出女性图片文件夹路径
output_female_folder = r"C:\Users\Sun_Philip\Desktop\recognition_gender-master\face\sex\female"

# 如果输出男性和女性的文件夹不存在，则创建它们
if not os.path.exists(output_male_folder):
    os.makedirs(output_male_folder)
if not os.path.exists(output_female_folder):
    os.makedirs(output_female_folder)

# 初始化人脸检测分类器
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
# 初始化模型
model = Model()
model.load()
name_list = get_name_list()

# 获取输入文件夹中所有图片文件的名称列表
image_file_names = get_file_name(input_image_folder)

for image_file_name in image_file_names:
    file_path = os.path.join(input_image_folder, image_file_name)
    # 读取图片
    frame = cv2.imread(file_path)
    if frame is None:
        print(f"无法读取文件 {file_path} 对应的图片，跳过该文件")
        continue
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 检测人脸
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        ROI = gray[x:x + w, y:y + h]
        ROI = cv2.resize(ROI, (128, 128), interpolation=cv2.INTER_LINEAR)
        label, prob = model.predict(ROI)
        if prob > 0.5:
            show_name = name_list[label]
            if show_name == 'female':
                output_folder = output_female_folder
            else:
                output_folder = output_male_folder
            # 构建输出文件路径，保持原文件名不变，将图片保存到对应的性别文件夹中
            output_path = os.path.join(output_folder, image_file_name)
            cv2.imwrite(output_path, frame)
            print(f"已将 {image_file_name} 分类并保存到 {output_folder}")
        else:
            print(f"图片 {image_file_name} 人脸性别判别结果不确定，跳过保存")

print("图片分类保存操作完成")