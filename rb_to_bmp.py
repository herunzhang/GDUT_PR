import os
import numpy as np
from PIL import Image

# 定义输入文件夹路径，存放二进制文件的文件夹，你可以根据实际情况修改
input_folder = r"C:\Users\Sun_Philip\Desktop\recognition_gender-master\face\rawdata"
# 定义输出文件夹路径，用于存放转换后的BMP文件，你可以根据实际情况修改
output_folder = r"C:\Users\Sun_Philip\Desktop\recognition_gender-master\face\rawdata_bmp"

# 如果输出文件夹不存在，则创建它
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历输入文件夹中的所有文件
for file_name in os.listdir(input_folder):
    file_path = os.path.join(input_folder, file_name)
    # 只处理文件（跳过文件夹等其他非文件类型的条目）
    if os.path.isfile(file_path):
        with open(file_path, 'rb') as file:
            data = file.read()
            # 尝试获取图像的实际尺寸（这里简单通过计算总元素个数开方来大致估算，假设图像是正方形，仅为一种示例方法，实际可能需更精准判断）
            num_elements = len(data)
            side_length = int(np.sqrt(num_elements))
            if side_length ** 2 == num_elements:
                # 将读取到的二进制数据转换为numpy数组，并重塑为合适的图像尺寸
                image_data = np.frombuffer(data, dtype=np.uint8).reshape(side_length, side_length)
            else:
                print(f"文件 {file_name} 中的数据无法确定合适的图像尺寸，跳过该文件")
                continue
            # 将numpy数组转换为PIL的Image对象，指定模式为'L'（灰度图像）
            image = Image.fromarray(image_data, mode='L')
            # 构建输出文件的路径，保持文件名不变，仅修改后缀为.bmp
            output_path = os.path.join(output_folder, os.path.splitext(file_name)[0] + ".bmp")
            # 保存图像为BMP格式文件
            image.save(output_path, 'BMP')
            print(f"已将 {file_name} 转换并保存为 {output_path}")

print("所有文件转换完成！")