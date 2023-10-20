import os

def delete_images_with_prefix(folder_path, prefix):
    # 获取文件夹中的所有文件
    file_list = os.listdir(folder_path)

    # 遍历文件夹中的每个文件
    for filename in file_list:
        if filename.startswith(prefix):
            file_path = os.path.join(folder_path, filename)
            
            # 删除以指定前缀开头的文件
            os.remove(file_path)
            print(f"Deleted: {file_path}")

# 指定文件夹路径和前缀名，删除以指定前缀名开头的图像文件
folder_path = "dataset/Training/no_tumor"
prefix = "enhanced"
delete_images_with_prefix(folder_path, prefix)
