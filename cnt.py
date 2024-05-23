import os
import glob
import codecs

def count_lines(directory):
    total_lines = 0
    file_count = 0

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                with codecs.open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    total_lines += len(lines)
                    file_count += 1

    return total_lines, file_count

# 统计当前文件夹及子文件夹下的 Python 文件的行数
directory = '.'  # 修改为你要统计的文件夹路径
lines, files = count_lines(directory)
print(f"总共有 {files} 个文件，共计 {lines} 行")
