import os
import re
import pandas as pd
from pathlib import Path

def add_pseudo_labels(input_file, output_file, pseudo_dir="OPTIC/Fundus/label"):
    """
    为现有文件添加伪标签列
    
    参数:
        input_file: 输入的CSV/TXT文件路径
        output_file: 输出的CSV文件路径
        pseudo_dir: 伪标签文件所在目录
    """
    with open(input_file, 'r') as f:
        lines = f.readlines()
    new_lines = []
    
    if lines[0].strip().startswith('image'):
        header = lines[0].strip()
        new_header = header + ',pseudo_label\n'
        new_lines.append(new_header)
        start_idx = 1
    
    # 处理数据行
    for line in lines[start_idx:]:
        line = line.strip()
        if not line:
            continue
            
        parts = line.split(',')
        if len(parts) >= 2:
            image_path = parts[0]
            mask_path = parts[1]
            image_filename = os.path.basename(image_path)
                
            pseudo_filename = f"test_{image_filename}_mask.png"
            pseudo_path = os.path.join(pseudo_dir, pseudo_filename)
                
            # 检查文件是否存在
            if os.path.exists(os.path.join("OPTIC", pseudo_path)):
                new_line = f"{image_path},{mask_path},{pseudo_path}\n"
            else:
                print(f"警告: 伪标签文件不存在: {pseudo_path}")
                new_line = f"{image_path},{mask_path}, \n"
        else:
            new_line = line + ', \n'
        
        new_lines.append(new_line)
    
    with open(output_file, 'w') as f:
        f.writelines(new_lines)
    
    print(f"处理完成! 结果已保存到: {output_file}")
    return new_lines


if __name__ == "__main__":
    Target_Dataset = ['RIM_ONE_r3', 'REFUGE', 'ORIGA', 'REFUGE_Valid', 'Drishti_GS']

    target_test_csv = []
    for target in Target_Dataset:
        if target != 'REFUGE_Valid':
            target_test_csv.append(target + '_train.csv')
            target_test_csv.append(target + '_test.csv')
        else:
            target_test_csv.append(target + '.csv')
    for csv_file in target_test_csv:
        print(f"Processing {csv_file}...")
        output_csv = csv_file.replace('.csv', '_pseudo.csv')
        add_pseudo_labels(
            input_file=os.path.join("OPTIC/Fundus", csv_file),
            output_file=os.path.join("OPTIC/Fundus", output_csv),
            pseudo_dir="Fundus/label"
        )