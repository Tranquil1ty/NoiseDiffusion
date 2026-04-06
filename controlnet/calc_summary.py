import csv
import sys
from collections import defaultdict

def calculate_category_averages(csv_path, target_categories):
    # 用来存储每个方法对应的指标列表
    # { method: { 'topiq': [val1, val2, ...], 'pdv': [val1, val2, ...] } }
    data = defaultdict(lambda: {'topiq': [], 'pdv': []})
    
    try:
        with open(csv_path, newline='', encoding='utf-8-sig') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                method = row['方法']
                category = row['类别']
                
                # 如果类别在目标列表中，或者是目标列表的一部分（处理逗号分隔等情况）
                if category in target_categories:
                    try:
                        topiq = float(row['mean_TOPIQ'])
                        pdv = float(row['PDV'])
                        data[method]['topiq'].append(topiq)
                        data[method]['pdv'].append(pdv)
                    except ValueError:
                        # 忽略无法转换为浮点数的值（如均值行可能存在的空值）
                        continue
    except FileNotFoundError:
        print(f"错误: 找不到文件 {csv_path}")
        return

    if not data:
        print(f"未找到指定类别 {target_categories} 的数据。")
        return

    print(f"{'方法':<25} | {'类别数量':<10} | {'平均 TOPIQ':<15} | {'平均 PDV':<10}")
    print("-" * 70)
    
    for method, metrics in data.items():
        if method == "【均值】": continue # 过滤掉原有的汇总行
        
        count = len(metrics['topiq'])
        if count == 0: continue
        
        avg_topiq = sum(metrics['topiq']) / count
        avg_pdv = sum(metrics['pdv']) / count
        
        print(f"{method:<25} | {count:<10} | {avg_topiq:<15.4f} | {avg_pdv:<10.4f}")

if __name__ == "__main__":
    csv_file = "/data2/home/pengfei/Graduate_experiment/NoiseDiffusion/controlnet/sample_results/evaluation_results/summary.csv"
    
    if len(sys.argv) > 1:
        # 从命令行参数获取类别，例如: python script.py apple banana
        categories = sys.argv[1:]
    else:
        # 默认示例
        categories = ["apple", "banana"]
    
    print(f"计算类别 {categories} 的均值数据:\n")
    calculate_category_averages(csv_file, categories)
