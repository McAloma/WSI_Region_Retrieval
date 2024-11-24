import os, sys
sys.path.append("/hpc2hdd/home/rsu704/MDI_RAG_project")


def save_results_as_comment(py_file_path, results):
    """
    将结果以注释的形式写入指定的 Python 文件。
    
    :param py_file_path: Python 文件的路径
    :param results: 需要写入的结果（可以是任意 Python 数据结构）
    """
    # 转换结果为字符串形式
    results_str = f"# 聚类结果: {results}\n"
    
    # 检查文件是否存在
    if not os.path.exists(py_file_path):
        raise FileNotFoundError(f"{py_file_path} 不存在！")

    # 读取现有文件内容
    with open(py_file_path, 'r', encoding='utf-8') as f:
        file_content = f.readlines()

    # 插入结果为注释（假设插在文件头部）
    # 你可以根据需求修改插入位置
    file_content.insert(0, results_str)
    
    # 将内容写回文件
    with open(py_file_path, 'w', encoding='utf-8') as f:
        f.writelines(file_content)