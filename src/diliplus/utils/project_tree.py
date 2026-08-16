"""
DILI-PLUS | 项目目录打印工具（包实现）

职责：递归打印当前项目树，并忽略缓存、日志和常见临时目录。
输入：集中配置中的项目根目录。
输出：终端中的目录树；不修改任何项目文件。
状态：开发辅助工具，不属于数据或实验流水线。
"""

import os
from pathlib import Path

from diliplus.config import load_settings

# 定义要忽略的文件夹和文件后缀
IGNORE_DIRS = {'.git', '.idea', '__pycache__', 'wandb', 'logs', '.ipynb_checkpoints'}
IGNORE_EXTS = {'.tmp', '.log'}

def print_tree(dir_path, prefix=''):
    path = Path(dir_path)
    # 获取所有条目并排序（文件夹在前，文件在后）
    try:
        entries = sorted(list(path.iterdir()), key=lambda e: (not e.is_dir(), e.name.lower()))
    except PermissionError:
        return

    entries = [e for e in entries if e.name not in IGNORE_DIRS and e.suffix not in IGNORE_EXTS]
    
    count = len(entries)
    for i, entry in enumerate(entries):
        connector = '└── ' if i == count - 1 else '├── '
        
        if entry.is_dir():
            print(f"{prefix}{connector}📂 {entry.name}/")
            new_prefix = prefix + ('    ' if i == count - 1 else '│   ')
            print_tree(entry, new_prefix)
        else:
            # 标注文件大小
            size_mb = entry.stat().st_size / (1024 * 1024)
            size_str = f"({size_mb:.1f} MB)" if size_mb > 1 else ""
            print(f"{prefix}{connector}📜 {entry.name} {size_str}")

def main(settings=None):
    settings = settings or load_settings()
    root = settings.paths.root
    print(f"📦 Project Root: {root}")
    print("=" * 50)
    print_tree(root)
    print("=" * 50)

if __name__ == '__main__':
    main()
