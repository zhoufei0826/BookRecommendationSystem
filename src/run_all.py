"""
成员 A 一键运行脚本：按顺序执行所有任务
1. data_analysis.py   - 数据检查与统计
2. make_split.py      - 固定训练/验证切分
3. evaluate.py        - 统一评估接口测试
4. popularity.py      - 热门度 Top-10 基线
5. bpr_model.py       - BPR 模型实现与评估
6. check_submission.py - 提交文件质量检查（需先生成 submission_final.csv）
"""

import subprocess
import sys
import os

SCRIPTS = [
    'src/data_analysis.py',
    'src/make_split.py',
    'src/evaluate.py',
    'src/popularity.py',
    'src/bpr_model.py',
]

def run_script(script_path):
    print(f"\n{'='*60}")
    print(f"执行: {script_path}")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable, script_path], cwd=os.getcwd())
    if result.returncode != 0:
        print(f"错误：{script_path} 执行失败，返回码 {result.returncode}")
        return False
    return True

def main():
    print("=" * 60)
    print("成员 A 任务一键运行开始")
    print("=" * 60)

    all_success = True
    for script in SCRIPTS:
        if not run_script(script):
            all_success = False
            break

    print(f"\n{'='*60}")
    if all_success:
        print("本次自动执行的成员 A 脚本全部成功。")
    else:
        print("部分任务执行失败，请检查上方错误信息。")
    print(f"{'='*60}")
    print("\n仍需完成的协作/验收任务：")
    print("1. 待成员 B、C 提供模型结果后，接入统一评估器并更新实验记录")
    print("2. 成员 C 生成最终 submission_final.csv 后，运行：")
    print("   python src/check_submission.py data/submission_final.csv")
    print("3. 更新 reports/experiments.csv，记录所有实验结果")
    return all_success

if __name__ == '__main__':
    raise SystemExit(0 if main() else 1)