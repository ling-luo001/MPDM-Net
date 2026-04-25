# import os
# import librosa
# import soundfile as sf
# from tqdm import tqdm
# import argparse
# import logging
#
#
# def resample_audio(input_path, output_path, target_sr=16000):
#     """将音频文件降采样到目标采样率"""
#     try:
#         # 加载音频文件（自动处理采样率）
#         y, orig_sr = librosa.load(input_path, sr=None, mono=True)
#
#         # 如果已经是目标采样率则跳过
#         if orig_sr == target_sr:
#             logging.info(f"跳过 {input_path}（已经是 {target_sr}Hz）")
#             return
#
#         # 执行降采样
#         y_resampled = librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)
#
#         # 保存降采样后的文件
#         sf.write(output_path, y_resampled, target_sr)
#         logging.info(f"成功处理: {input_path} -> {output_path}")
#
#     except Exception as e:
#         logging.error(f"处理 {input_path} 失败: {str(e)}")
#
#
# def batch_resample(input_dir, output_dir, target_sr=16000):
#     """批量处理目录中的音频文件"""
#     # 创建输出目录
#     os.makedirs(output_dir, exist_ok=True)
#
#     # 获取所有.wav文件（包括子目录）
#     file_list = []
#     for root, _, files in os.walk(input_dir):
#         for file in files:
#             if file.endswith(".wav"):
#                 file_list.append((
#                     os.path.join(root, file),
#                     os.path.join(output_dir, os.path.relpath(root, input_dir), file)
#                 ))
#
#     # 处理所有文件（带进度条）
#     for input_path, output_path in tqdm(file_list, desc="处理进度"):
#         os.makedirs(os.path.dirname(output_path), exist_ok=True)
#         resample_audio(input_path, output_path, target_sr)
#
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="音频降采样工具")
#     parser.add_argument("--input_dir", type=str, default="E:/PycharmProjects/VoiceBank+DEMAND/clean_trainset_wav" , help="输入目录（包含48kHz音频）")
#     parser.add_argument("--output_dir", type=str, default="E:/PycharmProjects/VoiceBank+DEMAND/VCTK_16k/clean_train_16k" , help="输出目录（将保存16kHz音频）")
#     parser.add_argument("--target_sr", type=int, default=16000, help="目标采样率（默认16000）")
#
#     args = parser.parse_args()
#
#     # 配置日志
#     logging.basicConfig(
#         level=logging.INFO,
#         format="%(asctime)s - %(levelname)s - %(message)s",
#         handlers=[
#             logging.FileHandler("resample.log"),
#             logging.StreamHandler()
#         ]
#     )
#
#     # 执行批量处理
#     batch_resample(args.input_dir, args.output_dir, args.target_sr)


"""
划分数据集
import os
import shutil
import argparse
import logging
from sklearn.model_selection import train_test_split

def split_voicebank_testset(
    root_dir="E:/PycharmProjects/VoiceBank+DEMAND",
    valid_ratio=0.7,
    test_ratio=0.3,
    seed=42,
    clean_test_dir="clean_testset_wav",
    noisy_test_dir="noisy_testset_wav",
    clean_valid_dir="clean_valid",
    noisy_valid_dir="noisy_valid",
    clean_new_test_dir="clean_test",
    noisy_new_test_dir="noisy_test"
):
    """
"""
    将 VoiceBank 的测试集划分为验证集和测试集
    Args:
        root_dir (str): 数据集根目录（包含 clean_test/ 和 noisy_test/ 的路径）
        valid_ratio (float): 验证集比例（占原始测试集的比例，默认 0.7）
        test_ratio (float): 测试集比例（占原始测试集的比例，默认 0.3）
        seed (int): 随机种子
    """
"""
    # 检查输入目录是否存在
    original_clean_test = os.path.join(root_dir, clean_test_dir)
    original_noisy_test = os.path.join(root_dir, noisy_test_dir)
    if not os.path.exists(original_clean_test) or not os.path.exists(original_noisy_test):
        raise FileNotFoundError(f"原始测试集目录 {original_clean_test} 或 {original_noisy_test} 不存在")

    # 获取所有干净测试集文件名（假设与带噪语音同名）
    all_clean_files = [f for f in os.listdir(original_clean_test) if f.endswith(".wav")]
    all_noisy_files = [f for f in os.listdir(original_noisy_test) if f.endswith(".wav")]

    # 检查文件名是否一一对应
    if set(all_clean_files) != set(all_noisy_files):
        logging.warning("clean_test 和 noisy_test 中的文件名不完全匹配，可能影响数据配对")

    # 按比例划分文件
    valid_files, test_files = train_test_split(
        all_clean_files,
        test_size=test_ratio,
        random_state=seed
    )

    # 创建目标目录
    os.makedirs(os.path.join(root_dir, clean_valid_dir), exist_ok=True)
    os.makedirs(os.path.join(root_dir, noisy_valid_dir), exist_ok=True)
    os.makedirs(os.path.join(root_dir, clean_new_test_dir), exist_ok=True)
    os.makedirs(os.path.join(root_dir, noisy_new_test_dir), exist_ok=True)

    # 复制验证集文件
    for file in valid_files:
        src_clean = os.path.join(original_clean_test, file)
        src_noisy = os.path.join(original_noisy_test, file)
        if os.path.exists(src_clean) and os.path.exists(src_noisy):
            shutil.copy2(src_clean, os.path.join(root_dir, clean_valid_dir, file))
            shutil.copy2(src_noisy, os.path.join(root_dir, noisy_valid_dir, file))
        else:
            logging.warning(f"文件 {file} 在 clean/noisy 目录中缺失，已跳过")

    # 复制新测试集文件
    for file in test_files:
        src_clean = os.path.join(original_clean_test, file)
        src_noisy = os.path.join(original_noisy_test, file)
        if os.path.exists(src_clean) and os.path.exists(src_noisy):
            shutil.copy2(src_clean, os.path.join(root_dir, clean_new_test_dir, file))
            shutil.copy2(src_noisy, os.path.join(root_dir, noisy_new_test_dir, file))
        else:
            logging.warning(f"文件 {file} 在 clean/noisy 目录中缺失，已跳过")

    logging.info(f"划分完成：验证集 {len(valid_files)} 条，测试集 {len(test_files)} 条")
    logging.info(f"验证集路径：{os.path.join(root_dir, clean_valid_dir)}")
    logging.info(f"新测试集路径：{os.path.join(root_dir, clean_new_test_dir)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True, help="数据集根目录（包含 clean_test/ 和 noisy_test/ 的路径）")
    parser.add_argument("--valid_ratio", type=float, default=0.7, help="验证集比例（占原始测试集的比例，默认 0.7）")
    parser.add_argument("--test_ratio", type=float, default=0.3, help="测试集比例（占原始测试集的比例，默认 0.3）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # 执行划分
    split_voicebank_testset(
        root_dir=args.root_dir,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
"""


import argparse
import os
import json

def list_files_in_directory(directory_path):
    # List all files in the directory
    files = []
    for root, dirs, filenames in os.walk(directory_path):
        for filename in filenames:
            if filename.endswith('.wav'):
                files.append(os.path.join(root, filename))
    return files

def save_files_to_json(files, output_file):
    with open(output_file, 'w') as json_file:
        json.dump(files, json_file, indent=4)

def make_json(directory_path, output_file):
    # Get the list of files and save to JSON
    files = list_files_in_directory(directory_path)
    save_files_to_json(files, output_file)

# create training set json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default="/media/lz/lc/voicebank_16k/")

    args = parser.parse_args()

    prepath = args.path if (args.path is not None) else "../"

    ## train_clean
    make_json(
        os.path.join(prepath, 'clean_train/'),
        'train_clean.json'
    )

    ## train_noisy
    make_json(
        os.path.join(prepath, 'noisy_train/'),
        'train_noisy.json'
    )

    ## valid_clean
    make_json(
         os.path.join(prepath, 'clean_valid/'),
        'valid_clean.json'
    )

    ## valid_noisy
    make_json(
        os.path.join(prepath, 'noisy_valid/'),
        'valid_noisy.json'
    )

    ## test_clean
    make_json(
       os.path.join(prepath, 'clean_test/'),
        'test_clean.json'
    )

    ## test_noisy
    make_json(
       os.path.join(prepath, 'noisy_test/'),
        'test_noisy.json'
    )
    # ----------------------------------------------------------#


if __name__ == '__main__':
    main()
