import glob
import os
import argparse
import json
import torch
import soundfile as sf
import numpy as np
import concurrent.futures

from models.stfts import mag_phase_stft, mag_phase_istft
from datasets.dataset import mag_pha_stft, mag_pha_istft
from cal_metrics.compute_metrics import compute_metrics
from models.generator import MambaSEUNet
from models.pcs400 import cal_pcs
from utils.util import load_config


def load_checkpoint(filepath, device):
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


# 保持原有的分片处理逻辑不变，以确保与你的 Mamba 架构完全兼容
def process_audio_segment(noisy_wav, model, device, n_fft, hop_size, win_size, compress_factor, sampling_rate,
                          segment_size):
    segment_size = segment_size
    n_fft = n_fft
    hop_size = hop_size
    win_size = win_size
    compress_factor = compress_factor
    sampling_rate = sampling_rate

    norm_factor = torch.sqrt(len(noisy_wav) / torch.sum(noisy_wav ** 2.0)).to(device)
    noisy_wav = (noisy_wav * norm_factor).unsqueeze(0)
    orig_size = noisy_wav.size(1)

    if noisy_wav.size(1) >= segment_size:
        num_segments = noisy_wav.size(1) // segment_size
        last_segment_size = noisy_wav.size(1) % segment_size
        if last_segment_size > 0:
            last_segment = noisy_wav[:, -segment_size:]
            noisy_wav = noisy_wav[:, :-last_segment_size]
            segments = torch.split(noisy_wav, segment_size, dim=1)
            segments = list(segments)
            segments.append(last_segment)
            reshapelast = 1
        else:
            segments = torch.split(noisy_wav, segment_size, dim=1)
            reshapelast = 0
    else:
        padded_zeros = torch.zeros(1, segment_size - noisy_wav.size(1)).to(device)
        noisy_wav = torch.cat((noisy_wav, padded_zeros), dim=1)
        segments = [noisy_wav]
        reshapelast = 0

    processed_segments = []

    for i, segment in enumerate(segments):
        noisy_amp, noisy_pha, noisy_com = mag_phase_stft(segment, n_fft, hop_size, win_size, compress_factor)
        amp_g, pha_g, com_g = model(noisy_amp.to(device, non_blocking=True), noisy_pha.to(device, non_blocking=True))
        audio_g = mag_pha_istft(amp_g, pha_g, n_fft, hop_size, win_size, compress_factor)
        audio_g = audio_g / norm_factor
        audio_g = audio_g.squeeze()
        if reshapelast == 1 and i == len(segments) - 2:
            audio_g = audio_g[:-(segment_size - last_segment_size)]

        processed_segments.append(audio_g)

    processed_audio = torch.cat(processed_segments, dim=-1)
    processed_audio = processed_audio[:orig_size]

    return processed_audio


# 定义一个独立的函数用于多进程计算，避免阻塞 GPU
def async_compute_metrics(clean_wav, output_audio, sr):
    # compute_metrics 返回: [pesq, csig, cbak, covl, ssnr, stoi]
    metrics = compute_metrics(clean_wav, output_audio, sr, 0)
    return metrics[0]  # 我们只需要 PESQ


def inference(args, device):
    cfg = load_config(args.config)
    n_fft, hop_size, win_size = cfg['stft_cfg']['n_fft'], cfg['stft_cfg']['hop_size'], cfg['stft_cfg']['win_size']
    compress_factor = cfg['model_cfg']['compress_factor']
    sampling_rate = cfg['stft_cfg']['sampling_rate']
    segment_size = cfg['training_cfg']['segment_size']

    model = MambaSEUNet(cfg).to(device)
    state_dict = load_checkpoint(args.checkpoint_file, device)
    model.load_state_dict(state_dict['generator'])

    model.eval()

    # 初始化用于存储最终结果的字典
    pesq_results_dict = {}

    print(f"Loading paths from {args.input_clean_json} and {args.input_noisy_json}...")
    with open(args.input_clean_json, 'r', encoding='utf-8') as f:
        clean_paths = json.load(f)
    with open(args.input_noisy_json, 'r', encoding='utf-8') as f:
        noisy_paths = json.load(f)

    assert len(clean_paths) == len(noisy_paths), "Error: Clean and Noisy JSON lists have different lengths!"

    # 创建进程池，最大化利用 CPU 核心来并行计算 PESQ
    max_workers = min(os.cpu_count() or 4, 16)  # 防止核心过多导致内存爆炸
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)
    future_to_path = {}

    print(f"Starting inference with async PESQ calculation ({max_workers} CPU workers)...")

    # 使用 inference_mode 代替 no_grad，PyTorch 底层速度更快
    with torch.inference_mode():
        for i, (clean_path, noisy_path) in enumerate(zip(clean_paths, noisy_paths)):
            fname = os.path.basename(clean_path)

            # 使用 soundfile 替代 librosa，I/O 速度提升近 10 倍
            # 注意：soundfile 默认返回 float64，必须转为 float32
            noisy_wav, _ = sf.read(noisy_path)
            noisy_wav = torch.FloatTensor(noisy_wav.astype(np.float32)).to(device)

            # GPU 推理
            output_audio = process_audio_segment(noisy_wav, model, device, n_fft, hop_size, win_size, compress_factor,
                                                 sampling_rate, segment_size)

            if args.post_processing_PCS:
                output_audio = cal_pcs(output_audio.squeeze().cpu().numpy())
            else:
                output_audio = output_audio.squeeze().cpu().numpy()

            # 读取干净语音
            clean_wav, sr = sf.read(clean_path)
            clean_wav = clean_wav.astype(np.float32)

            # 【核心加速点】：不在此处等待 PESQ 计算，而是将其丢进后台进程池
            # 主线程立即进入下一个循环进行下一个音频的 GPU 推理
            future = executor.submit(async_compute_metrics, clean_wav, output_audio, sr)
            future_to_path[future] = noisy_path

            if (i + 1) % 50 == 0:
                print(f"[{i + 1}/{len(clean_paths)}] Inferenced to GPU, PESQ calculating in background...")

    # 等待所有后台进程计算完毕并收集结果
    print("\nAll GPU inference finished. Waiting for remaining background PESQ calculations to complete...")
    for future in concurrent.futures.as_completed(future_to_path):
        noisy_path = future_to_path[future]
        try:
            pesq_score = future.result()
            pesq_results_dict[noisy_path] = float(pesq_score)
        except Exception as exc:
            print(f"File {noisy_path} generated an exception during PESQ calculation: {exc}")

    executor.shutdown()

    # 将结果写入 JSON 字典
    os.makedirs(os.path.dirname(args.output_json_labels), exist_ok=True)
    with open(args.output_json_labels, 'w', encoding='utf-8') as f:
        json.dump(pesq_results_dict, f, indent=4)

    print(f"\n=== SUCCESS! Processed {len(pesq_results_dict)} files. ===")
    print(f"PESQ scores successfully saved mapping to: {args.output_json_labels}")


def main():
    print('Initializing PESQ Label Generation Process..')
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_clean_json', default='../data/train_clean_list.json')
    parser.add_argument('--input_noisy_json', default='../data/train_noisy_list.json')

    # 替换原本的 output_folder 为 output_json_labels
    parser.add_argument('--output_json_labels', default='../data/pesq_labels.json',
                        help='Path to save the resulting PESQ dict')

    parser.add_argument('--config', default='recipes/Mamba-SEUNet/Mamba-SEUNet.yaml')
    parser.add_argument('--checkpoint_file', default='ckpts/g_best.pth')
    parser.add_argument('--post_processing_PCS', default=False)
    args = parser.parse_args()

    global device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        raise RuntimeError("Currently, CPU mode is not supported.")

    inference(args, device)


if __name__ == '__main__':
    main()