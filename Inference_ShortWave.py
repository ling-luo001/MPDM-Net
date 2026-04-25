import os
import argparse
import torch
import librosa
import soundfile as sf
import numpy as np

# 确保这些模块在你当前的环境/目录中可用
from models.stfts import mag_phase_stft, mag_phase_istft
from datasets.dataset import mag_pha_stft, mag_pha_istft
from models.generator import MambaSEUNet
from models.pcs400 import cal_pcs
from utils.util import load_config


def load_checkpoint(filepath, device):
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


# 处理音频切分和拼接的核心函数（保持原逻辑不变）
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

    # 判断是否需要补零
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
        # padding 补零
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

        # 处理最后一段重叠的情况
        if reshapelast == 1 and i == len(segments) - 2:
            audio_g = audio_g[:-(segment_size - last_segment_size)]

        processed_segments.append(audio_g)

    processed_audio = torch.cat(processed_segments, dim=-1)
    processed_audio = processed_audio[:orig_size]

    return processed_audio


def inference(args, device):
    # 加载配置
    cfg = load_config(args.config)
    n_fft, hop_size, win_size = cfg['stft_cfg']['n_fft'], cfg['stft_cfg']['hop_size'], cfg['stft_cfg']['win_size']
    compress_factor = cfg['model_cfg']['compress_factor']
    sampling_rate = cfg['stft_cfg']['sampling_rate']
    segment_size = cfg['training_cfg']['segment_size']

    # 初始化模型并加载权重
    model = MambaSEUNet(cfg).to(device)
    state_dict = load_checkpoint(args.checkpoint_file, device)
    model.load_state_dict(state_dict['generator'])

    # 创建输出文件夹
    os.makedirs(args.output_folder, exist_ok=True)
    model.eval()

    print(f"开始批量降噪，处理文件夹: {args.input_noisy_wavs_dir}")

    with torch.no_grad():
        for fname in os.listdir(args.input_noisy_wavs_dir):
            if not fname.endswith('.wav'):  # 过滤非wav文件
                continue

            print(f"Processing: {fname}")
            # 加载带噪音频
            input_path = os.path.join(args.input_noisy_wavs_dir, fname)
            noisy_wav, _ = librosa.load(input_path, sr=sampling_rate)
            noisy_wav = torch.FloatTensor(noisy_wav).to(device)

            # 模型推理
            output_audio = process_audio_segment(
                noisy_wav, model, device,
                n_fft, hop_size, win_size,
                compress_factor, sampling_rate, segment_size
            )

            # 后处理
            if args.post_processing_PCS:
                # cal_pcs 通常返回 numpy 数组
                output_audio = cal_pcs(output_audio.squeeze().cpu().numpy())
            else:
                # 否则手动转为 numpy 数组
                output_audio = output_audio.squeeze().cpu().numpy()

            # 保存降噪后的音频
            output_file = os.path.join(args.output_folder, fname)
            sf.write(output_file, output_audio, sampling_rate, 'PCM_16')

    print(f"所有音频处理完毕，保存在: {args.output_folder}")


def main():
    parser = argparse.ArgumentParser(description="Mamba-SEUNet 批量音频降噪脚本")
    # 修改为只需传入待降噪的文件夹路径
    parser.add_argument('--input_noisy_wavs_dir', type=str, default='G:/Temp/shortWave_2020_9_23',
                        help="存放需要降噪的音频的文件夹路径")
    parser.add_argument('--output_folder', type=str, default='G:/Temp/shortWave_2020_9_23/1',
                        help="降噪后音频的输出文件夹路径")
    parser.add_argument('--config', type=str, default='recipes/Mamba-SEUNet/Mamba-SEUNet.yaml', help="模型配置文件路径")
    parser.add_argument('--checkpoint_file', type=str, default='G:/Temp/shortWave_2020_9_23/1/g_00584000.pth', help="预训练模型权重路径")
    parser.add_argument('--post_processing_PCS', action='store_true', help="是否开启 PCS 后处理 (加上该参数即为 True)")
    args = parser.parse_args()

    # 环境检查
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        raise RuntimeError("目前模型不支持 CPU 推理，请确保环境配置了 CUDA。")

    inference(args, device)


if __name__ == '__main__':
    main()