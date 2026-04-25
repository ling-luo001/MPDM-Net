# Mamba-SEUNet: Mamba UNet for Monaural Speech Enhancement (Accepted at ICASSP 2025)

**Abstract:** 
 In recent speech enhancement (SE) research, transformer and its variants have emerged as the predominant methodologies. However, the quadratic complexity of the self-attention mechanism imposes certain limitations on practical deployment. Mamba, as a novel state-space model (SSM), has gained widespread application in natural language processing and computer vision due to its strong capabilities in modeling long sequences and relatively low computational complexity. In this work, we introduce Mamba-SEUNet, an innovative architecture that integrates Mamba with U-Net for SE tasks. By leveraging bidirectional Mamba to model forward and backward dependencies of speech signals at different resolutions, and incorporating skip connections to capture multi-scale information, our approach achieves state-of-the-art (SOTA) performance. Experimental results on the VCTK+DEMAND dataset indicate that Mamba-SEUNet attains a PESQ score of 3.59, while maintaining low computational complexity. When combined with the Perceptual Contrast Stretching technique, Mamba-SEUNet further improves the PESQ score to 3.73.

## Pre-requisites
1. Python >= 3.8.
2. Clone this repository.
3. Install python requirements. Please refer requirements.txt.
4. Download and extract the [VoiceBank+DEMAND dataset](https://datashare.ed.ac.uk/handle/10283/1942).

## Training
For single GPU (Recommend), Mamba-SEUNet needs at least 12GB GPU memery.
```
python train.py
```

## Training with your own data
Generate six dataset json files using data/make_dataset_json.py
```
python make_dataset_json.py
```

## Inference
```
python inference.py --checkpoint_file /PATH/TO/YOUR/CHECK_POINT/g_xxxxxxx
```

## Acknowledgements
We referred to [MP-SENet](https://github.com/yxlu-0102/MP-SENet), [MUSE](https://github.com/huaidanquede/MUSE-Speech-Enhancement), [SEMamba](https://github.com/RoyChao19477/SEMamba)

**

只用中间层做完整的多次的交叉注意力，最末端先用下采样的注意力？



 2 x TFMambaBlock(
      (time_mamba): MambaBlock(
        (forward_blocks): ModuleList(
          (0): Block(
            (mixer): Mamba(
              (in_proj): Linear(in_features=16, out_features=128, bias=False)
              (conv1d): Conv1d(64, 64, kernel_size=(4,), stride=(1,), padding=(3,), groups=64)
              (act): SiLU()
              (x_proj): Linear(in_features=64, out_features=33, bias=False)
              (dt_proj): Linear(in_features=1, out_features=64, bias=True)
              (out_proj): Linear(in_features=64, out_features=16, bias=False)
            )
            (norm): RMSNorm()
          )
        )
        (backward_blocks): ModuleList(
          (0): Block(
            (mixer): Mamba(
              (in_proj): Linear(in_features=16, out_features=128, bias=False)
              (conv1d): Conv1d(64, 64, kernel_size=(4,), stride=(1,), padding=(3,), groups=64)
              (act): SiLU()
              (x_proj): Linear(in_features=64, out_features=33, bias=False)
              (dt_proj): Linear(in_features=1, out_features=64, bias=True)
              (out_proj): Linear(in_features=64, out_features=16, bias=False)
            )
            (norm): RMSNorm()
          )
        )
      )
      (freq_mamba): MambaBlock(
        (forward_blocks): ModuleList(
          (0): Block(
            (mixer): Mamba(
              (in_proj): Linear(in_features=16, out_features=128, bias=False)
              (conv1d): Conv1d(64, 64, kernel_size=(4,), stride=(1,), padding=(3,), groups=64)
              (act): SiLU()
              (x_proj): Linear(in_features=64, out_features=33, bias=False)
              (dt_proj): Linear(in_features=1, out_features=64, bias=True)
              (out_proj): Linear(in_features=64, out_features=16, bias=False)
            )
            (norm): RMSNorm()
          )
        )
        (backward_blocks): ModuleList(
          (0): Block(
            (mixer): Mamba(
              (in_proj): Linear(in_features=16, out_features=128, bias=False)
              (conv1d): Conv1d(64, 64, kernel_size=(4,), stride=(1,), padding=(3,), groups=64)
              (act): SiLU()
              (x_proj): Linear(in_features=64, out_features=33, bias=False)
              (dt_proj): Linear(in_features=1, out_features=64, bias=True)
              (out_proj): Linear(in_features=64, out_features=16, bias=False)
            )
            (norm): RMSNorm()
          )
        )
      )
      (tlinear): ConvTranspose1d(32, 16, kernel_size=(1,), stride=(1,))
      (flinear): ConvTranspose1d(32, 16, kernel_size=(1,), stride=(1,))
    )
  )








(mambavision) g515528@515528:~/PycharmProjects/Mamba-SEUNet-main_3TSFM$ python train.py
Number of GPUs available: 1
GPU 0: NVIDIA GeForce RTX 4090
Batch size per GPU: 2
MambaSEUNet(
  (dense_encoder): DenseEncoder(
    (dense_conv_1): Sequential(
      (0): Conv2d(2, 32, kernel_size=(1, 1), stride=(1, 1))
      (1): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
      (2): PReLU(num_parameters=32)
    )
    (dense_block): DenseBlock(
      (dense_block): ModuleList(
        (0): Sequential(
          (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (2): PReLU(num_parameters=32)
        )
        (1): Sequential(
          (0): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(2, 1), dilation=(2, 1))
          (1): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (2): PReLU(num_parameters=32)
        )
        (2): Sequential(
          (0): Conv2d(96, 32, kernel_size=(3, 3), stride=(1, 1), padding=(4, 1), dilation=(4, 1))
          (1): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (2): PReLU(num_parameters=32)
        )
        (3): Sequential(
          (0): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(8, 1), dilation=(8, 1))
          (1): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (2): PReLU(num_parameters=32)
        )
      )
    )
    (dense_conv_2): Sequential(
      (0): Conv2d(32, 32, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
      (1): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
      (2): PReLU(num_parameters=32)
    )
  )
  (patch_embed_encoder_level1): Patch_Embed_stage(
    (patch_embeds): MB_Deform_Embedding(
      (patch_conv): DWConv2d_BN(
        (offset_generator): Sequential(
          (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
          (1): Conv2d(32, 18, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (dcn): DeformConv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
        (pwconv): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (act): Hardswish()
      )
      (norm): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    )
  )
  (TFMamba_attention_encoder1): ModuleList(
    (0-3): 4 x TF_mamba_attention(
      (time_block_mamba): Block_mamba(
        (B_mamba): MambaVisionMixer(
          (in_proj): Linear(in_features=32, out_features=32, bias=False)
          (x_proj): Linear(in_features=16, out_features=18, bias=False)
          (dt_proj): Linear(in_features=2, out_features=16, bias=True)
          (out_proj): Linear(in_features=32, out_features=32, bias=False)
          (conv1d_x): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
          (conv1d_z): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
        )
      )
      (freq_block_mamba): Block_mamba(
        (B_mamba): MambaVisionMixer(
          (in_proj): Linear(in_features=32, out_features=32, bias=False)
          (x_proj): Linear(in_features=16, out_features=18, bias=False)
          (dt_proj): Linear(in_features=2, out_features=16, bias=True)
          (out_proj): Linear(in_features=32, out_features=32, bias=False)
          (conv1d_x): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
          (conv1d_z): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
        )
      )
      (time_block_attention): Block_Attention(
        (B_Attention): Attention(
          (qkv): Linear(in_features=32, out_features=96, bias=True)
          (q_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (k_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (attn_drop): Dropout(p=0, inplace=False)
          (proj): Linear(in_features=32, out_features=32, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
      )
      (freq_block_attention): Block_Attention(
        (B_Attention): Attention(
          (qkv): Linear(in_features=32, out_features=96, bias=True)
          (q_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (k_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (attn_drop): Dropout(p=0, inplace=False)
          (proj): Linear(in_features=32, out_features=32, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
      )
    )
  )
  (down1_2): Downsample(
    (body): Sequential(
      (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
      (1): Conv2d(32, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (2): InstanceNorm2d(16, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      (3): PixelUnshuffle(downscale_factor=2)
    )
  )
  (patch_embed_encoder_level2): Patch_Embed_stage(
    (patch_embeds): MB_Deform_Embedding(
      (patch_conv): DWConv2d_BN(
        (offset_generator): Sequential(
          (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
          (1): Conv2d(64, 18, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (dcn): DeformConv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
        (pwconv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (act): Hardswish()
      )
      (norm): InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    )
  )
  (TFMamba_attention_encoder2): ModuleList(
    (0-3): 4 x TF_mamba_attention(
      (time_block_mamba): Block_mamba(
        (B_mamba): MambaVisionMixer(
          (in_proj): Linear(in_features=64, out_features=64, bias=False)
          (x_proj): Linear(in_features=32, out_features=20, bias=False)
          (dt_proj): Linear(in_features=4, out_features=32, bias=True)
          (out_proj): Linear(in_features=64, out_features=64, bias=False)
          (conv1d_x): Conv1d(32, 32, kernel_size=(3,), stride=(1,), groups=32, bias=False)
          (conv1d_z): Conv1d(32, 32, kernel_size=(3,), stride=(1,), groups=32, bias=False)
        )
      )
      (freq_block_mamba): Block_mamba(
        (B_mamba): MambaVisionMixer(
          (in_proj): Linear(in_features=64, out_features=64, bias=False)
          (x_proj): Linear(in_features=32, out_features=20, bias=False)
          (dt_proj): Linear(in_features=4, out_features=32, bias=True)
          (out_proj): Linear(in_features=64, out_features=64, bias=False)
          (conv1d_x): Conv1d(32, 32, kernel_size=(3,), stride=(1,), groups=32, bias=False)
          (conv1d_z): Conv1d(32, 32, kernel_size=(3,), stride=(1,), groups=32, bias=False)
        )
      )
      (time_block_attention): Block_Attention(
        (B_Attention): Attention(
          (qkv): Linear(in_features=64, out_features=192, bias=True)
          (q_norm): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
          (k_norm): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
          (attn_drop): Dropout(p=0, inplace=False)
          (proj): Linear(in_features=64, out_features=64, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
      )
      (freq_block_attention): Block_Attention(
        (B_Attention): Attention(
          (qkv): Linear(in_features=64, out_features=192, bias=True)
          (q_norm): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
          (k_norm): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
          (attn_drop): Dropout(p=0, inplace=False)
          (proj): Linear(in_features=64, out_features=64, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
      )
    )
  )
  (down2_3): Downsample(
    (body): Sequential(
      (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
      (1): Conv2d(64, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (2): InstanceNorm2d(24, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      (3): PixelUnshuffle(downscale_factor=2)
    )
  )
  (patch_embed_middle): Patch_Embed_stage(
    (patch_embeds): MB_Deform_Embedding(
      (patch_conv): DWConv2d_BN(
        (offset_generator): Sequential(
          (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=96, bias=False)
          (1): Conv2d(96, 18, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (dcn): DeformConv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=96, bias=False)
        (pwconv): Conv2d(96, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (act): Hardswish()
      )
      (norm): InstanceNorm2d(96, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    )
  )
  (TFMamba_attention_middle): ModuleList(
    (0-3): 4 x TF_mamba_attention(
      (time_block_mamba): Block_mamba(
        (B_mamba): MambaVisionMixer(
          (in_proj): Linear(in_features=96, out_features=96, bias=False)
          (x_proj): Linear(in_features=48, out_features=22, bias=False)
          (dt_proj): Linear(in_features=6, out_features=48, bias=True)
          (out_proj): Linear(in_features=96, out_features=96, bias=False)
          (conv1d_x): Conv1d(48, 48, kernel_size=(3,), stride=(1,), groups=48, bias=False)
          (conv1d_z): Conv1d(48, 48, kernel_size=(3,), stride=(1,), groups=48, bias=False)
        )
      )
      (freq_block_mamba): Block_mamba(
        (B_mamba): MambaVisionMixer(
          (in_proj): Linear(in_features=96, out_features=96, bias=False)
          (x_proj): Linear(in_features=48, out_features=22, bias=False)
          (dt_proj): Linear(in_features=6, out_features=48, bias=True)
          (out_proj): Linear(in_features=96, out_features=96, bias=False)
          (conv1d_x): Conv1d(48, 48, kernel_size=(3,), stride=(1,), groups=48, bias=False)
          (conv1d_z): Conv1d(48, 48, kernel_size=(3,), stride=(1,), groups=48, bias=False)
        )
      )
      (time_block_attention): Block_Attention(
        (B_Attention): Attention(
          (qkv): Linear(in_features=96, out_features=288, bias=True)
          (q_norm): LayerNorm((24,), eps=1e-05, elementwise_affine=True)
          (k_norm): LayerNorm((24,), eps=1e-05, elementwise_affine=True)
          (attn_drop): Dropout(p=0, inplace=False)
          (proj): Linear(in_features=96, out_features=96, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
      )
      (freq_block_attention): Block_Attention(
        (B_Attention): Attention(
          (qkv): Linear(in_features=96, out_features=288, bias=True)
          (q_norm): LayerNorm((24,), eps=1e-05, elementwise_affine=True)
          (k_norm): LayerNorm((24,), eps=1e-05, elementwise_affine=True)
          (attn_drop): Dropout(p=0, inplace=False)
          (proj): Linear(in_features=96, out_features=96, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
      )
    )
  )
  (up3_2): Upsample(
    (body): Sequential(
      (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=96, bias=False)
      (1): Conv2d(96, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (2): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      (3): PixelShuffle(upscale_factor=2)
    )
  )
  (concat_level2): Sequential(
    (0): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (patch_embed_decoder_level2): Patch_Embed_stage(
    (patch_embeds): MB_Deform_Embedding(
      (patch_conv): DWConv2d_BN(
        (offset_generator): Sequential(
          (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
          (1): Conv2d(64, 18, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (dcn): DeformConv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
        (pwconv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (act): Hardswish()
      )
      (norm): InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    )
  )
  (TFMamba_attention_decoder2): ModuleList(
    (0-3): 4 x TF_mamba_attention(
      (time_block_mamba): Block_mamba(
        (B_mamba): MambaVisionMixer(
          (in_proj): Linear(in_features=64, out_features=64, bias=False)
          (x_proj): Linear(in_features=32, out_features=20, bias=False)
          (dt_proj): Linear(in_features=4, out_features=32, bias=True)
          (out_proj): Linear(in_features=64, out_features=64, bias=False)
          (conv1d_x): Conv1d(32, 32, kernel_size=(3,), stride=(1,), groups=32, bias=False)
          (conv1d_z): Conv1d(32, 32, kernel_size=(3,), stride=(1,), groups=32, bias=False)
        )
      )
      (freq_block_mamba): Block_mamba(
        (B_mamba): MambaVisionMixer(
          (in_proj): Linear(in_features=64, out_features=64, bias=False)
          (x_proj): Linear(in_features=32, out_features=20, bias=False)
          (dt_proj): Linear(in_features=4, out_features=32, bias=True)
          (out_proj): Linear(in_features=64, out_features=64, bias=False)
          (conv1d_x): Conv1d(32, 32, kernel_size=(3,), stride=(1,), groups=32, bias=False)
          (conv1d_z): Conv1d(32, 32, kernel_size=(3,), stride=(1,), groups=32, bias=False)
        )
      )
      (time_block_attention): Block_Attention(
        (B_Attention): Attention(
          (qkv): Linear(in_features=64, out_features=192, bias=True)
          (q_norm): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
          (k_norm): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
          (attn_drop): Dropout(p=0, inplace=False)
          (proj): Linear(in_features=64, out_features=64, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
      )
      (freq_block_attention): Block_Attention(
        (B_Attention): Attention(
          (qkv): Linear(in_features=64, out_features=192, bias=True)
          (q_norm): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
          (k_norm): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
          (attn_drop): Dropout(p=0, inplace=False)
          (proj): Linear(in_features=64, out_features=64, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
      )
    )
  )
  (up2_1): Upsample(
    (body): Sequential(
      (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
      (1): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (2): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      (3): PixelShuffle(upscale_factor=2)
    )
  )
  (concat_level1): Sequential(
    (0): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (patch_embed_decoder_level1): Patch_Embed_stage(
    (patch_embeds): MB_Deform_Embedding(
      (patch_conv): DWConv2d_BN(
        (offset_generator): Sequential(
          (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
          (1): Conv2d(32, 18, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (dcn): DeformConv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
        (pwconv): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (act): Hardswish()
      )
      (norm): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    )
  )
  (TFMamba_attention_decoder1): ModuleList(
    (0-3): 4 x TF_mamba_attention(
      (time_block_mamba): Block_mamba(
        (B_mamba): MambaVisionMixer(
          (in_proj): Linear(in_features=32, out_features=32, bias=False)
          (x_proj): Linear(in_features=16, out_features=18, bias=False)
          (dt_proj): Linear(in_features=2, out_features=16, bias=True)
          (out_proj): Linear(in_features=32, out_features=32, bias=False)
          (conv1d_x): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
          (conv1d_z): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
        )
      )
      (freq_block_mamba): Block_mamba(
        (B_mamba): MambaVisionMixer(
          (in_proj): Linear(in_features=32, out_features=32, bias=False)
          (x_proj): Linear(in_features=16, out_features=18, bias=False)
          (dt_proj): Linear(in_features=2, out_features=16, bias=True)
          (out_proj): Linear(in_features=32, out_features=32, bias=False)
          (conv1d_x): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
          (conv1d_z): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
        )
      )
      (time_block_attention): Block_Attention(
        (B_Attention): Attention(
          (qkv): Linear(in_features=32, out_features=96, bias=True)
          (q_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (k_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (attn_drop): Dropout(p=0, inplace=False)
          (proj): Linear(in_features=32, out_features=32, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
      )
      (freq_block_attention): Block_Attention(
        (B_Attention): Attention(
          (qkv): Linear(in_features=32, out_features=96, bias=True)
          (q_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (k_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (attn_drop): Dropout(p=0, inplace=False)
          (proj): Linear(in_features=32, out_features=32, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
      )
    )
  )
  (mag_patch_embed_refinement): Patch_Embed_stage(
    (patch_embeds): MB_Deform_Embedding(
      (patch_conv): DWConv2d_BN(
        (offset_generator): Sequential(
          (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
          (1): Conv2d(32, 18, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (dcn): DeformConv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
        (pwconv): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (act): Hardswish()
      )
      (norm): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    )
  )
  (mag_refinement): ModuleList(
    (0-3): 4 x TF_mamba_attention(
      (time_block_mamba): Block_mamba(
        (B_mamba): MambaVisionMixer(
          (in_proj): Linear(in_features=32, out_features=32, bias=False)
          (x_proj): Linear(in_features=16, out_features=18, bias=False)
          (dt_proj): Linear(in_features=2, out_features=16, bias=True)
          (out_proj): Linear(in_features=32, out_features=32, bias=False)
          (conv1d_x): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
          (conv1d_z): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
        )
      )
      (freq_block_mamba): Block_mamba(
        (B_mamba): MambaVisionMixer(
          (in_proj): Linear(in_features=32, out_features=32, bias=False)
          (x_proj): Linear(in_features=16, out_features=18, bias=False)
          (dt_proj): Linear(in_features=2, out_features=16, bias=True)
          (out_proj): Linear(in_features=32, out_features=32, bias=False)
          (conv1d_x): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
          (conv1d_z): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
        )
      )
      (time_block_attention): Block_Attention(
        (B_Attention): Attention(
          (qkv): Linear(in_features=32, out_features=96, bias=True)
          (q_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (k_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (attn_drop): Dropout(p=0, inplace=False)
          (proj): Linear(in_features=32, out_features=32, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
      )
      (freq_block_attention): Block_Attention(
        (B_Attention): Attention(
          (qkv): Linear(in_features=32, out_features=96, bias=True)
          (q_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (k_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (attn_drop): Dropout(p=0, inplace=False)
          (proj): Linear(in_features=32, out_features=32, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
      )
    )
  )
  (mag_output): Sequential(
    (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  )
  (pha_patch_embed_refinement): Patch_Embed_stage(
    (patch_embeds): MB_Deform_Embedding(
      (patch_conv): DWConv2d_BN(
        (offset_generator): Sequential(
          (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
          (1): Conv2d(32, 18, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (dcn): DeformConv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
        (pwconv): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (act): Hardswish()
      )
      (norm): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    )
  )
  (pha_refinement): ModuleList(
    (0-3): 4 x TF_mamba_attention(
      (time_block_mamba): Block_mamba(
        (B_mamba): MambaVisionMixer(
          (in_proj): Linear(in_features=32, out_features=32, bias=False)
          (x_proj): Linear(in_features=16, out_features=18, bias=False)
          (dt_proj): Linear(in_features=2, out_features=16, bias=True)
          (out_proj): Linear(in_features=32, out_features=32, bias=False)
          (conv1d_x): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
          (conv1d_z): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
        )
      )
      (freq_block_mamba): Block_mamba(
        (B_mamba): MambaVisionMixer(
          (in_proj): Linear(in_features=32, out_features=32, bias=False)
          (x_proj): Linear(in_features=16, out_features=18, bias=False)
          (dt_proj): Linear(in_features=2, out_features=16, bias=True)
          (out_proj): Linear(in_features=32, out_features=32, bias=False)
          (conv1d_x): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
          (conv1d_z): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
        )
      )
      (time_block_attention): Block_Attention(
        (B_Attention): Attention(
          (qkv): Linear(in_features=32, out_features=96, bias=True)
          (q_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (k_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (attn_drop): Dropout(p=0, inplace=False)
          (proj): Linear(in_features=32, out_features=32, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
      )
      (freq_block_attention): Block_Attention(
        (B_Attention): Attention(
          (qkv): Linear(in_features=32, out_features=96, bias=True)
          (q_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (k_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (attn_drop): Dropout(p=0, inplace=False)
          (proj): Linear(in_features=32, out_features=32, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
      )
    )
  )
  (pha_output): Sequential(
    (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  )
  (mask_decoder): MagDecoder(
    (dense_block): DenseBlock(
      (dense_block): ModuleList(
        (0): Sequential(
          (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (2): PReLU(num_parameters=32)
        )
        (1): Sequential(
          (0): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(2, 1), dilation=(2, 1))
          (1): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (2): PReLU(num_parameters=32)
        )
        (2): Sequential(
          (0): Conv2d(96, 32, kernel_size=(3, 3), stride=(1, 1), padding=(4, 1), dilation=(4, 1))
          (1): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (2): PReLU(num_parameters=32)
        )
        (3): Sequential(
          (0): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(8, 1), dilation=(8, 1))
          (1): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (2): PReLU(num_parameters=32)
        )
      )
    )
    (mask_conv): Sequential(
      (0): Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (1): PixelShuffle(upscale_factor=2)
      (2): Conv2d(32, 32, kernel_size=(1, 3), stride=(2, 1), padding=(0, 1), groups=32, bias=False)
      (3): Conv2d(32, 1, kernel_size=(1, 1), stride=(1, 1))
      (4): InstanceNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
      (5): PReLU(num_parameters=1)
      (6): Conv2d(1, 1, kernel_size=(1, 1), stride=(1, 1))
    )
    (lsigmoid): LearnableSigmoid2D()
  )
  (phase_decoder): PhaseDecoder(
    (dense_block): DenseBlock(
      (dense_block): ModuleList(
        (0): Sequential(
          (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (2): PReLU(num_parameters=32)
        )
        (1): Sequential(
          (0): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(2, 1), dilation=(2, 1))
          (1): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (2): PReLU(num_parameters=32)
        )
        (2): Sequential(
          (0): Conv2d(96, 32, kernel_size=(3, 3), stride=(1, 1), padding=(4, 1), dilation=(4, 1))
          (1): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (2): PReLU(num_parameters=32)
        )
        (3): Sequential(
          (0): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(8, 1), dilation=(8, 1))
          (1): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (2): PReLU(num_parameters=32)
        )
      )
    )
    (phase_conv): Sequential(
      (0): Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (1): PixelShuffle(upscale_factor=2)
      (2): Conv2d(32, 32, kernel_size=(1, 3), stride=(2, 1), padding=(0, 1), groups=32, bias=False)
      (3): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
      (4): PReLU(num_parameters=32)
    )
    (phase_conv_r): Conv2d(32, 1, kernel_size=(1, 1), stride=(1, 1))
    (phase_conv_i): Conv2d(32, 1, kernel_size=(1, 1), stride=(1, 1))
  )
)
Generator Parameters : 1492296
checkpoints directory : exp/Mambavision_emb_07
Epoch: 1
Steps : 0, Gen Loss:  nan, Disc Loss: 0.232, Metric Loss:  nan, Mag Loss:  nan, Pha Loss:  nan, Com Loss:  nan, Time Loss:  nan, Cons Loss:  nan, s/b : 1.761
[rank0]: Traceback (most recent call last):
[rank0]:   File "/home/g515528/PycharmProjects/Mamba-SEUNet-main_3TSFM/train.py", line 485, in <module>
[rank0]:     main()
[rank0]:   File "/home/g515528/PycharmProjects/Mamba-SEUNet-main_3TSFM/train.py", line 482, in main
[rank0]:     train(0, args, cfg)
[rank0]:   File "/home/g515528/PycharmProjects/Mamba-SEUNet-main_3TSFM/train.py", line 322, in train
[rank0]:     raise ValueError("NaN values found in loss_gen_all")
[rank0]: ValueError: NaN values found in loss_gen_all
(mambavision) g515528@515528:~/PycharmProjects/Mamba-SEUNet-main_3TSFM$ python train.py
Number of GPUs available: 1
GPU 0: NVIDIA GeForce RTX 4090
Batch size per GPU: 2
MambaSEUNet(
  (dense_encoder): DenseEncoder(
    (dense_conv_1): Sequential(
      (0): Conv2d(2, 32, kernel_size=(1, 1), stride=(1, 1))
      (1): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
      (2): PReLU(num_parameters=32)
    )
    (dense_block): DenseBlock(
      (dense_block): ModuleList(
        (0): Sequential(
          (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (2): PReLU(num_parameters=32)
        )
        (1): Sequential(
          (0): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(2, 1), dilation=(2, 1))
          (1): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (2): PReLU(num_parameters=32)
        )
        (2): Sequential(
          (0): Conv2d(96, 32, kernel_size=(3, 3), stride=(1, 1), padding=(4, 1), dilation=(4, 1))
          (1): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (2): PReLU(num_parameters=32)
        )
        (3): Sequential(
          (0): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(8, 1), dilation=(8, 1))
          (1): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (2): PReLU(num_parameters=32)
        )
      )
    )
    (dense_conv_2): Sequential(
      (0): Conv2d(32, 32, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
      (1): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
      (2): PReLU(num_parameters=32)
    )
  )
  (patch_embed_encoder_level1): Patch_Embed_stage(
    (patch_embeds): MB_Deform_Embedding(
      (patch_conv): DWConv2d_BN(
        (offset_generator): Sequential(
          (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
          (1): Conv2d(32, 18, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (dcn): DeformConv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
        (pwconv): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (act): Hardswish()
      )
      (norm): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (TFMamba_attention_encoder1): ModuleList(
    (0-3): 4 x TF_mamba_attention(
      (time_block_mamba): Block_mamba(
        (B_mamba): MambaVisionMixer(
          (in_proj): Linear(in_features=32, out_features=32, bias=False)
          (x_proj): Linear(in_features=16, out_features=18, bias=False)
          (dt_proj): Linear(in_features=2, out_features=16, bias=True)
          (out_proj): Linear(in_features=32, out_features=32, bias=False)
          (conv1d_x): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
          (conv1d_z): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
        )
      )
      (freq_block_mamba): Block_mamba(
        (B_mamba): MambaVisionMixer(
          (in_proj): Linear(in_features=32, out_features=32, bias=False)
          (x_proj): Linear(in_features=16, out_features=18, bias=False)
          (dt_proj): Linear(in_features=2, out_features=16, bias=True)
          (out_proj): Linear(in_features=32, out_features=32, bias=False)
          (conv1d_x): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
          (conv1d_z): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
        )
      )
      (time_block_attention): Block_Attention(
        (B_Attention): Attention(
          (qkv): Linear(in_features=32, out_features=96, bias=True)
          (q_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (k_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (attn_drop): Dropout(p=0, inplace=False)
          (proj): Linear(in_features=32, out_features=32, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
      )
      (freq_block_attention): Block_Attention(
        (B_Attention): Attention(
          (qkv): Linear(in_features=32, out_features=96, bias=True)
          (q_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (k_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (attn_drop): Dropout(p=0, inplace=False)
          (proj): Linear(in_features=32, out_features=32, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
      )
    )
  )
  (down1_2): Downsample(
    (body): Sequential(
      (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
      (1): Conv2d(32, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (3): PixelUnshuffle(downscale_factor=2)
    )
  )
  (patch_embed_encoder_level2): Patch_Embed_stage(
    (patch_embeds): MB_Deform_Embedding(
      (patch_conv): DWConv2d_BN(
        (offset_generator): Sequential(
          (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
          (1): Conv2d(64, 18, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (dcn): DeformConv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
        (pwconv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (act): Hardswish()
      )
      (norm): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (TFMamba_attention_encoder2): ModuleList(
    (0-3): 4 x TF_mamba_attention(
      (time_block_mamba): Block_mamba(
        (B_mamba): MambaVisionMixer(
          (in_proj): Linear(in_features=64, out_features=64, bias=False)
          (x_proj): Linear(in_features=32, out_features=20, bias=False)
          (dt_proj): Linear(in_features=4, out_features=32, bias=True)
          (out_proj): Linear(in_features=64, out_features=64, bias=False)
          (conv1d_x): Conv1d(32, 32, kernel_size=(3,), stride=(1,), groups=32, bias=False)
          (conv1d_z): Conv1d(32, 32, kernel_size=(3,), stride=(1,), groups=32, bias=False)
        )
      )
      (freq_block_mamba): Block_mamba(
        (B_mamba): MambaVisionMixer(
          (in_proj): Linear(in_features=64, out_features=64, bias=False)
          (x_proj): Linear(in_features=32, out_features=20, bias=False)
          (dt_proj): Linear(in_features=4, out_features=32, bias=True)
          (out_proj): Linear(in_features=64, out_features=64, bias=False)
          (conv1d_x): Conv1d(32, 32, kernel_size=(3,), stride=(1,), groups=32, bias=False)
          (conv1d_z): Conv1d(32, 32, kernel_size=(3,), stride=(1,), groups=32, bias=False)
        )
      )
      (time_block_attention): Block_Attention(
        (B_Attention): Attention(
          (qkv): Linear(in_features=64, out_features=192, bias=True)
          (q_norm): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
          (k_norm): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
          (attn_drop): Dropout(p=0, inplace=False)
          (proj): Linear(in_features=64, out_features=64, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
      )
      (freq_block_attention): Block_Attention(
        (B_Attention): Attention(
          (qkv): Linear(in_features=64, out_features=192, bias=True)
          (q_norm): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
          (k_norm): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
          (attn_drop): Dropout(p=0, inplace=False)
          (proj): Linear(in_features=64, out_features=64, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
      )
    )
  )
  (down2_3): Downsample(
    (body): Sequential(
      (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
      (1): Conv2d(64, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (2): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (3): PixelUnshuffle(downscale_factor=2)
    )
  )
  (patch_embed_middle): Patch_Embed_stage(
    (patch_embeds): MB_Deform_Embedding(
      (patch_conv): DWConv2d_BN(
        (offset_generator): Sequential(
          (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=96, bias=False)
          (1): Conv2d(96, 18, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (dcn): DeformConv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=96, bias=False)
        (pwconv): Conv2d(96, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (act): Hardswish()
      )
      (norm): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (TFMamba_attention_middle): ModuleList(
    (0-3): 4 x TF_mamba_attention(
      (time_block_mamba): Block_mamba(
        (B_mamba): MambaVisionMixer(
          (in_proj): Linear(in_features=96, out_features=96, bias=False)
          (x_proj): Linear(in_features=48, out_features=22, bias=False)
          (dt_proj): Linear(in_features=6, out_features=48, bias=True)
          (out_proj): Linear(in_features=96, out_features=96, bias=False)
          (conv1d_x): Conv1d(48, 48, kernel_size=(3,), stride=(1,), groups=48, bias=False)
          (conv1d_z): Conv1d(48, 48, kernel_size=(3,), stride=(1,), groups=48, bias=False)
        )
      )
      (freq_block_mamba): Block_mamba(
        (B_mamba): MambaVisionMixer(
          (in_proj): Linear(in_features=96, out_features=96, bias=False)
          (x_proj): Linear(in_features=48, out_features=22, bias=False)
          (dt_proj): Linear(in_features=6, out_features=48, bias=True)
          (out_proj): Linear(in_features=96, out_features=96, bias=False)
          (conv1d_x): Conv1d(48, 48, kernel_size=(3,), stride=(1,), groups=48, bias=False)
          (conv1d_z): Conv1d(48, 48, kernel_size=(3,), stride=(1,), groups=48, bias=False)
        )
      )
      (time_block_attention): Block_Attention(
        (B_Attention): Attention(
          (qkv): Linear(in_features=96, out_features=288, bias=True)
          (q_norm): LayerNorm((24,), eps=1e-05, elementwise_affine=True)
          (k_norm): LayerNorm((24,), eps=1e-05, elementwise_affine=True)
          (attn_drop): Dropout(p=0, inplace=False)
          (proj): Linear(in_features=96, out_features=96, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
      )
      (freq_block_attention): Block_Attention(
        (B_Attention): Attention(
          (qkv): Linear(in_features=96, out_features=288, bias=True)
          (q_norm): LayerNorm((24,), eps=1e-05, elementwise_affine=True)
          (k_norm): LayerNorm((24,), eps=1e-05, elementwise_affine=True)
          (attn_drop): Dropout(p=0, inplace=False)
          (proj): Linear(in_features=96, out_features=96, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
      )
    )
  )
  (up3_2): Upsample(
    (body): Sequential(
      (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=96, bias=False)
      (1): Conv2d(96, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (3): PixelShuffle(upscale_factor=2)
    )
  )
  (concat_level2): Sequential(
    (0): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (patch_embed_decoder_level2): Patch_Embed_stage(
    (patch_embeds): MB_Deform_Embedding(
      (patch_conv): DWConv2d_BN(
        (offset_generator): Sequential(
          (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
          (1): Conv2d(64, 18, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (dcn): DeformConv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
        (pwconv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (act): Hardswish()
      )
      (norm): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (TFMamba_attention_decoder2): ModuleList(
    (0-3): 4 x TF_mamba_attention(
      (time_block_mamba): Block_mamba(
        (B_mamba): MambaVisionMixer(
          (in_proj): Linear(in_features=64, out_features=64, bias=False)
          (x_proj): Linear(in_features=32, out_features=20, bias=False)
          (dt_proj): Linear(in_features=4, out_features=32, bias=True)
          (out_proj): Linear(in_features=64, out_features=64, bias=False)
          (conv1d_x): Conv1d(32, 32, kernel_size=(3,), stride=(1,), groups=32, bias=False)
          (conv1d_z): Conv1d(32, 32, kernel_size=(3,), stride=(1,), groups=32, bias=False)
        )
      )
      (freq_block_mamba): Block_mamba(
        (B_mamba): MambaVisionMixer(
          (in_proj): Linear(in_features=64, out_features=64, bias=False)
          (x_proj): Linear(in_features=32, out_features=20, bias=False)
          (dt_proj): Linear(in_features=4, out_features=32, bias=True)
          (out_proj): Linear(in_features=64, out_features=64, bias=False)
          (conv1d_x): Conv1d(32, 32, kernel_size=(3,), stride=(1,), groups=32, bias=False)
          (conv1d_z): Conv1d(32, 32, kernel_size=(3,), stride=(1,), groups=32, bias=False)
        )
      )
      (time_block_attention): Block_Attention(
        (B_Attention): Attention(
          (qkv): Linear(in_features=64, out_features=192, bias=True)
          (q_norm): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
          (k_norm): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
          (attn_drop): Dropout(p=0, inplace=False)
          (proj): Linear(in_features=64, out_features=64, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
      )
      (freq_block_attention): Block_Attention(
        (B_Attention): Attention(
          (qkv): Linear(in_features=64, out_features=192, bias=True)
          (q_norm): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
          (k_norm): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
          (attn_drop): Dropout(p=0, inplace=False)
          (proj): Linear(in_features=64, out_features=64, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
      )
    )
  )
  (up2_1): Upsample(
    (body): Sequential(
      (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
      (1): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (3): PixelShuffle(upscale_factor=2)
    )
  )
  (concat_level1): Sequential(
    (0): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (patch_embed_decoder_level1): Patch_Embed_stage(
    (patch_embeds): MB_Deform_Embedding(
      (patch_conv): DWConv2d_BN(
        (offset_generator): Sequential(
          (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
          (1): Conv2d(32, 18, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (dcn): DeformConv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
        (pwconv): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (act): Hardswish()
      )
      (norm): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (TFMamba_attention_decoder1): ModuleList(
    (0-3): 4 x TF_mamba_attention(
      (time_block_mamba): Block_mamba(
        (B_mamba): MambaVisionMixer(
          (in_proj): Linear(in_features=32, out_features=32, bias=False)
          (x_proj): Linear(in_features=16, out_features=18, bias=False)
          (dt_proj): Linear(in_features=2, out_features=16, bias=True)
          (out_proj): Linear(in_features=32, out_features=32, bias=False)
          (conv1d_x): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
          (conv1d_z): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
        )
      )
      (freq_block_mamba): Block_mamba(
        (B_mamba): MambaVisionMixer(
          (in_proj): Linear(in_features=32, out_features=32, bias=False)
          (x_proj): Linear(in_features=16, out_features=18, bias=False)
          (dt_proj): Linear(in_features=2, out_features=16, bias=True)
          (out_proj): Linear(in_features=32, out_features=32, bias=False)
          (conv1d_x): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
          (conv1d_z): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
        )
      )
      (time_block_attention): Block_Attention(
        (B_Attention): Attention(
          (qkv): Linear(in_features=32, out_features=96, bias=True)
          (q_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (k_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (attn_drop): Dropout(p=0, inplace=False)
          (proj): Linear(in_features=32, out_features=32, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
      )
      (freq_block_attention): Block_Attention(
        (B_Attention): Attention(
          (qkv): Linear(in_features=32, out_features=96, bias=True)
          (q_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (k_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (attn_drop): Dropout(p=0, inplace=False)
          (proj): Linear(in_features=32, out_features=32, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
      )
    )
  )
  (mag_patch_embed_refinement): Patch_Embed_stage(
    (patch_embeds): MB_Deform_Embedding(
      (patch_conv): DWConv2d_BN(
        (offset_generator): Sequential(
          (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
          (1): Conv2d(32, 18, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (dcn): DeformConv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
        (pwconv): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (act): Hardswish()
      )
      (norm): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (mag_refinement): ModuleList(
    (0-3): 4 x TF_mamba_attention(
      (time_block_mamba): Block_mamba(
        (B_mamba): MambaVisionMixer(
          (in_proj): Linear(in_features=32, out_features=32, bias=False)
          (x_proj): Linear(in_features=16, out_features=18, bias=False)
          (dt_proj): Linear(in_features=2, out_features=16, bias=True)
          (out_proj): Linear(in_features=32, out_features=32, bias=False)
          (conv1d_x): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
          (conv1d_z): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
        )
      )
      (freq_block_mamba): Block_mamba(
        (B_mamba): MambaVisionMixer(
          (in_proj): Linear(in_features=32, out_features=32, bias=False)
          (x_proj): Linear(in_features=16, out_features=18, bias=False)
          (dt_proj): Linear(in_features=2, out_features=16, bias=True)
          (out_proj): Linear(in_features=32, out_features=32, bias=False)
          (conv1d_x): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
          (conv1d_z): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
        )
      )
      (time_block_attention): Block_Attention(
        (B_Attention): Attention(
          (qkv): Linear(in_features=32, out_features=96, bias=True)
          (q_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (k_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (attn_drop): Dropout(p=0, inplace=False)
          (proj): Linear(in_features=32, out_features=32, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
      )
      (freq_block_attention): Block_Attention(
        (B_Attention): Attention(
          (qkv): Linear(in_features=32, out_features=96, bias=True)
          (q_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (k_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (attn_drop): Dropout(p=0, inplace=False)
          (proj): Linear(in_features=32, out_features=32, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
      )
    )
  )
  (mag_output): Sequential(
    (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  )
  (pha_patch_embed_refinement): Patch_Embed_stage(
    (patch_embeds): MB_Deform_Embedding(
      (patch_conv): DWConv2d_BN(
        (offset_generator): Sequential(
          (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
          (1): Conv2d(32, 18, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (dcn): DeformConv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
        (pwconv): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (act): Hardswish()
      )
      (norm): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (pha_refinement): ModuleList(
    (0-3): 4 x TF_mamba_attention(
      (time_block_mamba): Block_mamba(
        (B_mamba): MambaVisionMixer(
          (in_proj): Linear(in_features=32, out_features=32, bias=False)
          (x_proj): Linear(in_features=16, out_features=18, bias=False)
          (dt_proj): Linear(in_features=2, out_features=16, bias=True)
          (out_proj): Linear(in_features=32, out_features=32, bias=False)
          (conv1d_x): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
          (conv1d_z): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
        )
      )
      (freq_block_mamba): Block_mamba(
        (B_mamba): MambaVisionMixer(
          (in_proj): Linear(in_features=32, out_features=32, bias=False)
          (x_proj): Linear(in_features=16, out_features=18, bias=False)
          (dt_proj): Linear(in_features=2, out_features=16, bias=True)
          (out_proj): Linear(in_features=32, out_features=32, bias=False)
          (conv1d_x): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
          (conv1d_z): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
        )
      )
      (time_block_attention): Block_Attention(
        (B_Attention): Attention(
          (qkv): Linear(in_features=32, out_features=96, bias=True)
          (q_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (k_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (attn_drop): Dropout(p=0, inplace=False)
          (proj): Linear(in_features=32, out_features=32, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
      )
      (freq_block_attention): Block_Attention(
        (B_Attention): Attention(
          (qkv): Linear(in_features=32, out_features=96, bias=True)
          (q_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (k_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (attn_drop): Dropout(p=0, inplace=False)
          (proj): Linear(in_features=32, out_features=32, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
      )
    )
  )
  (pha_output): Sequential(
    (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  )
  (mask_decoder): MagDecoder(
    (dense_block): DenseBlock(
      (dense_block): ModuleList(
        (0): Sequential(
          (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (2): PReLU(num_parameters=32)
        )
        (1): Sequential(
          (0): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(2, 1), dilation=(2, 1))
          (1): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (2): PReLU(num_parameters=32)
        )
        (2): Sequential(
          (0): Conv2d(96, 32, kernel_size=(3, 3), stride=(1, 1), padding=(4, 1), dilation=(4, 1))
          (1): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (2): PReLU(num_parameters=32)
        )
        (3): Sequential(
          (0): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(8, 1), dilation=(8, 1))
          (1): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (2): PReLU(num_parameters=32)
        )
      )
    )
    (mask_conv): Sequential(
      (0): Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (1): PixelShuffle(upscale_factor=2)
      (2): Conv2d(32, 32, kernel_size=(1, 3), stride=(2, 1), padding=(0, 1), groups=32, bias=False)
      (3): Conv2d(32, 1, kernel_size=(1, 1), stride=(1, 1))
      (4): InstanceNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
      (5): PReLU(num_parameters=1)
      (6): Conv2d(1, 1, kernel_size=(1, 1), stride=(1, 1))
    )
    (lsigmoid): LearnableSigmoid2D()
  )
  (phase_decoder): PhaseDecoder(
    (dense_block): DenseBlock(
      (dense_block): ModuleList(
        (0): Sequential(
          (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (2): PReLU(num_parameters=32)
        )
        (1): Sequential(
          (0): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(2, 1), dilation=(2, 1))
          (1): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (2): PReLU(num_parameters=32)
        )
        (2): Sequential(
          (0): Conv2d(96, 32, kernel_size=(3, 3), stride=(1, 1), padding=(4, 1), dilation=(4, 1))
          (1): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (2): PReLU(num_parameters=32)
        )
        (3): Sequential(
          (0): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(8, 1), dilation=(8, 1))
          (1): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (2): PReLU(num_parameters=32)
        )
      )
    )
    (phase_conv): Sequential(
      (0): Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (1): PixelShuffle(upscale_factor=2)
      (2): Conv2d(32, 32, kernel_size=(1, 3), stride=(2, 1), padding=(0, 1), groups=32, bias=False)
      (3): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
      (4): PReLU(num_parameters=32)
    )
    (phase_conv_r): Conv2d(32, 1, kernel_size=(1, 1), stride=(1, 1))
    (phase_conv_i): Conv2d(32, 1, kernel_size=(1, 1), stride=(1, 1))
  )
)
Generator Parameters : 1493848
checkpoints directory : exp/Mambavision_emb_07
Epoch: 1
Steps : 0, Gen Loss:  nan, Disc Loss: 0.232, Metric Loss:  nan, Mag Loss:  nan, Pha Loss:  nan, Com Loss:  nan, Time Loss:  nan, Cons Loss:  nan, s/b : 1.786
[rank0]: Traceback (most recent call last):
[rank0]:   File "/home/g515528/PycharmProjects/Mamba-SEUNet-main_3TSFM/train.py", line 485, in <module>
[rank0]:     main()
[rank0]:   File "/home/g515528/PycharmProjects/Mamba-SEUNet-main_3TSFM/train.py", line 482, in main
[rank0]:     train(0, args, cfg)
[rank0]:   File "/home/g515528/PycharmProjects/Mamba-SEUNet-main_3TSFM/train.py", line 322, in train
[rank0]:     raise ValueError("NaN values found in loss_gen_all")
[rank0]: ValueError: NaN values found in loss_gen_all
(mambavision) g515528@515528:~/PycharmProjects/Mamba-SEUNet-main_3TSFM$ python train.py
Number of GPUs available: 1
GPU 0: NVIDIA GeForce RTX 4090
Batch size per GPU: 2
MambaSEUNet(
  (dense_encoder): DenseEncoder(
    (dense_conv_1): Sequential(
      (0): Conv2d(2, 32, kernel_size=(1, 1), stride=(1, 1))
      (1): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
      (2): PReLU(num_parameters=32)
    )
    (dense_block): DenseBlock(
      (dense_block): ModuleList(
        (0): Sequential(
          (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (2): PReLU(num_parameters=32)
        )
        (1): Sequential(
          (0): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(2, 1), dilation=(2, 1))
          (1): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (2): PReLU(num_parameters=32)
        )
        (2): Sequential(
          (0): Conv2d(96, 32, kernel_size=(3, 3), stride=(1, 1), padding=(4, 1), dilation=(4, 1))
          (1): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (2): PReLU(num_parameters=32)
        )
        (3): Sequential(
          (0): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(8, 1), dilation=(8, 1))
          (1): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (2): PReLU(num_parameters=32)
        )
      )
    )
    (dense_conv_2): Sequential(
      (0): Conv2d(32, 32, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
      (1): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
      (2): PReLU(num_parameters=32)
    )
  )
  (patch_embed_encoder_level1): Patch_Embed_stage(
    (patch_embeds): MB_Deform_Embedding(
      (patch_conv): DWConv2d_BN(
        (offset_generator): Sequential(
          (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
          (1): Conv2d(32, 18, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (dcn): DeformConv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
        (pwconv): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (act): Hardswish()
      )
      (norm): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (TFMamba_attention_encoder1): ModuleList(
    (0-3): 4 x TF_mamba_attention(
      (time_block_mamba): Block_mamba(
        (norm1): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (B_mamba): MambaVisionMixer(
          (in_proj): Linear(in_features=32, out_features=32, bias=False)
          (x_proj): Linear(in_features=16, out_features=18, bias=False)
          (dt_proj): Linear(in_features=2, out_features=16, bias=True)
          (out_proj): Linear(in_features=32, out_features=32, bias=False)
          (conv1d_x): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
          (conv1d_z): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
        )
      )
      (freq_block_mamba): Block_mamba(
        (norm1): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (B_mamba): MambaVisionMixer(
          (in_proj): Linear(in_features=32, out_features=32, bias=False)
          (x_proj): Linear(in_features=16, out_features=18, bias=False)
          (dt_proj): Linear(in_features=2, out_features=16, bias=True)
          (out_proj): Linear(in_features=32, out_features=32, bias=False)
          (conv1d_x): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
          (conv1d_z): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
        )
      )
      (time_block_attention): Block_Attention(
        (norm1): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (B_Attention): Attention(
          (qkv): Linear(in_features=32, out_features=96, bias=True)
          (q_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (k_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (attn_drop): Dropout(p=0, inplace=False)
          (proj): Linear(in_features=32, out_features=32, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
      )
      (freq_block_attention): Block_Attention(
        (norm1): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (B_Attention): Attention(
          (qkv): Linear(in_features=32, out_features=96, bias=True)
          (q_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (k_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (attn_drop): Dropout(p=0, inplace=False)
          (proj): Linear(in_features=32, out_features=32, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
      )
    )
  )
  (down1_2): Downsample(
    (body): Sequential(
      (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
      (1): Conv2d(32, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (3): PixelUnshuffle(downscale_factor=2)
    )
  )
  (patch_embed_encoder_level2): Patch_Embed_stage(
    (patch_embeds): MB_Deform_Embedding(
      (patch_conv): DWConv2d_BN(
        (offset_generator): Sequential(
          (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
          (1): Conv2d(64, 18, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (dcn): DeformConv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
        (pwconv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (act): Hardswish()
      )
      (norm): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (TFMamba_attention_encoder2): ModuleList(
    (0-3): 4 x TF_mamba_attention(
      (time_block_mamba): Block_mamba(
        (norm1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
        (B_mamba): MambaVisionMixer(
          (in_proj): Linear(in_features=64, out_features=64, bias=False)
          (x_proj): Linear(in_features=32, out_features=20, bias=False)
          (dt_proj): Linear(in_features=4, out_features=32, bias=True)
          (out_proj): Linear(in_features=64, out_features=64, bias=False)
          (conv1d_x): Conv1d(32, 32, kernel_size=(3,), stride=(1,), groups=32, bias=False)
          (conv1d_z): Conv1d(32, 32, kernel_size=(3,), stride=(1,), groups=32, bias=False)
        )
      )
      (freq_block_mamba): Block_mamba(
        (norm1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
        (B_mamba): MambaVisionMixer(
          (in_proj): Linear(in_features=64, out_features=64, bias=False)
          (x_proj): Linear(in_features=32, out_features=20, bias=False)
          (dt_proj): Linear(in_features=4, out_features=32, bias=True)
          (out_proj): Linear(in_features=64, out_features=64, bias=False)
          (conv1d_x): Conv1d(32, 32, kernel_size=(3,), stride=(1,), groups=32, bias=False)
          (conv1d_z): Conv1d(32, 32, kernel_size=(3,), stride=(1,), groups=32, bias=False)
        )
      )
      (time_block_attention): Block_Attention(
        (norm1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
        (B_Attention): Attention(
          (qkv): Linear(in_features=64, out_features=192, bias=True)
          (q_norm): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
          (k_norm): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
          (attn_drop): Dropout(p=0, inplace=False)
          (proj): Linear(in_features=64, out_features=64, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
      )
      (freq_block_attention): Block_Attention(
        (norm1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
        (B_Attention): Attention(
          (qkv): Linear(in_features=64, out_features=192, bias=True)
          (q_norm): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
          (k_norm): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
          (attn_drop): Dropout(p=0, inplace=False)
          (proj): Linear(in_features=64, out_features=64, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
      )
    )
  )
  (down2_3): Downsample(
    (body): Sequential(
      (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
      (1): Conv2d(64, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (2): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (3): PixelUnshuffle(downscale_factor=2)
    )
  )
  (patch_embed_middle): Patch_Embed_stage(
    (patch_embeds): MB_Deform_Embedding(
      (patch_conv): DWConv2d_BN(
        (offset_generator): Sequential(
          (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=96, bias=False)
          (1): Conv2d(96, 18, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (dcn): DeformConv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=96, bias=False)
        (pwconv): Conv2d(96, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (act): Hardswish()
      )
      (norm): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (TFMamba_attention_middle): ModuleList(
    (0-3): 4 x TF_mamba_attention(
      (time_block_mamba): Block_mamba(
        (norm1): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
        (B_mamba): MambaVisionMixer(
          (in_proj): Linear(in_features=96, out_features=96, bias=False)
          (x_proj): Linear(in_features=48, out_features=22, bias=False)
          (dt_proj): Linear(in_features=6, out_features=48, bias=True)
          (out_proj): Linear(in_features=96, out_features=96, bias=False)
          (conv1d_x): Conv1d(48, 48, kernel_size=(3,), stride=(1,), groups=48, bias=False)
          (conv1d_z): Conv1d(48, 48, kernel_size=(3,), stride=(1,), groups=48, bias=False)
        )
      )
      (freq_block_mamba): Block_mamba(
        (norm1): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
        (B_mamba): MambaVisionMixer(
          (in_proj): Linear(in_features=96, out_features=96, bias=False)
          (x_proj): Linear(in_features=48, out_features=22, bias=False)
          (dt_proj): Linear(in_features=6, out_features=48, bias=True)
          (out_proj): Linear(in_features=96, out_features=96, bias=False)
          (conv1d_x): Conv1d(48, 48, kernel_size=(3,), stride=(1,), groups=48, bias=False)
          (conv1d_z): Conv1d(48, 48, kernel_size=(3,), stride=(1,), groups=48, bias=False)
        )
      )
      (time_block_attention): Block_Attention(
        (norm1): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
        (B_Attention): Attention(
          (qkv): Linear(in_features=96, out_features=288, bias=True)
          (q_norm): LayerNorm((24,), eps=1e-05, elementwise_affine=True)
          (k_norm): LayerNorm((24,), eps=1e-05, elementwise_affine=True)
          (attn_drop): Dropout(p=0, inplace=False)
          (proj): Linear(in_features=96, out_features=96, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
      )
      (freq_block_attention): Block_Attention(
        (norm1): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
        (B_Attention): Attention(
          (qkv): Linear(in_features=96, out_features=288, bias=True)
          (q_norm): LayerNorm((24,), eps=1e-05, elementwise_affine=True)
          (k_norm): LayerNorm((24,), eps=1e-05, elementwise_affine=True)
          (attn_drop): Dropout(p=0, inplace=False)
          (proj): Linear(in_features=96, out_features=96, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
      )
    )
  )
  (up3_2): Upsample(
    (body): Sequential(
      (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=96, bias=False)
      (1): Conv2d(96, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (3): PixelShuffle(upscale_factor=2)
    )
  )
  (concat_level2): Sequential(
    (0): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (patch_embed_decoder_level2): Patch_Embed_stage(
    (patch_embeds): MB_Deform_Embedding(
      (patch_conv): DWConv2d_BN(
        (offset_generator): Sequential(
          (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
          (1): Conv2d(64, 18, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (dcn): DeformConv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
        (pwconv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (act): Hardswish()
      )
      (norm): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (TFMamba_attention_decoder2): ModuleList(
    (0-3): 4 x TF_mamba_attention(
      (time_block_mamba): Block_mamba(
        (norm1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
        (B_mamba): MambaVisionMixer(
          (in_proj): Linear(in_features=64, out_features=64, bias=False)
          (x_proj): Linear(in_features=32, out_features=20, bias=False)
          (dt_proj): Linear(in_features=4, out_features=32, bias=True)
          (out_proj): Linear(in_features=64, out_features=64, bias=False)
          (conv1d_x): Conv1d(32, 32, kernel_size=(3,), stride=(1,), groups=32, bias=False)
          (conv1d_z): Conv1d(32, 32, kernel_size=(3,), stride=(1,), groups=32, bias=False)
        )
      )
      (freq_block_mamba): Block_mamba(
        (norm1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
        (B_mamba): MambaVisionMixer(
          (in_proj): Linear(in_features=64, out_features=64, bias=False)
          (x_proj): Linear(in_features=32, out_features=20, bias=False)
          (dt_proj): Linear(in_features=4, out_features=32, bias=True)
          (out_proj): Linear(in_features=64, out_features=64, bias=False)
          (conv1d_x): Conv1d(32, 32, kernel_size=(3,), stride=(1,), groups=32, bias=False)
          (conv1d_z): Conv1d(32, 32, kernel_size=(3,), stride=(1,), groups=32, bias=False)
        )
      )
      (time_block_attention): Block_Attention(
        (norm1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
        (B_Attention): Attention(
          (qkv): Linear(in_features=64, out_features=192, bias=True)
          (q_norm): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
          (k_norm): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
          (attn_drop): Dropout(p=0, inplace=False)
          (proj): Linear(in_features=64, out_features=64, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
      )
      (freq_block_attention): Block_Attention(
        (norm1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
        (B_Attention): Attention(
          (qkv): Linear(in_features=64, out_features=192, bias=True)
          (q_norm): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
          (k_norm): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
          (attn_drop): Dropout(p=0, inplace=False)
          (proj): Linear(in_features=64, out_features=64, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
      )
    )
  )
  (up2_1): Upsample(
    (body): Sequential(
      (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
      (1): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (3): PixelShuffle(upscale_factor=2)
    )
  )
  (concat_level1): Sequential(
    (0): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (patch_embed_decoder_level1): Patch_Embed_stage(
    (patch_embeds): MB_Deform_Embedding(
      (patch_conv): DWConv2d_BN(
        (offset_generator): Sequential(
          (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
          (1): Conv2d(32, 18, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (dcn): DeformConv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
        (pwconv): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (act): Hardswish()
      )
      (norm): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (TFMamba_attention_decoder1): ModuleList(
    (0-3): 4 x TF_mamba_attention(
      (time_block_mamba): Block_mamba(
        (norm1): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (B_mamba): MambaVisionMixer(
          (in_proj): Linear(in_features=32, out_features=32, bias=False)
          (x_proj): Linear(in_features=16, out_features=18, bias=False)
          (dt_proj): Linear(in_features=2, out_features=16, bias=True)
          (out_proj): Linear(in_features=32, out_features=32, bias=False)
          (conv1d_x): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
          (conv1d_z): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
        )
      )
      (freq_block_mamba): Block_mamba(
        (norm1): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (B_mamba): MambaVisionMixer(
          (in_proj): Linear(in_features=32, out_features=32, bias=False)
          (x_proj): Linear(in_features=16, out_features=18, bias=False)
          (dt_proj): Linear(in_features=2, out_features=16, bias=True)
          (out_proj): Linear(in_features=32, out_features=32, bias=False)
          (conv1d_x): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
          (conv1d_z): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
        )
      )
      (time_block_attention): Block_Attention(
        (norm1): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (B_Attention): Attention(
          (qkv): Linear(in_features=32, out_features=96, bias=True)
          (q_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (k_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (attn_drop): Dropout(p=0, inplace=False)
          (proj): Linear(in_features=32, out_features=32, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
      )
      (freq_block_attention): Block_Attention(
        (norm1): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (B_Attention): Attention(
          (qkv): Linear(in_features=32, out_features=96, bias=True)
          (q_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (k_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (attn_drop): Dropout(p=0, inplace=False)
          (proj): Linear(in_features=32, out_features=32, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
      )
    )
  )
  (mag_patch_embed_refinement): Patch_Embed_stage(
    (patch_embeds): MB_Deform_Embedding(
      (patch_conv): DWConv2d_BN(
        (offset_generator): Sequential(
          (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
          (1): Conv2d(32, 18, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (dcn): DeformConv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
        (pwconv): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (act): Hardswish()
      )
      (norm): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (mag_refinement): ModuleList(
    (0-3): 4 x TF_mamba_attention(
      (time_block_mamba): Block_mamba(
        (norm1): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (B_mamba): MambaVisionMixer(
          (in_proj): Linear(in_features=32, out_features=32, bias=False)
          (x_proj): Linear(in_features=16, out_features=18, bias=False)
          (dt_proj): Linear(in_features=2, out_features=16, bias=True)
          (out_proj): Linear(in_features=32, out_features=32, bias=False)
          (conv1d_x): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
          (conv1d_z): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
        )
      )
      (freq_block_mamba): Block_mamba(
        (norm1): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (B_mamba): MambaVisionMixer(
          (in_proj): Linear(in_features=32, out_features=32, bias=False)
          (x_proj): Linear(in_features=16, out_features=18, bias=False)
          (dt_proj): Linear(in_features=2, out_features=16, bias=True)
          (out_proj): Linear(in_features=32, out_features=32, bias=False)
          (conv1d_x): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
          (conv1d_z): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
        )
      )
      (time_block_attention): Block_Attention(
        (norm1): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (B_Attention): Attention(
          (qkv): Linear(in_features=32, out_features=96, bias=True)
          (q_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (k_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (attn_drop): Dropout(p=0, inplace=False)
          (proj): Linear(in_features=32, out_features=32, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
      )
      (freq_block_attention): Block_Attention(
        (norm1): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (B_Attention): Attention(
          (qkv): Linear(in_features=32, out_features=96, bias=True)
          (q_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (k_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (attn_drop): Dropout(p=0, inplace=False)
          (proj): Linear(in_features=32, out_features=32, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
      )
    )
  )
  (mag_output): Sequential(
    (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  )
  (pha_patch_embed_refinement): Patch_Embed_stage(
    (patch_embeds): MB_Deform_Embedding(
      (patch_conv): DWConv2d_BN(
        (offset_generator): Sequential(
          (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
          (1): Conv2d(32, 18, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (dcn): DeformConv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
        (pwconv): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (act): Hardswish()
      )
      (norm): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (pha_refinement): ModuleList(
    (0-3): 4 x TF_mamba_attention(
      (time_block_mamba): Block_mamba(
        (norm1): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (B_mamba): MambaVisionMixer(
          (in_proj): Linear(in_features=32, out_features=32, bias=False)
          (x_proj): Linear(in_features=16, out_features=18, bias=False)
          (dt_proj): Linear(in_features=2, out_features=16, bias=True)
          (out_proj): Linear(in_features=32, out_features=32, bias=False)
          (conv1d_x): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
          (conv1d_z): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
        )
      )
      (freq_block_mamba): Block_mamba(
        (norm1): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (B_mamba): MambaVisionMixer(
          (in_proj): Linear(in_features=32, out_features=32, bias=False)
          (x_proj): Linear(in_features=16, out_features=18, bias=False)
          (dt_proj): Linear(in_features=2, out_features=16, bias=True)
          (out_proj): Linear(in_features=32, out_features=32, bias=False)
          (conv1d_x): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
          (conv1d_z): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
        )
      )
      (time_block_attention): Block_Attention(
        (norm1): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (B_Attention): Attention(
          (qkv): Linear(in_features=32, out_features=96, bias=True)
          (q_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (k_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (attn_drop): Dropout(p=0, inplace=False)
          (proj): Linear(in_features=32, out_features=32, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
      )
      (freq_block_attention): Block_Attention(
        (norm1): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (B_Attention): Attention(
          (qkv): Linear(in_features=32, out_features=96, bias=True)
          (q_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (k_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (attn_drop): Dropout(p=0, inplace=False)
          (proj): Linear(in_features=32, out_features=32, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
      )
    )
  )
  (pha_output): Sequential(
    (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  )
  (mask_decoder): MagDecoder(
    (dense_block): DenseBlock(
      (dense_block): ModuleList(
        (0): Sequential(
          (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (2): PReLU(num_parameters=32)
        )
        (1): Sequential(
          (0): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(2, 1), dilation=(2, 1))
          (1): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (2): PReLU(num_parameters=32)
        )
        (2): Sequential(
          (0): Conv2d(96, 32, kernel_size=(3, 3), stride=(1, 1), padding=(4, 1), dilation=(4, 1))
          (1): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (2): PReLU(num_parameters=32)
        )
        (3): Sequential(
          (0): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(8, 1), dilation=(8, 1))
          (1): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (2): PReLU(num_parameters=32)
        )
      )
    )
    (mask_conv): Sequential(
      (0): Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (1): PixelShuffle(upscale_factor=2)
      (2): Conv2d(32, 32, kernel_size=(1, 3), stride=(2, 1), padding=(0, 1), groups=32, bias=False)
      (3): Conv2d(32, 1, kernel_size=(1, 1), stride=(1, 1))
      (4): InstanceNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
      (5): PReLU(num_parameters=1)
      (6): Conv2d(1, 1, kernel_size=(1, 1), stride=(1, 1))
    )
    (lsigmoid): LearnableSigmoid2D()
  )
  (phase_decoder): PhaseDecoder(
    (dense_block): DenseBlock(
      (dense_block): ModuleList(
        (0): Sequential(
          (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (2): PReLU(num_parameters=32)
        )
        (1): Sequential(
          (0): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(2, 1), dilation=(2, 1))
          (1): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (2): PReLU(num_parameters=32)
        )
        (2): Sequential(
          (0): Conv2d(96, 32, kernel_size=(3, 3), stride=(1, 1), padding=(4, 1), dilation=(4, 1))
          (1): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (2): PReLU(num_parameters=32)
        )
        (3): Sequential(
          (0): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(8, 1), dilation=(8, 1))
          (1): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (2): PReLU(num_parameters=32)
        )
      )
    )
    (phase_conv): Sequential(
      (0): Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (1): PixelShuffle(upscale_factor=2)
      (2): Conv2d(32, 32, kernel_size=(1, 3), stride=(2, 1), padding=(0, 1), groups=32, bias=False)
      (3): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
      (4): PReLU(num_parameters=32)
    )
    (phase_conv_r): Conv2d(32, 1, kernel_size=(1, 1), stride=(1, 1))
    (phase_conv_i): Conv2d(32, 1, kernel_size=(1, 1), stride=(1, 1))
  )
)
Generator Parameters : 1505112
checkpoints directory : exp/Mambavision_emb_07
Epoch: 1
Steps : 0, Gen Loss: 3.475, Disc Loss: 0.589, Metric Loss: 0.542, Mag Loss: 0.837, Pha Loss: 5.105, Com Loss: 2.387, Time Loss: 1.252, Cons Loss: 2.175, s/b : 1.858
Steps : 200, Gen Loss: 2.098, Disc Loss: 0.001, Metric Loss: 0.970, Mag Loss: 0.460, Pha Loss: 3.971, Com Loss: 1.230, Time Loss: 0.534, Cons Loss: 0.456, s/b : 0.320
Steps : 400, Gen Loss: 2.242, Disc Loss: 0.040, Metric Loss: 0.747, Mag Loss: 0.568, Pha Loss: 4.017, Com Loss: 1.348, Time Loss: 0.575, Cons Loss: 0.516, s/b : 0.336










checkpoints directory : exp/MambaSEUNet_emb_32
Loading 'exp/MambaSEUNet_emb_32/g_00690000.pth'
Complete.
Loading 'exp/MambaSEUNet_emb_32/do_00690000.pth'
Complete.
**
(mambavision) g515528@515528:~/PycharmProjects/Mamba-SEUNet-main_3TSFM$ python train.py
Number of GPUs available: 1
GPU 0: NVIDIA GeForce RTX 4090
Batch size per GPU: 2
MambaSEUNet(
  (dense_encoder): DenseEncoder(
    (dense_conv_1): Sequential(
      (0): Conv2d(2, 32, kernel_size=(1, 1), stride=(1, 1))
      (1): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
      (2): PReLU(num_parameters=32)
    )
    (dense_block): DenseBlock(
      (dense_block): ModuleList(
        (0): Sequential(
          (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (2): PReLU(num_parameters=32)
        )
        (1): Sequential(
          (0): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(2, 1), dilation=(2, 1))
          (1): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (2): PReLU(num_parameters=32)
        )
        (2): Sequential(
          (0): Conv2d(96, 32, kernel_size=(3, 3), stride=(1, 1), padding=(4, 1), dilation=(4, 1))
          (1): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (2): PReLU(num_parameters=32)
        )
        (3): Sequential(
          (0): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(8, 1), dilation=(8, 1))
          (1): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (2): PReLU(num_parameters=32)
        )
      )
    )
    (dense_conv_2): Sequential(
      (0): Conv2d(32, 32, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
      (1): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
      (2): PReLU(num_parameters=32)
    )
  )
  (patch_embed_encoder_level1): Patch_Embed_stage(
    (patch_embeds): MB_Deform_Embedding(
      (patch_conv): DWConv2d_BN(
        (offset_generator): Sequential(
          (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
          (1): Conv2d(32, 18, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (dcn): DeformConv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
        (pwconv): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (act): Hardswish()
      )
      (norm): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (TFMamba_attention_encoder1): ModuleList(
    (0-3): 4 x TF_mamba_attention(
      (time_block_mamba): Block_mamba(
        (norm1): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (B_mamba): MambaVisionMixer(
          (in_proj): Linear(in_features=32, out_features=32, bias=False)
          (x_proj): Linear(in_features=16, out_features=18, bias=False)
          (dt_proj): Linear(in_features=2, out_features=16, bias=True)
          (out_proj): Linear(in_features=32, out_features=32, bias=False)
          (conv1d_x): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
          (conv1d_z): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
        )
        (drop_path): DropPath(drop_prob=0.001)
        (norm2): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=32, out_features=128, bias=True)
          (act): GELU(approximate='none')
          (drop1): Dropout(p=0, inplace=False)
          (norm): Identity()
          (fc2): Linear(in_features=128, out_features=32, bias=True)
          (drop2): Dropout(p=0, inplace=False)
        )
      )
      (freq_block_mamba): Block_mamba(
        (norm1): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (B_mamba): MambaVisionMixer(
          (in_proj): Linear(in_features=32, out_features=32, bias=False)
          (x_proj): Linear(in_features=16, out_features=18, bias=False)
          (dt_proj): Linear(in_features=2, out_features=16, bias=True)
          (out_proj): Linear(in_features=32, out_features=32, bias=False)
          (conv1d_x): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
          (conv1d_z): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
        )
        (drop_path): DropPath(drop_prob=0.001)
        (norm2): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=32, out_features=128, bias=True)
          (act): GELU(approximate='none')
          (drop1): Dropout(p=0, inplace=False)
          (norm): Identity()
          (fc2): Linear(in_features=128, out_features=32, bias=True)
          (drop2): Dropout(p=0, inplace=False)
        )
      )
      (time_block_attention): Block_Attention(
        (norm1): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (B_Attention): Attention(
          (qkv): Linear(in_features=32, out_features=96, bias=True)
          (q_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (k_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (attn_drop): Dropout(p=0, inplace=False)
          (proj): Linear(in_features=32, out_features=32, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (drop_path): DropPath(drop_prob=0.001)
        (norm2): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=32, out_features=128, bias=True)
          (act): GELU(approximate='none')
          (drop1): Dropout(p=0.0, inplace=False)
          (norm): Identity()
          (fc2): Linear(in_features=128, out_features=32, bias=True)
          (drop2): Dropout(p=0.0, inplace=False)
        )
      )
      (freq_block_attention): Block_Attention(
        (norm1): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (B_Attention): Attention(
          (qkv): Linear(in_features=32, out_features=96, bias=True)
          (q_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (k_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (attn_drop): Dropout(p=0, inplace=False)
          (proj): Linear(in_features=32, out_features=32, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (drop_path): DropPath(drop_prob=0.001)
        (norm2): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=32, out_features=128, bias=True)
          (act): GELU(approximate='none')
          (drop1): Dropout(p=0.0, inplace=False)
          (norm): Identity()
          (fc2): Linear(in_features=128, out_features=32, bias=True)
          (drop2): Dropout(p=0.0, inplace=False)
        )
      )
    )
  )
  (down1_2): Downsample(
    (body): Sequential(
      (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
      (1): Conv2d(32, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (3): PixelUnshuffle(downscale_factor=2)
    )
  )
  (patch_embed_encoder_level2): Patch_Embed_stage(
    (patch_embeds): MB_Deform_Embedding(
      (patch_conv): DWConv2d_BN(
        (offset_generator): Sequential(
          (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
          (1): Conv2d(64, 18, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (dcn): DeformConv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
        (pwconv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (act): Hardswish()
      )
      (norm): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (TFMamba_attention_encoder2): ModuleList(
    (0-3): 4 x TF_mamba_attention(
      (time_block_mamba): Block_mamba(
        (norm1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
        (B_mamba): MambaVisionMixer(
          (in_proj): Linear(in_features=64, out_features=64, bias=False)
          (x_proj): Linear(in_features=32, out_features=20, bias=False)
          (dt_proj): Linear(in_features=4, out_features=32, bias=True)
          (out_proj): Linear(in_features=64, out_features=64, bias=False)
          (conv1d_x): Conv1d(32, 32, kernel_size=(3,), stride=(1,), groups=32, bias=False)
          (conv1d_z): Conv1d(32, 32, kernel_size=(3,), stride=(1,), groups=32, bias=False)
        )
        (drop_path): DropPath(drop_prob=0.001)
        (norm2): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=64, out_features=256, bias=True)
          (act): GELU(approximate='none')
          (drop1): Dropout(p=0, inplace=False)
          (norm): Identity()
          (fc2): Linear(in_features=256, out_features=64, bias=True)
          (drop2): Dropout(p=0, inplace=False)
        )
      )
      (freq_block_mamba): Block_mamba(
        (norm1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
        (B_mamba): MambaVisionMixer(
          (in_proj): Linear(in_features=64, out_features=64, bias=False)
          (x_proj): Linear(in_features=32, out_features=20, bias=False)
          (dt_proj): Linear(in_features=4, out_features=32, bias=True)
          (out_proj): Linear(in_features=64, out_features=64, bias=False)
          (conv1d_x): Conv1d(32, 32, kernel_size=(3,), stride=(1,), groups=32, bias=False)
          (conv1d_z): Conv1d(32, 32, kernel_size=(3,), stride=(1,), groups=32, bias=False)
        )
        (drop_path): DropPath(drop_prob=0.001)
        (norm2): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=64, out_features=256, bias=True)
          (act): GELU(approximate='none')
          (drop1): Dropout(p=0, inplace=False)
          (norm): Identity()
          (fc2): Linear(in_features=256, out_features=64, bias=True)
          (drop2): Dropout(p=0, inplace=False)
        )
      )
      (time_block_attention): Block_Attention(
        (norm1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
        (B_Attention): Attention(
          (qkv): Linear(in_features=64, out_features=192, bias=True)
          (q_norm): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
          (k_norm): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
          (attn_drop): Dropout(p=0, inplace=False)
          (proj): Linear(in_features=64, out_features=64, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (drop_path): DropPath(drop_prob=0.001)
        (norm2): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=64, out_features=256, bias=True)
          (act): GELU(approximate='none')
          (drop1): Dropout(p=0.0, inplace=False)
          (norm): Identity()
          (fc2): Linear(in_features=256, out_features=64, bias=True)
          (drop2): Dropout(p=0.0, inplace=False)
        )
      )
      (freq_block_attention): Block_Attention(
        (norm1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
        (B_Attention): Attention(
          (qkv): Linear(in_features=64, out_features=192, bias=True)
          (q_norm): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
          (k_norm): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
          (attn_drop): Dropout(p=0, inplace=False)
          (proj): Linear(in_features=64, out_features=64, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (drop_path): DropPath(drop_prob=0.001)
        (norm2): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=64, out_features=256, bias=True)
          (act): GELU(approximate='none')
          (drop1): Dropout(p=0.0, inplace=False)
          (norm): Identity()
          (fc2): Linear(in_features=256, out_features=64, bias=True)
          (drop2): Dropout(p=0.0, inplace=False)
        )
      )
    )
  )
  (down2_3): Downsample(
    (body): Sequential(
      (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
      (1): Conv2d(64, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (2): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (3): PixelUnshuffle(downscale_factor=2)
    )
  )
  (patch_embed_middle): Patch_Embed_stage(
    (patch_embeds): MB_Deform_Embedding(
      (patch_conv): DWConv2d_BN(
        (offset_generator): Sequential(
          (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=96, bias=False)
          (1): Conv2d(96, 18, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (dcn): DeformConv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=96, bias=False)
        (pwconv): Conv2d(96, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (act): Hardswish()
      )
      (norm): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (TFMamba_attention_middle): ModuleList(
    (0-3): 4 x TF_mamba_attention(
      (time_block_mamba): Block_mamba(
        (norm1): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
        (B_mamba): MambaVisionMixer(
          (in_proj): Linear(in_features=96, out_features=96, bias=False)
          (x_proj): Linear(in_features=48, out_features=22, bias=False)
          (dt_proj): Linear(in_features=6, out_features=48, bias=True)
          (out_proj): Linear(in_features=96, out_features=96, bias=False)
          (conv1d_x): Conv1d(48, 48, kernel_size=(3,), stride=(1,), groups=48, bias=False)
          (conv1d_z): Conv1d(48, 48, kernel_size=(3,), stride=(1,), groups=48, bias=False)
        )
        (drop_path): DropPath(drop_prob=0.001)
        (norm2): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=96, out_features=384, bias=True)
          (act): GELU(approximate='none')
          (drop1): Dropout(p=0, inplace=False)
          (norm): Identity()
          (fc2): Linear(in_features=384, out_features=96, bias=True)
          (drop2): Dropout(p=0, inplace=False)
        )
      )
      (freq_block_mamba): Block_mamba(
        (norm1): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
        (B_mamba): MambaVisionMixer(
          (in_proj): Linear(in_features=96, out_features=96, bias=False)
          (x_proj): Linear(in_features=48, out_features=22, bias=False)
          (dt_proj): Linear(in_features=6, out_features=48, bias=True)
          (out_proj): Linear(in_features=96, out_features=96, bias=False)
          (conv1d_x): Conv1d(48, 48, kernel_size=(3,), stride=(1,), groups=48, bias=False)
          (conv1d_z): Conv1d(48, 48, kernel_size=(3,), stride=(1,), groups=48, bias=False)
        )
        (drop_path): DropPath(drop_prob=0.001)
        (norm2): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=96, out_features=384, bias=True)
          (act): GELU(approximate='none')
          (drop1): Dropout(p=0, inplace=False)
          (norm): Identity()
          (fc2): Linear(in_features=384, out_features=96, bias=True)
          (drop2): Dropout(p=0, inplace=False)
        )
      )
      (time_block_attention): Block_Attention(
        (norm1): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
        (B_Attention): Attention(
          (qkv): Linear(in_features=96, out_features=288, bias=True)
          (q_norm): LayerNorm((24,), eps=1e-05, elementwise_affine=True)
          (k_norm): LayerNorm((24,), eps=1e-05, elementwise_affine=True)
          (attn_drop): Dropout(p=0, inplace=False)
          (proj): Linear(in_features=96, out_features=96, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (drop_path): DropPath(drop_prob=0.001)
        (norm2): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=96, out_features=384, bias=True)
          (act): GELU(approximate='none')
          (drop1): Dropout(p=0.0, inplace=False)
          (norm): Identity()
          (fc2): Linear(in_features=384, out_features=96, bias=True)
          (drop2): Dropout(p=0.0, inplace=False)
        )
      )
      (freq_block_attention): Block_Attention(
        (norm1): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
        (B_Attention): Attention(
          (qkv): Linear(in_features=96, out_features=288, bias=True)
          (q_norm): LayerNorm((24,), eps=1e-05, elementwise_affine=True)
          (k_norm): LayerNorm((24,), eps=1e-05, elementwise_affine=True)
          (attn_drop): Dropout(p=0, inplace=False)
          (proj): Linear(in_features=96, out_features=96, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (drop_path): DropPath(drop_prob=0.001)
        (norm2): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=96, out_features=384, bias=True)
          (act): GELU(approximate='none')
          (drop1): Dropout(p=0.0, inplace=False)
          (norm): Identity()
          (fc2): Linear(in_features=384, out_features=96, bias=True)
          (drop2): Dropout(p=0.0, inplace=False)
        )
      )
    )
  )
  (up3_2): Upsample(
    (body): Sequential(
      (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=96, bias=False)
      (1): Conv2d(96, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (3): PixelShuffle(upscale_factor=2)
    )
  )
  (concat_level2): Sequential(
    (0): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (patch_embed_decoder_level2): Patch_Embed_stage(
    (patch_embeds): MB_Deform_Embedding(
      (patch_conv): DWConv2d_BN(
        (offset_generator): Sequential(
          (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
          (1): Conv2d(64, 18, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (dcn): DeformConv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
        (pwconv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (act): Hardswish()
      )
      (norm): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (TFMamba_attention_decoder2): ModuleList(
    (0-3): 4 x TF_mamba_attention(
      (time_block_mamba): Block_mamba(
        (norm1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
        (B_mamba): MambaVisionMixer(
          (in_proj): Linear(in_features=64, out_features=64, bias=False)
          (x_proj): Linear(in_features=32, out_features=20, bias=False)
          (dt_proj): Linear(in_features=4, out_features=32, bias=True)
          (out_proj): Linear(in_features=64, out_features=64, bias=False)
          (conv1d_x): Conv1d(32, 32, kernel_size=(3,), stride=(1,), groups=32, bias=False)
          (conv1d_z): Conv1d(32, 32, kernel_size=(3,), stride=(1,), groups=32, bias=False)
        )
        (drop_path): DropPath(drop_prob=0.001)
        (norm2): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=64, out_features=256, bias=True)
          (act): GELU(approximate='none')
          (drop1): Dropout(p=0, inplace=False)
          (norm): Identity()
          (fc2): Linear(in_features=256, out_features=64, bias=True)
          (drop2): Dropout(p=0, inplace=False)
        )
      )
      (freq_block_mamba): Block_mamba(
        (norm1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
        (B_mamba): MambaVisionMixer(
          (in_proj): Linear(in_features=64, out_features=64, bias=False)
          (x_proj): Linear(in_features=32, out_features=20, bias=False)
          (dt_proj): Linear(in_features=4, out_features=32, bias=True)
          (out_proj): Linear(in_features=64, out_features=64, bias=False)
          (conv1d_x): Conv1d(32, 32, kernel_size=(3,), stride=(1,), groups=32, bias=False)
          (conv1d_z): Conv1d(32, 32, kernel_size=(3,), stride=(1,), groups=32, bias=False)
        )
        (drop_path): DropPath(drop_prob=0.001)
        (norm2): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=64, out_features=256, bias=True)
          (act): GELU(approximate='none')
          (drop1): Dropout(p=0, inplace=False)
          (norm): Identity()
          (fc2): Linear(in_features=256, out_features=64, bias=True)
          (drop2): Dropout(p=0, inplace=False)
        )
      )
      (time_block_attention): Block_Attention(
        (norm1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
        (B_Attention): Attention(
          (qkv): Linear(in_features=64, out_features=192, bias=True)
          (q_norm): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
          (k_norm): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
          (attn_drop): Dropout(p=0, inplace=False)
          (proj): Linear(in_features=64, out_features=64, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (drop_path): DropPath(drop_prob=0.001)
        (norm2): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=64, out_features=256, bias=True)
          (act): GELU(approximate='none')
          (drop1): Dropout(p=0.0, inplace=False)
          (norm): Identity()
          (fc2): Linear(in_features=256, out_features=64, bias=True)
          (drop2): Dropout(p=0.0, inplace=False)
        )
      )
      (freq_block_attention): Block_Attention(
        (norm1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
        (B_Attention): Attention(
          (qkv): Linear(in_features=64, out_features=192, bias=True)
          (q_norm): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
          (k_norm): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
          (attn_drop): Dropout(p=0, inplace=False)
          (proj): Linear(in_features=64, out_features=64, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (drop_path): DropPath(drop_prob=0.001)
        (norm2): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=64, out_features=256, bias=True)
          (act): GELU(approximate='none')
          (drop1): Dropout(p=0.0, inplace=False)
          (norm): Identity()
          (fc2): Linear(in_features=256, out_features=64, bias=True)
          (drop2): Dropout(p=0.0, inplace=False)
        )
      )
    )
  )
  (up2_1): Upsample(
    (body): Sequential(
      (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
      (1): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (3): PixelShuffle(upscale_factor=2)
    )
  )
  (concat_level1): Sequential(
    (0): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (patch_embed_decoder_level1): Patch_Embed_stage(
    (patch_embeds): MB_Deform_Embedding(
      (patch_conv): DWConv2d_BN(
        (offset_generator): Sequential(
          (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
          (1): Conv2d(32, 18, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (dcn): DeformConv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
        (pwconv): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (act): Hardswish()
      )
      (norm): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (TFMamba_attention_decoder1): ModuleList(
    (0-3): 4 x TF_mamba_attention(
      (time_block_mamba): Block_mamba(
        (norm1): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (B_mamba): MambaVisionMixer(
          (in_proj): Linear(in_features=32, out_features=32, bias=False)
          (x_proj): Linear(in_features=16, out_features=18, bias=False)
          (dt_proj): Linear(in_features=2, out_features=16, bias=True)
          (out_proj): Linear(in_features=32, out_features=32, bias=False)
          (conv1d_x): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
          (conv1d_z): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
        )
        (drop_path): DropPath(drop_prob=0.001)
        (norm2): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=32, out_features=128, bias=True)
          (act): GELU(approximate='none')
          (drop1): Dropout(p=0, inplace=False)
          (norm): Identity()
          (fc2): Linear(in_features=128, out_features=32, bias=True)
          (drop2): Dropout(p=0, inplace=False)
        )
      )
      (freq_block_mamba): Block_mamba(
        (norm1): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (B_mamba): MambaVisionMixer(
          (in_proj): Linear(in_features=32, out_features=32, bias=False)
          (x_proj): Linear(in_features=16, out_features=18, bias=False)
          (dt_proj): Linear(in_features=2, out_features=16, bias=True)
          (out_proj): Linear(in_features=32, out_features=32, bias=False)
          (conv1d_x): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
          (conv1d_z): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
        )
        (drop_path): DropPath(drop_prob=0.001)
        (norm2): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=32, out_features=128, bias=True)
          (act): GELU(approximate='none')
          (drop1): Dropout(p=0, inplace=False)
          (norm): Identity()
          (fc2): Linear(in_features=128, out_features=32, bias=True)
          (drop2): Dropout(p=0, inplace=False)
        )
      )
      (time_block_attention): Block_Attention(
        (norm1): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (B_Attention): Attention(
          (qkv): Linear(in_features=32, out_features=96, bias=True)
          (q_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (k_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (attn_drop): Dropout(p=0, inplace=False)
          (proj): Linear(in_features=32, out_features=32, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (drop_path): DropPath(drop_prob=0.001)
        (norm2): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=32, out_features=128, bias=True)
          (act): GELU(approximate='none')
          (drop1): Dropout(p=0.0, inplace=False)
          (norm): Identity()
          (fc2): Linear(in_features=128, out_features=32, bias=True)
          (drop2): Dropout(p=0.0, inplace=False)
        )
      )
      (freq_block_attention): Block_Attention(
        (norm1): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (B_Attention): Attention(
          (qkv): Linear(in_features=32, out_features=96, bias=True)
          (q_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (k_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (attn_drop): Dropout(p=0, inplace=False)
          (proj): Linear(in_features=32, out_features=32, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (drop_path): DropPath(drop_prob=0.001)
        (norm2): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=32, out_features=128, bias=True)
          (act): GELU(approximate='none')
          (drop1): Dropout(p=0.0, inplace=False)
          (norm): Identity()
          (fc2): Linear(in_features=128, out_features=32, bias=True)
          (drop2): Dropout(p=0.0, inplace=False)
        )
      )
    )
  )
  (mag_patch_embed_refinement): Patch_Embed_stage(
    (patch_embeds): MB_Deform_Embedding(
      (patch_conv): DWConv2d_BN(
        (offset_generator): Sequential(
          (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
          (1): Conv2d(32, 18, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (dcn): DeformConv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
        (pwconv): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (act): Hardswish()
      )
      (norm): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (mag_refinement): ModuleList(
    (0-3): 4 x TF_mamba_attention(
      (time_block_mamba): Block_mamba(
        (norm1): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (B_mamba): MambaVisionMixer(
          (in_proj): Linear(in_features=32, out_features=32, bias=False)
          (x_proj): Linear(in_features=16, out_features=18, bias=False)
          (dt_proj): Linear(in_features=2, out_features=16, bias=True)
          (out_proj): Linear(in_features=32, out_features=32, bias=False)
          (conv1d_x): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
          (conv1d_z): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
        )
        (drop_path): DropPath(drop_prob=0.001)
        (norm2): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=32, out_features=128, bias=True)
          (act): GELU(approximate='none')
          (drop1): Dropout(p=0, inplace=False)
          (norm): Identity()
          (fc2): Linear(in_features=128, out_features=32, bias=True)
          (drop2): Dropout(p=0, inplace=False)
        )
      )
      (freq_block_mamba): Block_mamba(
        (norm1): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (B_mamba): MambaVisionMixer(
          (in_proj): Linear(in_features=32, out_features=32, bias=False)
          (x_proj): Linear(in_features=16, out_features=18, bias=False)
          (dt_proj): Linear(in_features=2, out_features=16, bias=True)
          (out_proj): Linear(in_features=32, out_features=32, bias=False)
          (conv1d_x): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
          (conv1d_z): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
        )
        (drop_path): DropPath(drop_prob=0.001)
        (norm2): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=32, out_features=128, bias=True)
          (act): GELU(approximate='none')
          (drop1): Dropout(p=0, inplace=False)
          (norm): Identity()
          (fc2): Linear(in_features=128, out_features=32, bias=True)
          (drop2): Dropout(p=0, inplace=False)
        )
      )
      (time_block_attention): Block_Attention(
        (norm1): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (B_Attention): Attention(
          (qkv): Linear(in_features=32, out_features=96, bias=True)
          (q_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (k_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (attn_drop): Dropout(p=0, inplace=False)
          (proj): Linear(in_features=32, out_features=32, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (drop_path): DropPath(drop_prob=0.001)
        (norm2): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=32, out_features=128, bias=True)
          (act): GELU(approximate='none')
          (drop1): Dropout(p=0.0, inplace=False)
          (norm): Identity()
          (fc2): Linear(in_features=128, out_features=32, bias=True)
          (drop2): Dropout(p=0.0, inplace=False)
        )
      )
      (freq_block_attention): Block_Attention(
        (norm1): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (B_Attention): Attention(
          (qkv): Linear(in_features=32, out_features=96, bias=True)
          (q_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (k_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (attn_drop): Dropout(p=0, inplace=False)
          (proj): Linear(in_features=32, out_features=32, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (drop_path): DropPath(drop_prob=0.001)
        (norm2): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=32, out_features=128, bias=True)
          (act): GELU(approximate='none')
          (drop1): Dropout(p=0.0, inplace=False)
          (norm): Identity()
          (fc2): Linear(in_features=128, out_features=32, bias=True)
          (drop2): Dropout(p=0.0, inplace=False)
        )
      )
    )
  )
  (mag_output): Sequential(
    (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  )
  (pha_patch_embed_refinement): Patch_Embed_stage(
    (patch_embeds): MB_Deform_Embedding(
      (patch_conv): DWConv2d_BN(
        (offset_generator): Sequential(
          (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
          (1): Conv2d(32, 18, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (dcn): DeformConv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
        (pwconv): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (act): Hardswish()
      )
      (norm): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (pha_refinement): ModuleList(
    (0-3): 4 x TF_mamba_attention(
      (time_block_mamba): Block_mamba(
        (norm1): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (B_mamba): MambaVisionMixer(
          (in_proj): Linear(in_features=32, out_features=32, bias=False)
          (x_proj): Linear(in_features=16, out_features=18, bias=False)
          (dt_proj): Linear(in_features=2, out_features=16, bias=True)
          (out_proj): Linear(in_features=32, out_features=32, bias=False)
          (conv1d_x): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
          (conv1d_z): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
        )
        (drop_path): DropPath(drop_prob=0.001)
        (norm2): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=32, out_features=128, bias=True)
          (act): GELU(approximate='none')
          (drop1): Dropout(p=0, inplace=False)
          (norm): Identity()
          (fc2): Linear(in_features=128, out_features=32, bias=True)
          (drop2): Dropout(p=0, inplace=False)
        )
      )
      (freq_block_mamba): Block_mamba(
        (norm1): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (B_mamba): MambaVisionMixer(
          (in_proj): Linear(in_features=32, out_features=32, bias=False)
          (x_proj): Linear(in_features=16, out_features=18, bias=False)
          (dt_proj): Linear(in_features=2, out_features=16, bias=True)
          (out_proj): Linear(in_features=32, out_features=32, bias=False)
          (conv1d_x): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
          (conv1d_z): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
        )
        (drop_path): DropPath(drop_prob=0.001)
        (norm2): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=32, out_features=128, bias=True)
          (act): GELU(approximate='none')
          (drop1): Dropout(p=0, inplace=False)
          (norm): Identity()
          (fc2): Linear(in_features=128, out_features=32, bias=True)
          (drop2): Dropout(p=0, inplace=False)
        )
      )
      (time_block_attention): Block_Attention(
        (norm1): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (B_Attention): Attention(
          (qkv): Linear(in_features=32, out_features=96, bias=True)
          (q_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (k_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (attn_drop): Dropout(p=0, inplace=False)
          (proj): Linear(in_features=32, out_features=32, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (drop_path): DropPath(drop_prob=0.001)
        (norm2): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=32, out_features=128, bias=True)
          (act): GELU(approximate='none')
          (drop1): Dropout(p=0.0, inplace=False)
          (norm): Identity()
          (fc2): Linear(in_features=128, out_features=32, bias=True)
          (drop2): Dropout(p=0.0, inplace=False)
        )
      )
      (freq_block_attention): Block_Attention(
        (norm1): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (B_Attention): Attention(
          (qkv): Linear(in_features=32, out_features=96, bias=True)
          (q_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (k_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (attn_drop): Dropout(p=0, inplace=False)
          (proj): Linear(in_features=32, out_features=32, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (drop_path): DropPath(drop_prob=0.001)
        (norm2): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=32, out_features=128, bias=True)
          (act): GELU(approximate='none')
          (drop1): Dropout(p=0.0, inplace=False)
          (norm): Identity()
          (fc2): Linear(in_features=128, out_features=32, bias=True)
          (drop2): Dropout(p=0.0, inplace=False)
        )
      )
    )
  )
  (pha_output): Sequential(
    (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  )
  (mask_decoder): MagDecoder(
    (dense_block): DenseBlock(
      (dense_block): ModuleList(
        (0): Sequential(
          (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (2): PReLU(num_parameters=32)
        )
        (1): Sequential(
          (0): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(2, 1), dilation=(2, 1))
          (1): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (2): PReLU(num_parameters=32)
        )
        (2): Sequential(
          (0): Conv2d(96, 32, kernel_size=(3, 3), stride=(1, 1), padding=(4, 1), dilation=(4, 1))
          (1): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (2): PReLU(num_parameters=32)
        )
        (3): Sequential(
          (0): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(8, 1), dilation=(8, 1))
          (1): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (2): PReLU(num_parameters=32)
        )
      )
    )
    (mask_conv): Sequential(
      (0): Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (1): PixelShuffle(upscale_factor=2)
      (2): Conv2d(32, 32, kernel_size=(1, 3), stride=(2, 1), padding=(0, 1), groups=32, bias=False)
      (3): Conv2d(32, 1, kernel_size=(1, 1), stride=(1, 1))
      (4): InstanceNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
      (5): PReLU(num_parameters=1)
      (6): Conv2d(1, 1, kernel_size=(1, 1), stride=(1, 1))
    )
    (lsigmoid): LearnableSigmoid2D()
  )
  (phase_decoder): PhaseDecoder(
    (dense_block): DenseBlock(
      (dense_block): ModuleList(
        (0): Sequential(
          (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (2): PReLU(num_parameters=32)
        )
        (1): Sequential(
          (0): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(2, 1), dilation=(2, 1))
          (1): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (2): PReLU(num_parameters=32)
        )
        (2): Sequential(
          (0): Conv2d(96, 32, kernel_size=(3, 3), stride=(1, 1), padding=(4, 1), dilation=(4, 1))
          (1): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (2): PReLU(num_parameters=32)
        )
        (3): Sequential(
          (0): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(8, 1), dilation=(8, 1))
          (1): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (2): PReLU(num_parameters=32)
        )
      )
    )
    (phase_conv): Sequential(
      (0): Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (1): PixelShuffle(upscale_factor=2)
      (2): Conv2d(32, 32, kernel_size=(1, 3), stride=(2, 1), padding=(0, 1), groups=32, bias=False)
      (3): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
      (4): PReLU(num_parameters=32)
    )
    (phase_conv_r): Conv2d(32, 1, kernel_size=(1, 1), stride=(1, 1))
    (phase_conv_i): Conv2d(32, 1, kernel_size=(1, 1), stride=(1, 1))
  )
)
Generator Parameters : 4308312


(mambavision) g515528@515528:~/PycharmProjects/Mamba-SEUNet-main_3TSFM$ python train.py
Number of GPUs available: 1
GPU 0: NVIDIA GeForce RTX 4090
Batch size per GPU: 2
MambaSEUNet(
  (dense_encoder): DenseEncoder(
    (dense_conv_1): Sequential(
      (0): Conv2d(2, 32, kernel_size=(1, 1), stride=(1, 1))
      (1): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
      (2): PReLU(num_parameters=32)
    )
    (dense_block): DenseBlock(
      (dense_block): ModuleList(
        (0): Sequential(
          (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (2): PReLU(num_parameters=32)
        )
        (1): Sequential(
          (0): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(2, 1), dilation=(2, 1))
          (1): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (2): PReLU(num_parameters=32)
        )
        (2): Sequential(
          (0): Conv2d(96, 32, kernel_size=(3, 3), stride=(1, 1), padding=(4, 1), dilation=(4, 1))
          (1): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (2): PReLU(num_parameters=32)
        )
        (3): Sequential(
          (0): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(8, 1), dilation=(8, 1))
          (1): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (2): PReLU(num_parameters=32)
        )
      )
    )
    (dense_conv_2): Sequential(
      (0): Conv2d(32, 32, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
      (1): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
      (2): PReLU(num_parameters=32)
    )
  )
  (patch_embed_encoder_level1): Patch_Embed_stage(
    (patch_embeds): MB_Deform_Embedding(
      (patch_conv): DWConv2d_BN(
        (offset_generator): Sequential(
          (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
          (1): Conv2d(32, 18, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (dcn): DeformConv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
        (pwconv): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (act): Hardswish()
      )
      (norm): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (TFMamba_attention_encoder1): ModuleList(
    (0-3): 4 x TF_mamba_attention(
      (time_block_mamba): Block_mamba(
        (norm1): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (B_mamba): MambaVisionMixer(
          (in_proj): Linear(in_features=32, out_features=32, bias=False)
          (x_proj): Linear(in_features=16, out_features=18, bias=False)
          (dt_proj): Linear(in_features=2, out_features=16, bias=True)
          (out_proj): Linear(in_features=32, out_features=32, bias=False)
          (conv1d_x): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
          (conv1d_z): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
        )
        (drop_path): Identity()
        (norm2): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=32, out_features=128, bias=True)
          (act): GELU(approximate='none')
          (drop1): Dropout(p=0, inplace=False)
          (norm): Identity()
          (fc2): Linear(in_features=128, out_features=32, bias=True)
          (drop2): Dropout(p=0, inplace=False)
        )
      )
      (freq_block_mamba): Block_mamba(
        (norm1): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (B_mamba): MambaVisionMixer(
          (in_proj): Linear(in_features=32, out_features=32, bias=False)
          (x_proj): Linear(in_features=16, out_features=18, bias=False)
          (dt_proj): Linear(in_features=2, out_features=16, bias=True)
          (out_proj): Linear(in_features=32, out_features=32, bias=False)
          (conv1d_x): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
          (conv1d_z): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
        )
        (drop_path): Identity()
        (norm2): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=32, out_features=128, bias=True)
          (act): GELU(approximate='none')
          (drop1): Dropout(p=0, inplace=False)
          (norm): Identity()
          (fc2): Linear(in_features=128, out_features=32, bias=True)
          (drop2): Dropout(p=0, inplace=False)
        )
      )
      (time_block_attention): Block_Attention(
        (norm1): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (B_Attention): Attention(
          (qkv): Linear(in_features=32, out_features=96, bias=True)
          (q_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (k_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (attn_drop): Dropout(p=0, inplace=False)
          (proj): Linear(in_features=32, out_features=32, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (drop_path): Identity()
        (norm2): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=32, out_features=128, bias=True)
          (act): GELU(approximate='none')
          (drop1): Dropout(p=0.0, inplace=False)
          (norm): Identity()
          (fc2): Linear(in_features=128, out_features=32, bias=True)
          (drop2): Dropout(p=0.0, inplace=False)
        )
      )
      (freq_block_attention): Block_Attention(
        (norm1): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (B_Attention): Attention(
          (qkv): Linear(in_features=32, out_features=96, bias=True)
          (q_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (k_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (attn_drop): Dropout(p=0, inplace=False)
          (proj): Linear(in_features=32, out_features=32, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (drop_path): Identity()
        (norm2): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=32, out_features=128, bias=True)
          (act): GELU(approximate='none')
          (drop1): Dropout(p=0.0, inplace=False)
          (norm): Identity()
          (fc2): Linear(in_features=128, out_features=32, bias=True)
          (drop2): Dropout(p=0.0, inplace=False)
        )
      )
    )
  )
  (down1_2): Downsample(
    (body): Sequential(
      (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
      (1): Conv2d(32, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (3): PixelUnshuffle(downscale_factor=2)
    )
  )
  (patch_embed_encoder_level2): Patch_Embed_stage(
    (patch_embeds): MB_Deform_Embedding(
      (patch_conv): DWConv2d_BN(
        (offset_generator): Sequential(
          (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
          (1): Conv2d(64, 18, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (dcn): DeformConv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
        (pwconv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (act): Hardswish()
      )
      (norm): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (TFMamba_attention_encoder2): ModuleList(
    (0-3): 4 x TF_mamba_attention(
      (time_block_mamba): Block_mamba(
        (norm1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
        (B_mamba): MambaVisionMixer(
          (in_proj): Linear(in_features=64, out_features=64, bias=False)
          (x_proj): Linear(in_features=32, out_features=20, bias=False)
          (dt_proj): Linear(in_features=4, out_features=32, bias=True)
          (out_proj): Linear(in_features=64, out_features=64, bias=False)
          (conv1d_x): Conv1d(32, 32, kernel_size=(3,), stride=(1,), groups=32, bias=False)
          (conv1d_z): Conv1d(32, 32, kernel_size=(3,), stride=(1,), groups=32, bias=False)
        )
        (drop_path): Identity()
        (norm2): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=64, out_features=256, bias=True)
          (act): GELU(approximate='none')
          (drop1): Dropout(p=0, inplace=False)
          (norm): Identity()
          (fc2): Linear(in_features=256, out_features=64, bias=True)
          (drop2): Dropout(p=0, inplace=False)
        )
      )
      (freq_block_mamba): Block_mamba(
        (norm1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
        (B_mamba): MambaVisionMixer(
          (in_proj): Linear(in_features=64, out_features=64, bias=False)
          (x_proj): Linear(in_features=32, out_features=20, bias=False)
          (dt_proj): Linear(in_features=4, out_features=32, bias=True)
          (out_proj): Linear(in_features=64, out_features=64, bias=False)
          (conv1d_x): Conv1d(32, 32, kernel_size=(3,), stride=(1,), groups=32, bias=False)
          (conv1d_z): Conv1d(32, 32, kernel_size=(3,), stride=(1,), groups=32, bias=False)
        )
        (drop_path): Identity()
        (norm2): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=64, out_features=256, bias=True)
          (act): GELU(approximate='none')
          (drop1): Dropout(p=0, inplace=False)
          (norm): Identity()
          (fc2): Linear(in_features=256, out_features=64, bias=True)
          (drop2): Dropout(p=0, inplace=False)
        )
      )
      (time_block_attention): Block_Attention(
        (norm1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
        (B_Attention): Attention(
          (qkv): Linear(in_features=64, out_features=192, bias=True)
          (q_norm): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
          (k_norm): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
          (attn_drop): Dropout(p=0, inplace=False)
          (proj): Linear(in_features=64, out_features=64, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (drop_path): Identity()
        (norm2): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=64, out_features=256, bias=True)
          (act): GELU(approximate='none')
          (drop1): Dropout(p=0.0, inplace=False)
          (norm): Identity()
          (fc2): Linear(in_features=256, out_features=64, bias=True)
          (drop2): Dropout(p=0.0, inplace=False)
        )
      )
      (freq_block_attention): Block_Attention(
        (norm1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
        (B_Attention): Attention(
          (qkv): Linear(in_features=64, out_features=192, bias=True)
          (q_norm): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
          (k_norm): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
          (attn_drop): Dropout(p=0, inplace=False)
          (proj): Linear(in_features=64, out_features=64, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (drop_path): Identity()
        (norm2): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=64, out_features=256, bias=True)
          (act): GELU(approximate='none')
          (drop1): Dropout(p=0.0, inplace=False)
          (norm): Identity()
          (fc2): Linear(in_features=256, out_features=64, bias=True)
          (drop2): Dropout(p=0.0, inplace=False)
        )
      )
    )
  )
  (down2_3): Downsample(
    (body): Sequential(
      (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
      (1): Conv2d(64, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (2): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (3): PixelUnshuffle(downscale_factor=2)
    )
  )
  (patch_embed_middle): Patch_Embed_stage(
    (patch_embeds): MB_Deform_Embedding(
      (patch_conv): DWConv2d_BN(
        (offset_generator): Sequential(
          (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=96, bias=False)
          (1): Conv2d(96, 18, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (dcn): DeformConv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=96, bias=False)
        (pwconv): Conv2d(96, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (act): Hardswish()
      )
      (norm): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (TFMamba_attention_middle): ModuleList(
    (0-3): 4 x TF_mamba_attention(
      (time_block_mamba): Block_mamba(
        (norm1): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
        (B_mamba): MambaVisionMixer(
          (in_proj): Linear(in_features=96, out_features=96, bias=False)
          (x_proj): Linear(in_features=48, out_features=22, bias=False)
          (dt_proj): Linear(in_features=6, out_features=48, bias=True)
          (out_proj): Linear(in_features=96, out_features=96, bias=False)
          (conv1d_x): Conv1d(48, 48, kernel_size=(3,), stride=(1,), groups=48, bias=False)
          (conv1d_z): Conv1d(48, 48, kernel_size=(3,), stride=(1,), groups=48, bias=False)
        )
        (drop_path): Identity()
        (norm2): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=96, out_features=384, bias=True)
          (act): GELU(approximate='none')
          (drop1): Dropout(p=0, inplace=False)
          (norm): Identity()
          (fc2): Linear(in_features=384, out_features=96, bias=True)
          (drop2): Dropout(p=0, inplace=False)
        )
      )
      (freq_block_mamba): Block_mamba(
        (norm1): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
        (B_mamba): MambaVisionMixer(
          (in_proj): Linear(in_features=96, out_features=96, bias=False)
          (x_proj): Linear(in_features=48, out_features=22, bias=False)
          (dt_proj): Linear(in_features=6, out_features=48, bias=True)
          (out_proj): Linear(in_features=96, out_features=96, bias=False)
          (conv1d_x): Conv1d(48, 48, kernel_size=(3,), stride=(1,), groups=48, bias=False)
          (conv1d_z): Conv1d(48, 48, kernel_size=(3,), stride=(1,), groups=48, bias=False)
        )
        (drop_path): Identity()
        (norm2): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=96, out_features=384, bias=True)
          (act): GELU(approximate='none')
          (drop1): Dropout(p=0, inplace=False)
          (norm): Identity()
          (fc2): Linear(in_features=384, out_features=96, bias=True)
          (drop2): Dropout(p=0, inplace=False)
        )
      )
      (time_block_attention): Block_Attention(
        (norm1): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
        (B_Attention): Attention(
          (qkv): Linear(in_features=96, out_features=288, bias=True)
          (q_norm): LayerNorm((24,), eps=1e-05, elementwise_affine=True)
          (k_norm): LayerNorm((24,), eps=1e-05, elementwise_affine=True)
          (attn_drop): Dropout(p=0, inplace=False)
          (proj): Linear(in_features=96, out_features=96, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (drop_path): Identity()
        (norm2): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=96, out_features=384, bias=True)
          (act): GELU(approximate='none')
          (drop1): Dropout(p=0.0, inplace=False)
          (norm): Identity()
          (fc2): Linear(in_features=384, out_features=96, bias=True)
          (drop2): Dropout(p=0.0, inplace=False)
        )
      )
      (freq_block_attention): Block_Attention(
        (norm1): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
        (B_Attention): Attention(
          (qkv): Linear(in_features=96, out_features=288, bias=True)
          (q_norm): LayerNorm((24,), eps=1e-05, elementwise_affine=True)
          (k_norm): LayerNorm((24,), eps=1e-05, elementwise_affine=True)
          (attn_drop): Dropout(p=0, inplace=False)
          (proj): Linear(in_features=96, out_features=96, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (drop_path): Identity()
        (norm2): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=96, out_features=384, bias=True)
          (act): GELU(approximate='none')
          (drop1): Dropout(p=0.0, inplace=False)
          (norm): Identity()
          (fc2): Linear(in_features=384, out_features=96, bias=True)
          (drop2): Dropout(p=0.0, inplace=False)
        )
      )
    )
  )
  (up3_2): Upsample(
    (body): Sequential(
      (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=96, bias=False)
      (1): Conv2d(96, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (3): PixelShuffle(upscale_factor=2)
    )
  )
  (concat_level2): Sequential(
    (0): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (patch_embed_decoder_level2): Patch_Embed_stage(
    (patch_embeds): MB_Deform_Embedding(
      (patch_conv): DWConv2d_BN(
        (offset_generator): Sequential(
          (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
          (1): Conv2d(64, 18, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (dcn): DeformConv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
        (pwconv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (act): Hardswish()
      )
      (norm): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (TFMamba_attention_decoder2): ModuleList(
    (0-3): 4 x TF_mamba_attention(
      (time_block_mamba): Block_mamba(
        (norm1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
        (B_mamba): MambaVisionMixer(
          (in_proj): Linear(in_features=64, out_features=64, bias=False)
          (x_proj): Linear(in_features=32, out_features=20, bias=False)
          (dt_proj): Linear(in_features=4, out_features=32, bias=True)
          (out_proj): Linear(in_features=64, out_features=64, bias=False)
          (conv1d_x): Conv1d(32, 32, kernel_size=(3,), stride=(1,), groups=32, bias=False)
          (conv1d_z): Conv1d(32, 32, kernel_size=(3,), stride=(1,), groups=32, bias=False)
        )
        (drop_path): Identity()
        (norm2): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=64, out_features=256, bias=True)
          (act): GELU(approximate='none')
          (drop1): Dropout(p=0, inplace=False)
          (norm): Identity()
          (fc2): Linear(in_features=256, out_features=64, bias=True)
          (drop2): Dropout(p=0, inplace=False)
        )
      )
      (freq_block_mamba): Block_mamba(
        (norm1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
        (B_mamba): MambaVisionMixer(
          (in_proj): Linear(in_features=64, out_features=64, bias=False)
          (x_proj): Linear(in_features=32, out_features=20, bias=False)
          (dt_proj): Linear(in_features=4, out_features=32, bias=True)
          (out_proj): Linear(in_features=64, out_features=64, bias=False)
          (conv1d_x): Conv1d(32, 32, kernel_size=(3,), stride=(1,), groups=32, bias=False)
          (conv1d_z): Conv1d(32, 32, kernel_size=(3,), stride=(1,), groups=32, bias=False)
        )
        (drop_path): Identity()
        (norm2): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=64, out_features=256, bias=True)
          (act): GELU(approximate='none')
          (drop1): Dropout(p=0, inplace=False)
          (norm): Identity()
          (fc2): Linear(in_features=256, out_features=64, bias=True)
          (drop2): Dropout(p=0, inplace=False)
        )
      )
      (time_block_attention): Block_Attention(
        (norm1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
        (B_Attention): Attention(
          (qkv): Linear(in_features=64, out_features=192, bias=True)
          (q_norm): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
          (k_norm): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
          (attn_drop): Dropout(p=0, inplace=False)
          (proj): Linear(in_features=64, out_features=64, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (drop_path): Identity()
        (norm2): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=64, out_features=256, bias=True)
          (act): GELU(approximate='none')
          (drop1): Dropout(p=0.0, inplace=False)
          (norm): Identity()
          (fc2): Linear(in_features=256, out_features=64, bias=True)
          (drop2): Dropout(p=0.0, inplace=False)
        )
      )
      (freq_block_attention): Block_Attention(
        (norm1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
        (B_Attention): Attention(
          (qkv): Linear(in_features=64, out_features=192, bias=True)
          (q_norm): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
          (k_norm): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
          (attn_drop): Dropout(p=0, inplace=False)
          (proj): Linear(in_features=64, out_features=64, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (drop_path): Identity()
        (norm2): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=64, out_features=256, bias=True)
          (act): GELU(approximate='none')
          (drop1): Dropout(p=0.0, inplace=False)
          (norm): Identity()
          (fc2): Linear(in_features=256, out_features=64, bias=True)
          (drop2): Dropout(p=0.0, inplace=False)
        )
      )
    )
  )
  (up2_1): Upsample(
    (body): Sequential(
      (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
      (1): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (3): PixelShuffle(upscale_factor=2)
    )
  )
  (concat_level1): Sequential(
    (0): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (patch_embed_decoder_level1): Patch_Embed_stage(
    (patch_embeds): MB_Deform_Embedding(
      (patch_conv): DWConv2d_BN(
        (offset_generator): Sequential(
          (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
          (1): Conv2d(32, 18, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (dcn): DeformConv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
        (pwconv): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (act): Hardswish()
      )
      (norm): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (TFMamba_attention_decoder1): ModuleList(
    (0-3): 4 x TF_mamba_attention(
      (time_block_mamba): Block_mamba(
        (norm1): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (B_mamba): MambaVisionMixer(
          (in_proj): Linear(in_features=32, out_features=32, bias=False)
          (x_proj): Linear(in_features=16, out_features=18, bias=False)
          (dt_proj): Linear(in_features=2, out_features=16, bias=True)
          (out_proj): Linear(in_features=32, out_features=32, bias=False)
          (conv1d_x): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
          (conv1d_z): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
        )
        (drop_path): Identity()
        (norm2): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=32, out_features=128, bias=True)
          (act): GELU(approximate='none')
          (drop1): Dropout(p=0, inplace=False)
          (norm): Identity()
          (fc2): Linear(in_features=128, out_features=32, bias=True)
          (drop2): Dropout(p=0, inplace=False)
        )
      )
      (freq_block_mamba): Block_mamba(
        (norm1): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (B_mamba): MambaVisionMixer(
          (in_proj): Linear(in_features=32, out_features=32, bias=False)
          (x_proj): Linear(in_features=16, out_features=18, bias=False)
          (dt_proj): Linear(in_features=2, out_features=16, bias=True)
          (out_proj): Linear(in_features=32, out_features=32, bias=False)
          (conv1d_x): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
          (conv1d_z): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
        )
        (drop_path): Identity()
        (norm2): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=32, out_features=128, bias=True)
          (act): GELU(approximate='none')
          (drop1): Dropout(p=0, inplace=False)
          (norm): Identity()
          (fc2): Linear(in_features=128, out_features=32, bias=True)
          (drop2): Dropout(p=0, inplace=False)
        )
      )
      (time_block_attention): Block_Attention(
        (norm1): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (B_Attention): Attention(
          (qkv): Linear(in_features=32, out_features=96, bias=True)
          (q_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (k_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (attn_drop): Dropout(p=0, inplace=False)
          (proj): Linear(in_features=32, out_features=32, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (drop_path): Identity()
        (norm2): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=32, out_features=128, bias=True)
          (act): GELU(approximate='none')
          (drop1): Dropout(p=0.0, inplace=False)
          (norm): Identity()
          (fc2): Linear(in_features=128, out_features=32, bias=True)
          (drop2): Dropout(p=0.0, inplace=False)
        )
      )
      (freq_block_attention): Block_Attention(
        (norm1): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (B_Attention): Attention(
          (qkv): Linear(in_features=32, out_features=96, bias=True)
          (q_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (k_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (attn_drop): Dropout(p=0, inplace=False)
          (proj): Linear(in_features=32, out_features=32, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (drop_path): Identity()
        (norm2): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=32, out_features=128, bias=True)
          (act): GELU(approximate='none')
          (drop1): Dropout(p=0.0, inplace=False)
          (norm): Identity()
          (fc2): Linear(in_features=128, out_features=32, bias=True)
          (drop2): Dropout(p=0.0, inplace=False)
        )
      )
    )
  )
  (mag_patch_embed_refinement): Patch_Embed_stage(
    (patch_embeds): MB_Deform_Embedding(
      (patch_conv): DWConv2d_BN(
        (offset_generator): Sequential(
          (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
          (1): Conv2d(32, 18, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (dcn): DeformConv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
        (pwconv): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (act): Hardswish()
      )
      (norm): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (mag_refinement): ModuleList(
    (0-3): 4 x TF_mamba_attention(
      (time_block_mamba): Block_mamba(
        (norm1): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (B_mamba): MambaVisionMixer(
          (in_proj): Linear(in_features=32, out_features=32, bias=False)
          (x_proj): Linear(in_features=16, out_features=18, bias=False)
          (dt_proj): Linear(in_features=2, out_features=16, bias=True)
          (out_proj): Linear(in_features=32, out_features=32, bias=False)
          (conv1d_x): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
          (conv1d_z): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
        )
        (drop_path): Identity()
        (norm2): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=32, out_features=128, bias=True)
          (act): GELU(approximate='none')
          (drop1): Dropout(p=0, inplace=False)
          (norm): Identity()
          (fc2): Linear(in_features=128, out_features=32, bias=True)
          (drop2): Dropout(p=0, inplace=False)
        )
      )
      (freq_block_mamba): Block_mamba(
        (norm1): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (B_mamba): MambaVisionMixer(
          (in_proj): Linear(in_features=32, out_features=32, bias=False)
          (x_proj): Linear(in_features=16, out_features=18, bias=False)
          (dt_proj): Linear(in_features=2, out_features=16, bias=True)
          (out_proj): Linear(in_features=32, out_features=32, bias=False)
          (conv1d_x): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
          (conv1d_z): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
        )
        (drop_path): Identity()
        (norm2): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=32, out_features=128, bias=True)
          (act): GELU(approximate='none')
          (drop1): Dropout(p=0, inplace=False)
          (norm): Identity()
          (fc2): Linear(in_features=128, out_features=32, bias=True)
          (drop2): Dropout(p=0, inplace=False)
        )
      )
      (time_block_attention): Block_Attention(
        (norm1): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (B_Attention): Attention(
          (qkv): Linear(in_features=32, out_features=96, bias=True)
          (q_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (k_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (attn_drop): Dropout(p=0, inplace=False)
          (proj): Linear(in_features=32, out_features=32, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (drop_path): Identity()
        (norm2): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=32, out_features=128, bias=True)
          (act): GELU(approximate='none')
          (drop1): Dropout(p=0.0, inplace=False)
          (norm): Identity()
          (fc2): Linear(in_features=128, out_features=32, bias=True)
          (drop2): Dropout(p=0.0, inplace=False)
        )
      )
      (freq_block_attention): Block_Attention(
        (norm1): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (B_Attention): Attention(
          (qkv): Linear(in_features=32, out_features=96, bias=True)
          (q_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (k_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (attn_drop): Dropout(p=0, inplace=False)
          (proj): Linear(in_features=32, out_features=32, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (drop_path): Identity()
        (norm2): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=32, out_features=128, bias=True)
          (act): GELU(approximate='none')
          (drop1): Dropout(p=0.0, inplace=False)
          (norm): Identity()
          (fc2): Linear(in_features=128, out_features=32, bias=True)
          (drop2): Dropout(p=0.0, inplace=False)
        )
      )
    )
  )
  (mag_output): Sequential(
    (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  )
  (pha_patch_embed_refinement): Patch_Embed_stage(
    (patch_embeds): MB_Deform_Embedding(
      (patch_conv): DWConv2d_BN(
        (offset_generator): Sequential(
          (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
          (1): Conv2d(32, 18, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (dcn): DeformConv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
        (pwconv): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (act): Hardswish()
      )
      (norm): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (pha_refinement): ModuleList(
    (0-3): 4 x TF_mamba_attention(
      (time_block_mamba): Block_mamba(
        (norm1): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (B_mamba): MambaVisionMixer(
          (in_proj): Linear(in_features=32, out_features=32, bias=False)
          (x_proj): Linear(in_features=16, out_features=18, bias=False)
          (dt_proj): Linear(in_features=2, out_features=16, bias=True)
          (out_proj): Linear(in_features=32, out_features=32, bias=False)
          (conv1d_x): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
          (conv1d_z): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
        )
        (drop_path): Identity()
        (norm2): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=32, out_features=128, bias=True)
          (act): GELU(approximate='none')
          (drop1): Dropout(p=0, inplace=False)
          (norm): Identity()
          (fc2): Linear(in_features=128, out_features=32, bias=True)
          (drop2): Dropout(p=0, inplace=False)
        )
      )
      (freq_block_mamba): Block_mamba(
        (norm1): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (B_mamba): MambaVisionMixer(
          (in_proj): Linear(in_features=32, out_features=32, bias=False)
          (x_proj): Linear(in_features=16, out_features=18, bias=False)
          (dt_proj): Linear(in_features=2, out_features=16, bias=True)
          (out_proj): Linear(in_features=32, out_features=32, bias=False)
          (conv1d_x): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
          (conv1d_z): Conv1d(16, 16, kernel_size=(3,), stride=(1,), groups=16, bias=False)
        )
        (drop_path): Identity()
        (norm2): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=32, out_features=128, bias=True)
          (act): GELU(approximate='none')
          (drop1): Dropout(p=0, inplace=False)
          (norm): Identity()
          (fc2): Linear(in_features=128, out_features=32, bias=True)
          (drop2): Dropout(p=0, inplace=False)
        )
      )
      (time_block_attention): Block_Attention(
        (norm1): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (B_Attention): Attention(
          (qkv): Linear(in_features=32, out_features=96, bias=True)
          (q_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (k_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (attn_drop): Dropout(p=0, inplace=False)
          (proj): Linear(in_features=32, out_features=32, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (drop_path): Identity()
        (norm2): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=32, out_features=128, bias=True)
          (act): GELU(approximate='none')
          (drop1): Dropout(p=0.0, inplace=False)
          (norm): Identity()
          (fc2): Linear(in_features=128, out_features=32, bias=True)
          (drop2): Dropout(p=0.0, inplace=False)
        )
      )
      (freq_block_attention): Block_Attention(
        (norm1): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (B_Attention): Attention(
          (qkv): Linear(in_features=32, out_features=96, bias=True)
          (q_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (k_norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
          (attn_drop): Dropout(p=0, inplace=False)
          (proj): Linear(in_features=32, out_features=32, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (drop_path): Identity()
        (norm2): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=32, out_features=128, bias=True)
          (act): GELU(approximate='none')
          (drop1): Dropout(p=0.0, inplace=False)
          (norm): Identity()
          (fc2): Linear(in_features=128, out_features=32, bias=True)
          (drop2): Dropout(p=0.0, inplace=False)
        )
      )
    )
  )
  (pha_output): Sequential(
    (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  )
  (mask_decoder): MagDecoder(
    (dense_block): DenseBlock(
      (dense_block): ModuleList(
        (0): Sequential(
          (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (2): PReLU(num_parameters=32)
        )
        (1): Sequential(
          (0): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(2, 1), dilation=(2, 1))
          (1): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (2): PReLU(num_parameters=32)
        )
        (2): Sequential(
          (0): Conv2d(96, 32, kernel_size=(3, 3), stride=(1, 1), padding=(4, 1), dilation=(4, 1))
          (1): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (2): PReLU(num_parameters=32)
        )
        (3): Sequential(
          (0): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(8, 1), dilation=(8, 1))
          (1): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (2): PReLU(num_parameters=32)
        )
      )
    )
    (mask_conv): Sequential(
      (0): Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (1): PixelShuffle(upscale_factor=2)
      (2): Conv2d(32, 32, kernel_size=(1, 3), stride=(2, 1), padding=(0, 1), groups=32, bias=False)
      (3): Conv2d(32, 1, kernel_size=(1, 1), stride=(1, 1))
      (4): InstanceNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
      (5): PReLU(num_parameters=1)
      (6): Conv2d(1, 1, kernel_size=(1, 1), stride=(1, 1))
    )
    (lsigmoid): LearnableSigmoid2D()
  )
  (phase_decoder): PhaseDecoder(
    (dense_block): DenseBlock(
      (dense_block): ModuleList(
        (0): Sequential(
          (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (2): PReLU(num_parameters=32)
        )
        (1): Sequential(
          (0): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(2, 1), dilation=(2, 1))
          (1): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (2): PReLU(num_parameters=32)
        )
        (2): Sequential(
          (0): Conv2d(96, 32, kernel_size=(3, 3), stride=(1, 1), padding=(4, 1), dilation=(4, 1))
          (1): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (2): PReLU(num_parameters=32)
        )
        (3): Sequential(
          (0): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(8, 1), dilation=(8, 1))
          (1): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (2): PReLU(num_parameters=32)
        )
      )
    )
    (phase_conv): Sequential(
      (0): Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (1): PixelShuffle(upscale_factor=2)
      (2): Conv2d(32, 32, kernel_size=(1, 3), stride=(2, 1), padding=(0, 1), groups=32, bias=False)
      (3): InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
      (4): PReLU(num_parameters=32)
    )
    (phase_conv_r): Conv2d(32, 1, kernel_size=(1, 1), stride=(1, 1))
    (phase_conv_i): Conv2d(32, 1, kernel_size=(1, 1), stride=(1, 1))
  )
)




        #中间三层换为TFtransformer，
        x2 = self.patch_embed_encoder_level2(x2)
        # for block in self.TSMamba2_encoder:
        #     x2 = block(x2)
        x2 = self.TSTransformer2_encoder(x2)
        x2 = copy2 + x2

        x3 = self.down2_3(x2)

        copy3 = x3
        x3 = self.patch_embed_middle(x3)
        # for block in self.TSMamba_middle:
        #     x3 = block(x3)
        x3 = self.TSTransformer_middle(x3)
        x3 = copy3 + x3

        y2 = self.up3_2(x3)
        y2 = torch.cat([y2, x2], 1)
        y2 = self.concat_level2(y2)

        copy_de2 = y2
        y2 = self.patch_embed_decoder_level2(y2)
        # for block in self.TSMamba2_decoder:
        #     y2 = block(y2)
        y2 = self.TSTransformer2_decoder(y2)
        y2 = copy_de2 + y2



(MambaSEUNet) g515528@515528:~/PycharmProjects/Mamba-SEUNet-main_3TSFM$ python train.py
Traceback (most recent call last):
  File "/home/g515528/PycharmProjects/Mamba-SEUNet-main_3TSFM/train.py", line 460, in <module>
    main()
  File "/home/g515528/PycharmProjects/Mamba-SEUNet-main_3TSFM/train.py", line 428, in main
    cfg = load_config(args.config)
          ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/g515528/PycharmProjects/Mamba-SEUNet-main_3TSFM/utils/util.py", line 11, in load_config
    return yaml.safe_load(file)
           ^^^^^^^^^^^^^^^^^^^^
  File "/home/g515528/software/anaconda3/envs/MambaSEUNet/lib/python3.11/site-packages/yaml/__init__.py", line 125, in safe_load
    return load(stream, SafeLoader)
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/g515528/software/anaconda3/envs/MambaSEUNet/lib/python3.11/site-packages/yaml/__init__.py", line 81, in load
    return loader.get_single_data()
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/g515528/software/anaconda3/envs/MambaSEUNet/lib/python3.11/site-packages/yaml/constructor.py", line 49, in get_single_data
    node = self.get_single_node()
           ^^^^^^^^^^^^^^^^^^^^^^
  File "/home/g515528/software/anaconda3/envs/MambaSEUNet/lib/python3.11/site-packages/yaml/composer.py", line 36, in get_single_node
    document = self.compose_document()
               ^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/g515528/software/anaconda3/envs/MambaSEUNet/lib/python3.11/site-packages/yaml/composer.py", line 55, in compose_document
    node = self.compose_node(None, None)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/g515528/software/anaconda3/envs/MambaSEUNet/lib/python3.11/site-packages/yaml/composer.py", line 84, in compose_node
    node = self.compose_mapping_node(anchor)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/g515528/software/anaconda3/envs/MambaSEUNet/lib/python3.11/site-packages/yaml/composer.py", line 133, in compose_mapping_node
    item_value = self.compose_node(node, item_key)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/g515528/software/anaconda3/envs/MambaSEUNet/lib/python3.11/site-packages/yaml/composer.py", line 84, in compose_node
    node = self.compose_mapping_node(anchor)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/g515528/software/anaconda3/envs/MambaSEUNet/lib/python3.11/site-packages/yaml/composer.py", line 127, in compose_mapping_node
    while not self.check_event(MappingEndEvent):
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/g515528/software/anaconda3/envs/MambaSEUNet/lib/python3.11/site-packages/yaml/parser.py", line 98, in check_event
    self.current_event = self.state()
                         ^^^^^^^^^^^^
  File "/home/g515528/software/anaconda3/envs/MambaSEUNet/lib/python3.11/site-packages/yaml/parser.py", line 428, in parse_block_mapping_key
    if self.check_token(KeyToken):
       ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/g515528/software/anaconda3/envs/MambaSEUNet/lib/python3.11/site-packages/yaml/scanner.py", line 115, in check_token
    while self.need_more_tokens():
          ^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/g515528/software/anaconda3/envs/MambaSEUNet/lib/python3.11/site-packages/yaml/scanner.py", line 152, in need_more_tokens
    self.stale_possible_simple_keys()
  File "/home/g515528/software/anaconda3/envs/MambaSEUNet/lib/python3.11/site-packages/yaml/scanner.py", line 291, in stale_possible_simple_keys
    raise ScannerError("while scanning a simple key", key.mark,
yaml.scanner.ScannerError: while scanning a simple key
  in "recipes/Mamba-SEUNet/Mamba-SEUNet.yaml", line 58, column 3
could not find expected ':'
  in "recipes/Mamba-SEUNet/Mamba-SEUNet.yaml", line 59, column 3

[B=batch_size, F=80, T=100] → 原始输入维度
↓
noisy_mag = rearrange(noisy_mag, 'b f t -> b t f').unsqueeze(1) → [B,1,100,80]
noisy_pha = rearrange(noisy_pha, 'b f t -> b t f').unsqueeze(1) → [B,1,100,80]
x = torch.cat((noisy_mag, noisy_pha), dim=1) → [B,2,100,80]

↓↓↓ 编码阶段 ↓↓↓
x1 = self.dense_encoder(x) → [B,64,100,80] (hid_feature=64)
copy1 = x1 → [B,64,100,80]

x1 = self.patch_embed_encoder_level1(x1) → [B,64,100,80]
TSMamba1_encoder处理后 → [B,64,100,80]
x1 = copy1 + x1 → [B,64,100,80]

x2 = self.down1_2(x1) → [B,128,50,40] (空间下采样2×)
copy2 = x2 → [B,128,50,40]

x2 = self.patch_embed_encoder_level2(x2) → [B,128,50,40]
TSMamba2_encoder处理后 → [B,128,50,40]
x2 = copy2 + x2 → [B,128,50,40]

x3 = self.down2_3(x2) → [B,256,25,20]
copy3 = x3 → [B,256,25,20]

x3 = self.patch_embed_middle(x3) → [B,256,25,20]
TSMamba_middle处理后 → [B,256,25,20]
x3 = copy3 + x3 → [B,256,25,20]

↓↓↓ 解码阶段 ↓↓↓
y2 = self.up3_2(x3) → [B,128,50,40] (空间上采样2×)
y2 = torch.cat([y2, x2], 1) → [B,256,50,40] (通道拼接)
concat_level2处理后 → [B,128,50,40] (1×1卷积压缩通道)
copy_de2 = y2 → [B,128,50,40]

y2 = self.patch_embed_decoder_level2(y2) → [B,128,50,40]
TSMamba2_decoder处理后 → [B,128,50,40]
y2 = copy_de2 + y2 → [B,128,50,40]

y1 = self.up2_1(y2) → [B,64,100,80]
y1 = torch.cat([y1, x1], 1) → [B,128,100,80]
concat_level1处理后 → [B,64,100,80]
copy_de1 = y1 → [B,64,100,80]

y1 = self.patch_embed_decoder_level1(y1) → [B,64,100,80]
TSMamba1_decoder处理后 → [B,64,100,80]
y1 = copy_de1 + y1 → [B,64,100,80]

↓↓↓ 精调阶段 ↓↓↓
mag_input = y1 → [B,64,100,80]
copy_mag = mag_input → [B,64,100,80]
mag_refinement处理后 → [B,64,100,80]
mag = copy_mag + mag_input → [B,64,100,80]
mag_output处理后 → [B,64,100,80] + copy1 → [B,64,100,80]

相位分支同理保持相同维度

↓↓↓ 最终输出 ↓↓↓
mask_decoder(mag) → [B,1,100,80]
denoised_mag = rearrange后 → [B,80,100]

phase_decoder(pha) → [B,1,100,80]
denoised_pha = rearrange后 → [B,80,100]

denoised_com → [B,80,100,2]
