这是一个由GE2E损失训练的把发言人语音嵌入（d-vector）的PyTorch实现。

原始关于GE2E损失的论文在这里：[Generalized End-to-End Loss for Speaker Verification](https://arxiv.org/abs/1710.10467)。

本文档由原始英文文档翻译修订而来。

**使用方法**

```python
import torch
import torchaudio

wav2mel = torch.jit.load("wav2mel.pt")
dvector = torch.jit.load("dvector.pt").eval()

wav_tensor, sample_rate = torchaudio.load("example.wav")
mel_tensor = wav2mel(wav_tensor, sample_rate)  # shape: (frames, mel_dim)
emb_tensor = dvector.embed_utterance(mel_tensor)  # shape: (emb_dim)
```

你也可以一次嵌入一个发言人的多句话：

```python
emb_tensor = dvector.embed_utterances([mel_tensor_1, mel_tensor_2])  # shape: (emb_dim)
```

这个例子中有两个模块：

1. `wav2mel.pt` 是预处理模块，由两个子模块组成：
   - `sox_effects.pt` 用于标准化音量，去除静音，将音频重采样为16 KHz，16位，并将所有通道混合为单通道。
   - `log_melspectrogram.pt` 用于将波形转换为对数梅尔（MEL）频谱图。

2. `dvector.pt` 是发言人编码器。

由于所有模块都使用TorchScript编译，所以你可以简单地加载它们并在任何地方使用，无需任何依赖。

**预训练模型和预处理模块**

你可以从[Releases页面](https://github.com/yistLin/dvector/releases)下载它们。

**评估模型性能**

你可以使用等错误率（equal error rate）来评估模型的性能。例如，从[VoxCeleb1数据集](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html)下载官方测试切分（veri_test.txt和veri_test2.txt），然后运行以下命令：

```python
python equal_error_rate.py VoxCeleb1/test VoxCeleb1/test/veri_test.txt -w wav2mel.pt -c dvector.pt
```

到目前为止，发布的检查点只在VoxCeleb1上进行了训练，没有进行任何数据增强。它在VoxCeleb1的官方测试切分上的性能如下：

| 测试切分 | 等错误率 | 阈值 |
| :---: | :---: | :---: |
| veri_test.txt | 12.0% | 0.222 |
| veri_test2.txt | 11.9% | 0.223 |

**从头开始训练**

- 预处理训练数据
要使用此处提供的脚本，您必须按以下方式组织原始数据：

	来自同一人的所有语音都放在对应发言人目录下
	所有发言人目录都放在根目录
	发言人目录可以有子目录，语音可以放在子目录下

您可以从多个根目录中提取话语，例如
```python
python preprocess.py VoxCeleb1/dev LibriSpeech/train-clean-360 -o preprocessed
```
如果你需要修改一些音频预处理超参数，直接修改data/wav2mel.py。

预处理后，3个预处理模块将保存在输出目录中：

	wav2mel.pt
	sox_effects.pt
	log_melspectrogram.pt
第一个模块wav2mel.pt由第二和第三模块组成。这些模块都使用TorchScript编译，可以在任何地方用于预处理音频数据。


- 训练模型

   你必须指定存储检查点和日志的位置，例如：

   ```python
   python train.py preprocessed <model_dir>
   ```

   在训练过程中，日志将放在`<model_dir>/logs`下，检查点将放在`<model_dir>/checkpoints`下。更多细节，可以使用`python train.py -h`查看使用方法。

- 使用不同的发言人编码器

   默认情况下，我使用的是带有注意力池化的3层LSTM作为发言人编码器，但你可以使用不同架构的发言人编码器。更多信息，请查看`modules/dvector.py`。

- 可视化语音嵌入

  
  你可以使用训练好的d-vector来可视化语音嵌入。注意，你必须像预处理一样组织语音的目录。例如：

   ```python
   python visualize.py LibriSpeech/dev-clean -w wav2mel.pt -c dvector.pt -o tsne.jpg
   ```

   下图是使用t-SNE对LibriSpeech中一些话语的维度降维结果的可视化。
   ![image](https://github.com/hackermengzhi/solid-octo-spork/assets/50409074/b8abbb5b-2ef0-4a94-849e-6c8b8114e653)

更多详细信息，请参阅[GitHub项目页面](https://github.com/yistLin/dvector)。
