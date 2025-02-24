import pyaudio
import numpy as np
import torch
import torchaudio
from datetime import datetime
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

# device=torch.device("mps")
# 加载 Silero VAD 模型
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              trust_repo=True)
sampling_rate = 16000  # Silero VAD 支持的采样率

# 初始化 FunASR 语音识别模型
asr_model = AutoModel(
    model="iic/SenseVoiceSmall",
    model_revision="v2.0.4",
)
# model.to(device)

# 音频参数设置
FORMAT = pyaudio.paInt16
CHANNELS = 1
FRAME_SIZE = 512  # Silero VAD 要求每次输入 512 样本
CHUNK_DURATION = FRAME_SIZE / sampling_rate  # 每次处理 32ms 音频

# 语音缓冲区配置
SPEECH_BUFFER_MAX_DURATION = 10  # 最大缓冲时长（秒）
SILENCE_TOLERANCE_DURATION = 1.0  # 静音容忍时长（秒）
speech_buffer = []
current_speech = []
last_speech_time = None  # 记录最后一次检测到语音的时间

# 初始化 PyAudio
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=sampling_rate,
                input=True,
                frames_per_buffer=FRAME_SIZE)

print("-----启动语音识启动----")

try:
    while True:
        # 读取音频数据
        data = stream.read(FRAME_SIZE, exception_on_overflow=False)
        audio_chunk = np.frombuffer(data, dtype=np.int16)

        # 转换为浮点张量
        audio_tensor = torch.from_numpy(audio_chunk.copy()).float() / 32768.0

        # VAD 检测
        speech_prob = model(audio_tensor, sampling_rate).item()

        if speech_prob > 0.35:  # 检测到语音
            print("\033[32m.\033[0m", end="", flush=True)

            # 将原始 int16 数据加入缓冲区（ASR 需要原始格式）
            current_speech.append(audio_chunk)

            # 更新最后一次检测到语音的时间
            last_speech_time = datetime.now()

            # 当缓冲达到最大时长时立即处理
            if len(current_speech) * CHUNK_DURATION >= SPEECH_BUFFER_MAX_DURATION:
                # 合并缓冲音频
                full_audio = np.concatenate(current_speech)

                # 将音频转换为 float32 并归一化
                full_audio_float32 = full_audio.astype(np.float32) / 32768.0

                # 执行语音识别
                text_result = asr_model.generate(input=full_audio_float32, language="zh")[0]["text"]

                # 输出识别结果
                print(f"\n识别结果 {text_result}")

                # 重置缓冲区
                current_speech = []
                last_speech_time = None

        else:  # 无语音
            if len(current_speech) > 0:
                # 计算静音时长
                silence_duration = (datetime.now() - last_speech_time).total_seconds() if last_speech_time else 0

                # 如果静音时长超过容忍时长，则触发识别
                if silence_duration >= SILENCE_TOLERANCE_DURATION:
                    # print("\n\033[93m检测到语音结束，开始识别...\033[0m")

                    # 合并缓冲音频
                    full_audio = np.concatenate(current_speech)

                    # 将音频转换为 float32 并归一化
                    full_audio_float32 = full_audio.astype(np.float32) / 32768.0

                    # 执行语音识别
                    text_result = asr_model.generate(input=full_audio_float32,
                                                     language="zh",  # "zh", "en", "yue", "ja", "ko", "nospeech"
                                                     use_itn=True,
                                                     batch_size_s=60,
                                                     merge_vad=True,  #
                                                     merge_length_s=15,disable_pbar=True
                                                     )

                    # 输出识别结果
                    # print(f"最终识别结果：\033[1m{text_result}\033[0m")

                    text = rich_transcription_postprocess(text_result[0]["text"])

                    print(f"\033[33m识别:\033[0m \033[32m{text}\033[0m")

                    # # 保存音频文件（可选）
                    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    # filename = f"speech_{timestamp}.wav"
                    # torchaudio.save(
                    #     filename,
                    #     torch.from_numpy(full_audio).unsqueeze(0),
                    #     sampling_rate
                    # )
                    # print(f"音频已保存：{filename}")

                    # 重置缓冲区
                    current_speech = []
                    last_speech_time = None

            print("\033[91m \033[0m", end="\r", flush=True)

except KeyboardInterrupt:
    print("\n正在停止...")

finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
