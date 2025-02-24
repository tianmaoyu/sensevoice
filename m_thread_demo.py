import threading
import queue
import time
import pyaudio
import numpy as np
from antlr4.tree.Trees import Trees
from funasr import AutoModel
import torch
from sympy.codegen.cnodes import struct

# 配置模型参数
chunk_size = [0, 10, 5]  # [0, 10, 5] 600ms, [0, 8, 4] 480ms
encoder_chunk_look_back = 6  # number of chunks to lookback for encoder self-attention
decoder_chunk_look_back = 1  # number of encoder chunks to lookback for decoder cross-attention
model = AutoModel(model="paraformer-zh-streaming", model_revision="v2.0.4")
# 加载 Silero VAD 模型
# model_vad, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', trust_repo=True)



# 录音参数
FORMAT = pyaudio.paInt16  # 16-bit audio format
CHANNELS = 1  # 单声道
sampling_rate = 16000  # 采样率 (Hz)
chunk_duration = 0.6  # 每个音频块的时长（秒）
FRAME_SIZE = int(sampling_rate * chunk_duration)  # 每个音频块的样本数

audio_queue = queue.Queue()  # 用于在线程间传递音频块的队列

# 缓存用于流式识别
cache = {}


# 录音线程函数
def record_audio():
    p = pyaudio.PyAudio()
    # 打开音频流
    stream = p.open(format=FORMAT,channels=CHANNELS,  rate=sampling_rate,input=True, frames_per_buffer=FRAME_SIZE)
    print("开始录音...")

    try:
        while True:
            # 读取音频数据
            data = stream.read(FRAME_SIZE, exception_on_overflow=False)
            # 将二进制数据转换为 numpy 数组
            audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

            audio_queue.put(audio_data)
            # # VAD 检测 只能 512 个样本也就是32ms
            # audio_tensor = torch.from_numpy(audio_data).float()
            # speech_prob = model_vad(audio_tensor, sampling_rate).item()
            #
            # if speech_prob > 0.35:
            #     print("\033[32m.\033[0m", end="", flush=True)
            #     # 将音频块放入队列
            #     audio_queue.put(audio_data)

    except KeyboardInterrupt:
        print("录音终止")
    finally:
        # 关闭音频流
        stream.stop_stream()
        stream.close()
        p.terminate()

cach_text=[]
# 识别线程函数
def recognize_audio():
    while True:
        # audio_queue 自动阻塞
        speech_chunk = audio_queue.get()
        is_final = False  # 这里可以根据需要设置是否为最后一个块
        res = model.generate(input=speech_chunk,
                             cache=cache,
                             language="zh",
                             is_final=is_final,
                             vad_model="fsmn-vad",
                             disable_pbar=True,
                             chunk_size=chunk_size,
                             encoder_chunk_look_back=encoder_chunk_look_back,
                             decoder_chunk_look_back=decoder_chunk_look_back)

        if res[0]["text"]:
            cach_text.append(res[0]["text"])
            print(''.join(cach_text),end="\r",flush=Trees)




# 启动录音线程
record_thread = threading.Thread(target=record_audio, daemon=True)
record_thread.start()

# 启动识别线程
recognize_thread = threading.Thread(target=recognize_audio, daemon=True)
recognize_thread.start()

# 主线程保持运行，防止程序退出
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("程序终止")
