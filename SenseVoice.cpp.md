现在到本地
from modelscope import snapshot_download
model_dir = snapshot_download('lovemefan/SenseVoiceGGUF',local_dir='modes')

编译成功后直接使用-流式效果不错
./bin/sense-voice-main -m /Users/eric/PycharmProjects/pythonProject-test/modes/sense-voice-small-fp16.gguf /path/asr_example_zh.wav  -t 4 -ng
