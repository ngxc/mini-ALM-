import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import torch.nn as nn
import librosa
import numpy as np
from transformers import (
    ClapModel,
    ClapProcessor,
    GPT2LMHeadModel,
    GPT2Tokenizer
)

# ============================
# 1. 路径配置
# ============================
BASE = os.getcwd()

CLAP_LOCAL = os.path.join(BASE, "clap_local")
GPT2_LOCAL = os.path.join(BASE, "gpt2_local")


CLAP_ID = "laion/clap-htsat-unfused"
GPT2_ID = "gpt2"


# ============================
# 2. 模型定义
# ============================

class TemporalQFormerBridge(nn.Module):
    def __init__(self, dim=768, num_queries=32, nhead=8, num_layers=3, max_audio_tokens=64):
        super().__init__()
        # 与训练一致的位置编码
        self.queries = nn.Parameter(torch.randn(1, num_queries, dim))
        self.query_pos = nn.Parameter(torch.randn(1, num_queries, dim))
        self.audio_pos = nn.Parameter(torch.randn(1, max_audio_tokens, dim))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim, nhead=nhead, dim_feedforward=dim * 4,
            dropout=0.1, activation='gelu', batch_first=True, norm_first=True
        )
        self.q_former = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, audio_features):
        batch_size = audio_features.size(0)
        # 注入时间位置信息
        audio_features = audio_features + self.audio_pos
        queries = self.queries.expand(batch_size, -1, -1) + self.query_pos
        query_outputs = self.q_former(tgt=queries, memory=audio_features)
        return self.norm(self.proj(query_outputs))


class AudioCaptioningModel(nn.Module):
    def __init__(self, clap_path, gpt2_path):
        super().__init__()
        # 智能加载
        self.audio_encoder = ClapModel.from_pretrained(clap_path).audio_model
        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_path)
        self.bridge = TemporalQFormerBridge(dim=768, num_queries=32)

    def forward(self, input_features):
        audio_out = self.audio_encoder(input_features).last_hidden_state
        if audio_out.ndim == 4: audio_out = audio_out.flatten(2)
        audio_out = audio_out.transpose(1, 2)
        # 通过 Q-Former 压缩并提取时序特征
        audio_embeds = self.bridge(audio_out)
        return audio_embeds


# ============================
# 3. 推理封装类
# ============================

class AudioCaptioner:
    def __init__(self, weights_path, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # 路径选择逻辑
        clap_path = CLAP_LOCAL if os.path.exists(CLAP_LOCAL) else CLAP_ID
        gpt2_path = GPT2_LOCAL if os.path.exists(GPT2_LOCAL) else GPT2_ID

        print(f">>> 加载配置: CLAP={clap_path}, GPT2={gpt2_path}")

        # 加载分词器和处理器
        self.processor = ClapProcessor.from_pretrained(clap_path)
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_path)

        # 初始化模型
        self.model = AudioCaptioningModel(clap_path, gpt2_path).to(self.device)

        # 加载训练好的权重
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"未找到权重文件: {weights_path}")

        print(f">>> 正在从 {weights_path} 加载权重...")
        state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(">>> 推理引擎已就绪。")





    @torch.no_grad()
    def predict(self, audio_file, max_new_tokens=150, temperature=0.7):

        audio, _ = librosa.load(audio_file, sr=48000, mono=True)
        if len(audio) > 480000:
            audio = audio[:480000]
        else:
            audio = np.pad(audio, (0, 480000 - len(audio)))

        inputs = self.processor(audios=[audio], sampling_rate=48000, return_tensors="pt")
        input_features = inputs.input_features.to(self.device)


        audio_embeds = self.model(input_features)  # [1, 32, 768]


        attention_mask = torch.ones(audio_embeds.shape[:2], device=self.device)

        outputs = self.model.gpt2.generate(
            inputs_embeds=audio_embeds,
            attention_mask=attention_mask,
            max_new_tokens=80,
            do_sample=True,
            temperature=0.8,
            top_p=0.6,
            repetition_penalty=1.5,
            no_repeat_ngram_size=3,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id
        )

        caption = self.tokenizer.decode(outputs[0], skip_special_tokens=True)


        period_pos = caption.find(".")
        if period_pos != -1:
            caption = caption[:period_pos + 1]

        return caption




if __name__ == "__main__":

    MY_WEIGHTS = "temporal_qformer_final.pt"

    MY_AUDIO = "subway.wav"

    try:
        engine = AudioCaptioner(MY_WEIGHTS)
        result = engine.predict(MY_AUDIO)

        print("\n" + "=" * 50)
        print(f"【测试音频】: {MY_AUDIO}")
        print(f"【模型描述】: {result}")
        print("=" * 50 + "\n")

    except Exception as e:
        print(f"推理失败: {e}")