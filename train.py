import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import (
    ClapModel,
    GPT2LMHeadModel,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm

# ============================
# 1. 配置与路径
# ============================
BASE = os.getcwd()
PROCESSED_DIR = os.path.join(BASE, "processed_data")

# 训练超参数
PHYSICAL_BATCH_SIZE = 4
LOGICAL_BATCH_SIZE = 80
ACCUM_STEPS = LOGICAL_BATCH_SIZE // PHYSICAL_BATCH_SIZE
EPOCHS = 50
LR = 4e-5
UNFREEZE_LAYERS = 8
NUM_QUERIES = 32  # 压缩后的时间节点数量
MAX_AUDIO_TOKENS = 64  # CLAP HTSAT 输出的特征长度


# ============================
# 2. 时序感知 Q-Former 桥接层
# ============================
class TemporalQFormerBridge(nn.Module):
    def __init__(self, dim=768, num_queries=32, nhead=8, num_layers=3):
        super().__init__()
        # 1. 可学习的 Query 内容
        self.queries = nn.Parameter(torch.randn(1, num_queries, dim))
        # 2. Query 的位置编码：让 32 个 Query 知道自己的先后顺序
        self.query_pos = nn.Parameter(torch.randn(1, num_queries, dim))
        # 3. 音频特征的位置编码：让模型识别 64 个特征点的时间点
        self.audio_pos = nn.Parameter(torch.randn(1, MAX_AUDIO_TOKENS, dim))

        # Transformer Decoder 作为核心提取器
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=nhead,
            dim_feedforward=dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.q_former = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, audio_features):
        # audio_features: [B, 64, 768]
        batch_size = audio_features.size(0)

        # --- 注入时序信息 ---
        # 为原始音频特征注入 64 个时间点的位置信息
        audio_features = audio_features + self.audio_pos  # [B, 64, 768]

        # 为 32 个查询向量注入位置信息，使其能够定向提取不同时间段的内容
        queries = self.queries.expand(batch_size, -1, -1)
        queries = queries + self.query_pos  # [B, 32, 768]

        # 执行 Cross-Attention 提取
        # tgt: 带位置的查询, memory: 带位置的音频
        query_outputs = self.q_former(tgt=queries, memory=audio_features)

        return self.norm(self.proj(query_outputs))  # 输出 [B, 32, 768]


# ============================
# 3. 完整的音频描述模型
# ============================
class AudioCaptioningModel(nn.Module):
    def __init__(self, train_gpt_layers=UNFREEZE_LAYERS):
        super().__init__()
        # 加载预训练模型
        self.audio_encoder = ClapModel.from_pretrained("laion/clap-htsat-unfused").audio_model
        self.gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")

        # 使用时序感知的桥接层
        self.bridge = TemporalQFormerBridge(dim=768, num_queries=NUM_QUERIES)

        # 冻结策略
        for p in self.audio_encoder.parameters():
            p.requires_grad = False
        for p in self.gpt2.parameters():
            p.requires_grad = False

        # 解冻 GPT2 后 8 层进行微调
        n_total = len(self.gpt2.transformer.h)
        for i in range(n_total - train_gpt_layers, n_total):
            for p in self.gpt2.transformer.h[i].parameters():
                p.requires_grad = True

        # 解冻桥接层与输出头
        for p in self.bridge.parameters(): p.requires_grad = True
        for p in self.gpt2.lm_head.parameters(): p.requires_grad = True
        for p in self.gpt2.transformer.ln_f.parameters(): p.requires_grad = True

    def forward(self, input_features, input_ids, attention_mask):
        # 1. 提取原始音频序列
        audio_out = self.audio_encoder(input_features).last_hidden_state
        if audio_out.ndim == 4: audio_out = audio_out.flatten(2)
        audio_out = audio_out.transpose(1, 2)  # [B, 64, 768]

        # 2. 通过时序 Q-Former 压缩到 32 个带时间顺序的 Token
        audio_embeds = self.bridge(audio_out)  # [B, 32, 768]

        # 3. 获取文本 Embedding
        text_embeds = self.gpt2.transformer.wte(input_ids)

        # 4. 拼接
        combined = torch.cat([audio_embeds, text_embeds], dim=1)

        # 5. 构造 Attention Mask
        audio_mask = torch.ones(audio_embeds.shape[:2], device=input_ids.device)
        mask = torch.cat([audio_mask, attention_mask], dim=1)

        # 6. 构造 Labels (音频位置不计算 Loss)
        labels_audio = torch.full(audio_embeds.shape[:2], -100, device=input_ids.device)
        labels = torch.cat([labels_audio, input_ids], dim=1)

        return self.gpt2(inputs_embeds=combined, attention_mask=mask, labels=labels)


# ============================
# 4. 训练流程
# ============================
class FastDataset(Dataset):
    def __init__(self, dir):
        self.files = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith(".pt")]

    def __len__(self): return len(self.files)

    def __getitem__(self, idx): return torch.load(self.files[idx])


def run_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> 使用设备: {device} ")

    dataset = FastDataset(PROCESSED_DIR)
    loader = DataLoader(dataset, batch_size=PHYSICAL_BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    model = AudioCaptioningModel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    total_steps = (len(loader) // ACCUM_STEPS) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, total_steps // 10, total_steps)

    model.train()
    for epoch in range(EPOCHS):
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        optimizer.zero_grad()

        for i, batch in enumerate(pbar):
            feat = batch["input_features"].to(device)
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)

            out = model(feat, ids, mask)
            loss = out.loss / ACCUM_STEPS
            loss.backward()

            if (i + 1) % ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            pbar.set_postfix(loss=loss.item() * ACCUM_STEPS)

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"temporal_qformer_epoch_{epoch}.pt")

    torch.save(model.state_dict(), "temporal_qformer_final.pt")
    print("训练成功完成！")


if __name__ == "__main__":
    run_training()