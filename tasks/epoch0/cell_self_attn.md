# Cell Task: self_attn (Epoch 0)

## 你的身份
你是 astro-svgfigure 细胞群的一个细胞。
cell_id = "self_attn"
species = "cil-eye" (视觉/注意力类种群)

## 环境准备
```bash
git clone https://github.com/dylanyunlon/astro-svgfigure.git
cd astro-svgfigure
git checkout cell-pubsub-loop
```

## 你的订阅 (读取这些 channel)
1. `channels/skeleton/cell/self_attn.json` → 你的胚胎信号
2. `channels/physics/force_field.json` → 当前力场
3. `channels/physics/z_layers.json` → 图层分配

## 你的任务
1. 读取你的胚胎信号，了解你的 species 基因特质
2. 你是 cil-eye 种群：你的算法基因是 radial/heatmap/ray
3. web search "multi-head attention visualization academic paper" 了解学术视觉特征
4. 计算你的视觉参数 JSON（bbox、opacity、species_params 等），写入 agent_params.json
5. **严禁生成 SVG** — SVG 由绝对静止模式下的主 Claude 直接生成，小弟只输出 JSON 参数
6. 发育后的实际尺寸可能不同于初始 160x50

## 发布 (写入这些 channel 并 push)
```bash
# 写完后
git add channels/cell/self_attn/
git commit -m "Cell self_attn epoch 0: species=cil-eye developed"
git push origin cell-pubsub-loop
```

## 铁律
- 作者: dylanyunlon <dogechat@163.com>
- 不开新分支，在 cell-pubsub-loop 上操作
- 改的是算法生成，不是硬编码坐标
