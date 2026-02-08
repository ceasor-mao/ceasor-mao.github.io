---
title: StreamVGGT因果注意力版VGGT
tags: 神经网络 前馈三维重建
mathjax: true  
mathjax_autoNumber: true
---

论文pdf链接: [Streaming 4D Visual Geometry Transformer](https://arxiv.org/pdf/2507.11539)

# 研究思路与历史意义
VGGT本身其实还是比较heavy的架构，注意力机制是存在很大的优化空间的（至少是发论文的空间）。除此之外还存在一个严重缺陷，一次将所有图像一同输入所有结果一同输出，对于三维重建来说是合理的，但是到了实时系统上如SLAM等就存在问题，实时系统希望的应该是输入一张图像然后输出这张图像的结果。

因此该论文将Transformer修改为了循环网络的架构，把空间注意力修改为了可缓存机制，把全局注意力改写成因果注意力。

# 核心
## 因果注意力
VGGT原始的注意力是全局的自注意力，在T时刻的token可以看到（0、1、...、T-1、T+1、...）所有的token,为了实现因果关系，该论文通过掩码机制（其实就是Transformer Decoder中的机制）在训练时通过下三角掩码，让T时刻的token只能看到以前的token，在计算未来时，也会将先前计算好的token都保存起来，以供下次计算使用。剖开表面，其实就是将VGGT的Encoder结构替换为了Decoder结构。
```python
def _process_global_attention(
        self,
        tokens,
        B,
        S,
        P,
        C,
        global_idx,
        pos=None,
        past_key_values_block=None,
        use_cache=False,
        past_frame_idx=0
    ) -> Union[Tuple[torch.Tensor, int, List[torch.Tensor]], Tuple[torch.Tensor, int, List[torch.Tensor], List]]:
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
                """
        
        if tokens.shape != (B, S * P, C):
            tokens = tokens.reshape(B, S, P, C).reshape(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.reshape(B, S, P, 2).reshape(B, S * P, 2)
            
        intermediates = []

        for _ in range(self.aa_block_size):
            if not use_cache:
                L = S * P
                frame_ids = torch.arange(L, device=tokens.device) // P  # [0,0,...,1,1,...,S-1]
                future_frame = frame_ids.unsqueeze(1) < frame_ids.unsqueeze(0)
                attn_mask = future_frame.to(tokens.dtype) * torch.finfo(tokens.dtype).min
            else:
                attn_mask = None
                
            if use_cache:
                tokens, block_kv = self.global_blocks[global_idx](
                    tokens, 
                    pos=pos, 
                    attn_mask=attn_mask, 
                    past_key_values=past_key_values_block,
                    use_cache=True
                )
            else:
                tokens = self.global_blocks[global_idx](tokens, pos=pos, attn_mask=attn_mask)
            global_idx += 1
            intermediates.append(tokens.reshape(B, S, P, C))

            # if self.use_causal_global:
            #     del attn_mask
        if use_cache:
            return tokens, global_idx, intermediates, block_kv
        return tokens, global_idx, intermediates
```

代码的核心改动在```if not use_cache:...```处，在计算前先计算了attn_mask

## 蒸馏学习
原论文引入蒸馏学习是为了解决因果注意力的漂移问题，通过具有全局注意力的模型来引导只有因果注意力的模型进行学习，确实是一个方面，但个人认为还有另一方面，由于模型的核心部分进行了修改，所以需要对整个模型进行重新训练。但是重新训练的成本太大，题外话VGGT的训练成本确实不是个人或小组能够承担的，光数据集就达到了25T。因此这里引入了蒸馏学习，这也是很自然的做法，如果作为作者，我的论文核心创新在与范式上的改变，在效果上能逼近VGGT就已经不错了。


# 实验

# 优势

# 缺陷