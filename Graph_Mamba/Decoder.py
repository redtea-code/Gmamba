import torch
from torch import nn
import torch.nn.functional as F
from cross_atten.sd_cross_atten import CrossAttention
from cross_atten.corss_ft_transformer import FeedForward


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class Attention(nn.Module):
    """因果注意力"""

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            casual: bool = True
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.casual = casual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if self.casual:
            mask = torch.tril(torch.ones(N, N)).bool().to(x.device)
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model=64, d_cross=64, dropout_rate=0.4, d_ff=2, n_head=4):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = Attention(dim=d_model, num_heads=n_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.dec_enc_attn = CrossAttention(d_embed=d_model, n_heads=n_head, d_cross=d_cross)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, mult=d_ff, dropout=0)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, dec_layer_inputs, enc_outputs, dec_self_attn_mask=None, dec_enc_attn_mask=None):
        """
        dec_layer_inputs: [batch_size, tgt_len, d_model]
        enc_outputs: [batch_size, src_len, d_model]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        """
        residual1 = dec_layer_inputs.clone()
        dec_self_attn_outputs = self.dec_self_attn(dec_layer_inputs)
        outputs1 = self.norm1(dec_self_attn_outputs + residual1)

        residual2 = outputs1.clone()
        dec_enc_attn_outputs = self.dec_enc_attn(outputs1, enc_outputs)
        outputs2 = self.norm2(dec_enc_attn_outputs + residual2)

        residual3 = outputs2.clone()
        ffn_outputs = self.ffn(outputs2)
        ffn_outputs = self.dropout(ffn_outputs)
        outputs3 = self.norm3(ffn_outputs + residual3)

        return outputs3


class DecoderLayer2(nn.Module):
    def __init__(self, d_model=64, d_cross=64, dropout_rate=0.4, d_ff=2, n_head=4):
        super(DecoderLayer2, self).__init__()
        self.dec_enc_attn = CrossAttention(d_embed=d_model, n_heads=n_head, d_cross=d_cross)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, mult=d_ff, dropout=dropout_rate)
        self.norm3 = nn.Identity()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, dec_layer_inputs, enc_outputs):
        """
        dec_layer_inputs: [batch_size, tgt_len, d_model]
        enc_outputs: [batch_size, src_len, d_model]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        """
        residual2 = dec_layer_inputs.clone()
        dec_enc_attn_outputs = self.dec_enc_attn(dec_layer_inputs, enc_outputs)
        outputs2 = self.norm2(dec_enc_attn_outputs + residual2)

        residual3 = outputs2.clone()
        ffn_outputs = self.ffn(outputs2)
        ffn_outputs = self.dropout(ffn_outputs)
        outputs3 = self.norm3(ffn_outputs + residual3)

        return outputs3


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # approx_gelu = lambda: nn.GELU(approximate="tanh")

        self.mlp = FeedForward(hidden_size, mult=mlp_ratio, dropout=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, ):
        """x: (B,N*2) c: (B,N)
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=2)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class DiTBlock2(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.attn = CrossAttention(n_heads=num_heads, d_embed=hidden_size, d_cross=hidden_size)
        # self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # approx_gelu = lambda: nn.GELU(approximate="tanh")

        self.mlp = FeedForward(hidden_size, mult=mlp_ratio, dropout=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        )

    def forward(self, x, c, ):
        """x: (B,N*2) c: (B,N)"""
        cond_msa, gate_msa, gate_mlp = self.adaLN_modulation(c).chunk(3, dim=2)
        x = x + gate_msa * self.attn(self.norm1(x), cond_msa)
        x = x + gate_mlp * self.mlp(self.norm2(x))
        return x


class TriModalCrossAttention(nn.Module):
    def __init__(self, input_dim):
        super(TriModalCrossAttention, self).__init__()
        self.W_q1 = nn.Linear(input_dim, input_dim)
        self.W_k1 = nn.Linear(input_dim, input_dim)
        self.W_v1 = nn.Linear(input_dim, input_dim)

        self.W_q2 = nn.Linear(input_dim, input_dim)
        self.W_k2 = nn.Linear(input_dim, input_dim)
        self.W_v2 = nn.Linear(input_dim, input_dim)

        self.W_q3 = nn.Linear(input_dim, input_dim)
        self.W_k3 = nn.Linear(input_dim, input_dim)
        self.W_v3 = nn.Linear(input_dim, input_dim)

        self.W_o1 = nn.Linear(input_dim * 2, input_dim)
        self.W_o2 = nn.Linear(input_dim * 2, input_dim)
        self.W_o3 = nn.Linear(input_dim * 2, input_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x1, x2, x3):
        # x1, x2, x3: [B, N, input_dim] [8, 256, 1]
        batch_size, seq_len, _ = x1.size()

        # Linear transformations for each modality
        queries1 = self.W_q1(x1)
        keys2 = self.W_k2(x2)
        values2 = self.W_v2(x2)

        queries2 = self.W_q2(x2)
        keys3 = self.W_k3(x3)
        values3 = self.W_v3(x3)

        queries3 = self.W_q3(x3)
        keys1 = self.W_k1(x1)
        values1 = self.W_v1(x1)

        # Scaled dot-product attention
        attention_scores1 = torch.matmul(queries1, keys2.transpose(-2, -1)) / (x1.size(-1) ** 0.5)  # [B, N, N]
        attention_weights1 = F.softmax(attention_scores1, dim=-1)
        context1 = torch.matmul(self.dropout(attention_weights1), values2)  # [B, N, input_dim]

        attention_scores2 = torch.matmul(queries2, keys3.transpose(-2, -1)) / (x2.size(-1) ** 0.5)  # [B, N, N]
        attention_weights2 = F.softmax(attention_scores2, dim=-1)
        context2 = torch.matmul(self.dropout(attention_weights2), values3)  # [B, N, input_dim]

        attention_scores3 = torch.matmul(queries3, keys1.transpose(-2, -1)) / (x3.size(-1) ** 0.5)  # [B, N, N]
        attention_weights3 = F.softmax(attention_scores3, dim=-1)
        context3 = torch.matmul(self.dropout(attention_weights3), values1)  # [B, N, input_dim]

        # Concatenate context with input for each modality
        combined1 = torch.cat((x1, context1), dim=-1)  # [B, N, input_dim * 2]
        combined2 = torch.cat((x2, context2), dim=-1)  # [B, N, input_dim * 2]
        combined3 = torch.cat((x3, context3), dim=-1)  # [B, N, input_dim * 2]

        # Linear transformations and output for each modality
        output1 = self.W_o1(combined1)
        output2 = self.W_o2(combined2)
        output3 = self.W_o3(combined3)

        global_feature = torch.cat((output1, output2, output3), dim=1)  # (2, 768, 1)
        return output1, output2, output3, global_feature


class TriModalCrossAttention_ver2(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads):
        super(TriModalCrossAttention_ver2, self).__init__()
        # 多头注意力机制
        self.cross_attention1 = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)
        self.cross_attention2 = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)
        self.cross_attention3 = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)

        # 用于将q拼接之后的维度映射到最终输出的维度
        self.q_linear = nn.Linear(input_dim * 3, input_dim)
        # self.linear = nn.Linear(3 * input_dim, output_dim)

    def forward(self, tensor1, tensor2, tensor3):
        # 假设输入的张量维度为 [B, N, input_dim]

        # 拼接三个模态张量作为q
        q = torch.cat((tensor1, tensor2, tensor3), dim=-1)  # [B, N, 3 * input_dim]
        q = self.q_linear(q)

        # 第一个交叉注意力：tensor1 作为 k, v
        attn_output1, _ = self.cross_attention1(q, tensor1, tensor1)  # [B, N, input_dim]

        # 第二个交叉注意力：tensor2 作为 k, v
        attn_output2, _ = self.cross_attention2(q, tensor2, tensor2)  # [B, N, input_dim]

        # 第三个交叉注意力：tensor3 作为 k, v
        attn_output3, _ = self.cross_attention3(q, tensor3, tensor3)  # [B, N, input_dim]

        # 将三个交叉注意力的输出拼接
        # combined_output = torch.cat((attn_output1, attn_output2, attn_output3), dim=2)  # [B, N, 3 * input_dim]
        combined_output = torch.cat((attn_output1, attn_output2, attn_output3), dim=1)  # [B, 3 * N, input_dim]

        # 将拼接后的输出映射到最终的输出维度
        # output = self.linear(combined_output)  # [B, N, output_dim]

        return combined_output

if __name__ == '__main__':
    x = torch.rand(1,1,512)
    x1 = torch.rand(1,1,512)
    x2 = torch.rand(1,1,512)
    # model = TriModalCrossAttention_ver2(512,512,8)
    model = TriModalCrossAttention(512)

    y = model(x,x1,x2)
    print(y.shape)