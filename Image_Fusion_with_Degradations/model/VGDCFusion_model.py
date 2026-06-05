import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
import math
from einops import rearrange


class VGDCFusion(nn.Module):
    def __init__(self, model_clip, inp_A_channels=3, inp_B_channels=3, out_channels=3,
                 dim=48, num_blocks=[2, 2, 2, 2],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 msconv_layers = [3,3,3,3]
                 ):

        super(VGDCFusion, self).__init__()

        self.model_clip = model_clip
        self.model_clip.eval()
        self.encoder_A = Encoder_A(inp_channels=inp_A_channels, dim=dim, num_blocks=num_blocks, heads=heads,
                                   ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type,msconv_layer=msconv_layers)
        self.encoder_B = Encoder_B(inp_channels=inp_B_channels, dim=dim, num_blocks=num_blocks, heads=heads,
                                   ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type,msconv_layer=msconv_layers)
        
        self.decoder_level4 = GuidanceFusion(in_channels=dim * 2 ** 3, transformer_depth=num_blocks[3], num_heads=heads[3],
                                             ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type,
                                             msconv_layers= msconv_layers[3],use_x_fu=False)

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        
        self.decoder_level3 = GuidanceFusion(in_channels=dim * 2 ** 2, transformer_depth=num_blocks[2],
                                             num_heads=heads[2],
                                             ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                             LayerNorm_type=LayerNorm_type,
                                             msconv_layers=msconv_layers[2],use_x_fu=True)

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        
        self.decoder_level2 = GuidanceFusion(in_channels=dim * 2 ** 1, transformer_depth=num_blocks[1],
                                             num_heads=heads[1],
                                             ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                             LayerNorm_type=LayerNorm_type,
                                             msconv_layers=msconv_layers[1],use_x_fu=True)

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
    
        self.decoder_level1 = GuidanceFusion(in_channels=dim, transformer_depth=num_blocks[0],
                                             num_heads=heads[0],
                                             ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                             LayerNorm_type=LayerNorm_type,
                                             msconv_layers=msconv_layers[0],use_x_fu=True)

        self.re_conv = nn.Sequential(
            nn.Conv2d(int(dim), int(dim * 2 / 3), kernel_size=3, stride=1, padding=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(dim * 2 / 3), int(dim / 3), kernel_size=3, stride=1, padding=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(dim / 3), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        )

    def forward(self, inp_img_A, inp_img_B, text_ir, text_vi):
        b = inp_img_A.shape[0]
        
        text_feat_ir, text_feat_vi = self.get_text_feature(text_ir.expand(b, -1), text_vi.expand(b, -1))
        text_feat_ir = text_feat_ir.to(inp_img_A.dtype)
        text_feat_vi = text_feat_vi.to(inp_img_A.dtype)

        out_enc_level4_vi, out_enc_level3_vi, out_enc_level2_vi, out_enc_level1_vi = self.encoder_A(inp_img_A, text_feat_vi)
        out_enc_level4_ir, out_enc_level3_ir, out_enc_level2_ir, out_enc_level1_ir = self.encoder_B(inp_img_B, text_feat_ir)

        out_dec_level4 = self.decoder_level4(out_enc_level4_ir,out_enc_level4_vi,text_feat_ir,text_feat_vi)
        
        inp_dec_level3 = self.up4_3(out_dec_level4)

        out_dec_level3 = self.decoder_level3(out_enc_level3_ir,out_enc_level3_vi,text_feat_ir,text_feat_vi, inp_dec_level3)
    
        inp_dec_level2 = self.up3_2(out_dec_level3)

        out_dec_level2 = self.decoder_level2(out_enc_level2_ir,out_enc_level2_vi,text_feat_ir,text_feat_vi, inp_dec_level2)
        
        inp_dec_level1 = self.up2_1(out_dec_level2)

        out_dec_level1 = self.decoder_level1(out_enc_level1_ir,out_enc_level1_vi,text_feat_ir,text_feat_vi, inp_dec_level1)

        output_image = self.re_conv(out_dec_level1)

        return output_image

    @torch.no_grad()
    def get_text_feature(self, text_ir, text_vi):
        text_feature_ir = self.model_clip.encode_text(text_ir)
        text_feature_vi = self.model_clip.encode_text(text_vi)
        return text_feature_ir, text_feature_vi


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=True):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.MLP = nn.Sequential(
            nn.Linear(in_channels, in_channels * 2),
            nn.LeakyReLU(),
            nn.Linear(in_channels * 2, out_channels * (1 + self.use_affine_level))
        )

    def forward(self, x, text_embed):
        text_embed = text_embed.unsqueeze(1)#[B,1,in_channels]
        batch = x.shape[0]
        if self.use_affine_level:
            #[B,1,out_channels * 2] -> [B,2*out_channels, 1, 1] -> 2 * [B, out_channels, 1, 1]
            gamma, beta = self.MLP(text_embed).view(batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        return x

class Encoder_A(nn.Module):
    def __init__(self, inp_channels, dim, num_blocks, heads, ffn_expansion_factor, bias,LayerNorm_type, msconv_layer):
        super(Encoder_A, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = GuidanceExtractor(in_channels=dim, transformer_depth=num_blocks[0], num_heads=heads[0],
                                                ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                                LayerNorm_type=LayerNorm_type, msconv_layers=msconv_layer[0])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
    
        self.encoder_level2 = GuidanceExtractor(in_channels=dim * 2 ** 1, transformer_depth=num_blocks[1], num_heads=heads[1],
                                                ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                                LayerNorm_type=LayerNorm_type, msconv_layers=msconv_layer[1])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        
        self.encoder_level3 = GuidanceExtractor(in_channels=dim * 2 ** 2, transformer_depth=num_blocks[2],
                                                num_heads=heads[2],
                                                ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                                LayerNorm_type=LayerNorm_type, msconv_layers=msconv_layer[2])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        
        self.encoder_level4 = GuidanceExtractor(in_channels=dim * 2 ** 3, transformer_depth=num_blocks[3],
                                                num_heads=heads[3],
                                                ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                                LayerNorm_type=LayerNorm_type, msconv_layers=msconv_layer[3])

    def forward(self, inp_img_A, text_vi):
        inp_enc_level1_A = self.patch_embed(inp_img_A)
        out_enc_level1_A = self.encoder_level1(inp_enc_level1_A, text_vi)

        inp_enc_level2_A = self.down1_2(out_enc_level1_A)
        out_enc_level2_A = self.encoder_level2(inp_enc_level2_A, text_vi)

        inp_enc_level3_A = self.down2_3(out_enc_level2_A)
        out_enc_level3_A = self.encoder_level3(inp_enc_level3_A, text_vi)

        inp_enc_level4_A = self.down3_4(out_enc_level3_A)
        out_enc_level4_A = self.encoder_level4(inp_enc_level4_A, text_vi)

        return out_enc_level4_A, out_enc_level3_A, out_enc_level2_A, out_enc_level1_A


class Encoder_B(nn.Module):
    def __init__(self, inp_channels, dim, num_blocks, heads, ffn_expansion_factor,bias, LayerNorm_type, msconv_layer):
        super(Encoder_B, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = GuidanceExtractor(in_channels=dim, transformer_depth=num_blocks[0], num_heads=heads[0],
                                                ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                                LayerNorm_type=LayerNorm_type, msconv_layers=msconv_layer[0])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        
        self.encoder_level2 = GuidanceExtractor(in_channels=dim * 2 ** 1, transformer_depth=num_blocks[1],
                                                num_heads=heads[1],
                                                ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                                LayerNorm_type=LayerNorm_type, msconv_layers=msconv_layer[1])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        
        self.encoder_level3 = GuidanceExtractor(in_channels=dim * 2 ** 2, transformer_depth=num_blocks[2],
                                                num_heads=heads[2],
                                                ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                                LayerNorm_type=LayerNorm_type, msconv_layers=msconv_layer[2])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
       
        self.encoder_level4 = GuidanceExtractor(in_channels=dim * 2 ** 3, transformer_depth=num_blocks[3],
                                                num_heads=heads[3],
                                                ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                                LayerNorm_type=LayerNorm_type, msconv_layers=msconv_layer[3])

    def forward(self, inp_img_B, text_ir):
        inp_enc_level1_B = self.patch_embed(inp_img_B)
        out_enc_level1_B = self.encoder_level1(inp_enc_level1_B,text_ir)

        inp_enc_level2_B = self.down1_2(out_enc_level1_B)
        out_enc_level2_B = self.encoder_level2(inp_enc_level2_B,text_ir)

        inp_enc_level3_B = self.down2_3(out_enc_level2_B)
        out_enc_level3_B = self.encoder_level3(inp_enc_level3_B,text_ir)

        inp_enc_level4_B = self.down3_4(out_enc_level3_B)
        out_enc_level4_B = self.encoder_level4(inp_enc_level4_B,text_ir)

        return out_enc_level4_B, out_enc_level3_B, out_enc_level2_B, out_enc_level1_B

class Fusion_Embed(nn.Module):
    def __init__(self, embed_dim, bias=False):
        super(Fusion_Embed, self).__init__()

        self.fusion_proj = nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1, stride=1, bias=bias)

    def forward(self, x_A, x_B):
        x = torch.concat([x_A, x_B], dim=1)
        x = self.fusion_proj(x)
        return x

##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.dwconv(x)
        x = F.gelu(x)
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))


class MultiScaleConv(nn.Module):
    def __init__(self, in_channels, num_layers):
        super(MultiScaleConv, self).__init__()
        self.num_layers = num_layers

        kernel_sizes = [1, 3, 5]
        self.branches = nn.ModuleList()

        for k in kernel_sizes:
            layers = []
            for _ in range(num_layers):
                layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=k, padding=k // 2))
                layers.append(nn.LeakyReLU(0.2, inplace=True))
            self.branches.append(nn.Sequential(*layers))

        self.fusion_conv = nn.Conv2d(in_channels * len(kernel_sizes), in_channels, kernel_size=1)
        self.norm = nn.GroupNorm(8, in_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        features = [branch(x) for branch in self.branches]
        x_multi = torch.cat(features, dim=1)
        x_multi = self.fusion_conv(x_multi)
        x_out = self.act(self.norm(x_multi + x))  # Residual connection
        return x_out


class GuidanceFusion(nn.Module):
    def __init__(self, in_channels, transformer_depth, num_heads, ffn_expansion_factor, bias, LayerNorm_type, msconv_layers, use_x_fu = True):
        super(GuidanceFusion, self).__init__()

        self.use_x_fu = use_x_fu

        #textfusion
        self.text_fuse_proj = nn.Linear(1024,512)

        # Spatial Attention (Assume provided)
        self.spatial_att_ir = SpatialAttention()
        self.spatial_att_vi = SpatialAttention()

        # After spatial attention + concat + conv for channel adjustment
        self.spatial_fuse_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

        # FeatureWiseAffine for modulation (Assume provided)
        self.fwa_spatial = FeatureWiseAffine(in_channels = 512, out_channels = in_channels)
        if use_x_fu:
            self.fwa_fusion = FeatureWiseAffine(in_channels=512, out_channels=in_channels)

        # Channel and spatial attention for Channel Modulation
        channel_in = in_channels * 2 if use_x_fu else in_channels # still 2C since x_spa_mod + x_spa
        self.channel_att = ChannelAttention(channel_in)
        self.channel_mod_conv = nn.Conv2d(channel_in, in_channels, kernel_size=1)

        # Multi-scale conv
        self.multi_scale_conv = MultiScaleConv(in_channels, msconv_layers)

        # Transformer blocks (L blocks)
        self.transformers = nn.Sequential(*[
            TransformerBlock(dim=in_channels, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for _ in range(transformer_depth)
        ])

        # Final fusion conv
        self.fusion_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

    def forward(self, x_ir, x_vi, text_ir, text_vi, x_fu=None):
        #text fusion
        text_fused = torch.cat([text_ir, text_vi], dim=-1)  # [B, 1024]
        text_fused = self.text_fuse_proj(text_fused)  

        # Spatial Modulation
        x_ir_att = self.spatial_att_ir(x_ir) * x_ir
        x_vi_att = self.spatial_att_vi(x_vi) * x_vi
        x_spa = torch.cat([x_ir_att, x_vi_att], dim=1)
        x_spa = self.spatial_fuse_conv(x_spa)

        # Feature-wise affine modulation with text feature
        x_spa_mod = self.fwa_spatial(x_spa, text_fused)
        # Channel Modulation
        if self.use_x_fu and x_fu is not None:
            x_fu_mod = self.fwa_fusion(x_fu, text_fused)
            x_cat = torch.cat([x_spa_mod, x_fu_mod], dim=1)  # B, 2C, H, W
        else:
            x_cat = x_spa_mod
        x_cat = x_cat * self.channel_att(x_cat)
        x_channel = self.channel_mod_conv(x_cat)

        # Multi-scale conv
        x_local = self.multi_scale_conv(x_channel)

        # Transformer branch
        x_global = self.transformers(x_channel)

        # Final fusion
        x_fused = torch.cat([x_local, x_global], dim=1)
        x_fusion = self.fusion_conv(x_fused)

        return x_fusion


class GuidanceExtractor(nn.Module):
    def __init__(self, in_channels, transformer_depth, num_heads, ffn_expansion_factor, bias, LayerNorm_type, msconv_layers):
        super(GuidanceExtractor, self).__init__()

        self.fwa_guidance = FeatureWiseAffine(in_channels = 512, out_channels = in_channels)

        self.multi_scale_conv = MultiScaleConv(in_channels, msconv_layers)

        self.transformers = nn.Sequential(*[
            TransformerBlock(dim=in_channels, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                            LayerNorm_type=LayerNorm_type) for _ in range(transformer_depth)
        ])

        self.channel_attention = ChannelAttention(2 * in_channels)
        self.channel_conv = nn.Conv2d(2 * in_channels, in_channels, 1)

    def forward(self, x, f_t):
        x_mod = self.fwa_guidance(x, f_t)

        x_local = self.multi_scale_conv(x_mod)
        x_global = self.transformers(x_mod)

        x_concat = torch.cat([x_local, x_global], dim=1)
        x_att = x_concat * self.channel_attention(x_concat)
        x_out = self.channel_conv(x_att)

        return x_out