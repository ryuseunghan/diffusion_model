import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch.amp import autocast
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam

from torchvision import transforms as T, utils

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from scipy.optimize import linear_sum_assignment

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator

from denoising_diffusion_pytorch.attend import Attend

from denoising_diffusion_pytorch.version import __version__

# constants

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions

def exists(x):  # x가 None이 아니면 True, 그렇지 않으면 False를 반환합니다.
    return x is not None 

def default(val, d): # 값 val이 존재하면 그대로 반환하고, 존재하지 않으면 기본값 d를 반환합니다.
    if exists(val):
        return val
    return d() if callable(d) else d

def cast_tuple(t, length = 1):  # t가 튜플인지 확인하고, 그렇지 않으면 t를 길이 length만큼의 튜플로 변환합니다.  cast_tuple(5, 3) → (5, 5, 5)
    if isinstance(t, tuple):
        return t
    return ((t,) * length)

def divisible_by(numer, denom): # numer가 denom으로 나누어 떨어지는지(즉, 나머지가 0인지) 확인
    return (numer % denom) == 0

def identity(t, *args, **kwargs): # 입력받은 값 t를 그대로 반환하는 항등 함수입니다. 흔히 기본값으로 사용하는 함수입니다.
    return t

def cycle(dl): # dl이라는 데이터 리스트를 무한히 반복하는 제너레이터(generator) 함수
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num): # 주어진 숫자 num이 정수 제곱근을 가지고 있는지 확인합니다.
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):  # num을 divisor로 나누어 그룹으로 나눈 배열을 반환합니다. 마지막 그룹은 나머지를 포함할 수 있습니다.
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):  # 이미지 image가 주어진 img_type 모드에 맞는지 확인하고, 맞지 않으면 해당 모드로 변환
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):   # [0, 1] 범위의 이미지 값을 [-1, 1] 범위로 변환
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):  #  [-1, 1] 범위의 값을 [0, 1] 범위로 변환
    return (t + 1) * 0.5

# small helper modules

"""
역할: 입력된 텐서의 해상도를 2배로 증가시키는 업샘플링 모듈을 정의합니다.
nn.Upsample(scale_factor=2, mode='nearest'): 입력 이미지를 2배로 확대하며, **nearest**는 최근접 이웃(nearest-neighbor) 방식을 사용하여 업샘플링을 수행합니다.
nn.Conv2d(dim, default(dim_out, dim), 3, padding=1): 업샘플링 후 3x3 컨볼루션을 적용합니다. dim_out이 주어지지 않으면 입력 차원과 동일한 출력 차원을 사용합니다.
"""
def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )
"""
텐서의 차원을 변형하여 2x2 패치 단위로 채널을 확장합니다. 즉, 높이와 너비를 2배로 줄이고, 그에 따라 채널을 4배로 확장합니다.
채널을 4배로 확장한 후 1x1 컨볼루션을 적용하여 차원을 줄입니다. 이때 출력 차원은 dim_out이 주어지지 않으면 입력 차원 dim과 동일하게 설정됩니다.
"""
def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )
"""
입력된 텐서에 대해 **RMSNorm (Root Mean Square Normalization)**을 수행합니다. 이는 입력 피처를 정규화하는 방식 중 하나입니다.
self.scale = dim ** 0.5: dim을 기반으로 정규화에 사용할 스케일 팩터를 계산합니다.
self.g = nn.Parameter(torch.ones(1, dim, 1, 1)): 정규화된 텐서에 곱할 학습 가능한 스케일 파라미터 g를 정의합니다.
F.normalize(x, dim=1): 입력 텐서를 정규화합니다. 이때 RMSNorm은 평균이 0이 되지 않고 표준편차만 정규화하는 방식입니다.
결과: 정규화된 텐서에 스케일 팩터와 **학습 가능한 파라미터 g**를 곱하여 결과를 반환합니다.
"""
class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * self.scale
    

# sinusoidal positional embeds

class SinusoidalPosEmb(Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules
"""
Block 클래스는 기본적인 컨볼루션 블록입니다.
구성 요소:
nn.Conv2d(dim, dim_out, 3): 3x3 커널을 사용하는 2D 컨볼루션 레이어.
RMSNorm(dim_out): RMS 정규화를 수행하는 레이어.
nn.SiLU(): 활성화 함수로 SiLU(Sigmoid-weighted Linear Unit)를 사용.
nn.Dropout(dropout): 드롭아웃을 적용하여 정규화.
scale_shift: 임베딩에서 제공된 scale과 shift 값을 곱하고 더하여 출력을 조정.
"""
class Block(Module):
    def __init__(self, dim, dim_out, dropout = 0.):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return self.dropout(x)
"""
**ResnetBlock**은 Residual Block으로, 원래 입력을 출력에 더하는 skip connection을 가지고 있습니다.
구성 요소:
self.mlp: 시간 임베딩을 위한 다층 퍼셉트론(MLP). 시간 임베딩이 주어졌을 때, 이를 기반으로 scale과 shift를 계산.
block1, block2: 두 개의 Block으로 구성되어 있으며, 중간에 scale_shift가 적용됩니다.
res_conv: 입력과 출력의 차원을 맞추기 위한 1x1 컨볼루션입니다. 차원이 같다면, Identity 레이어가 사용됩니다.
시간 임베딩: 시간 임베딩은 diffusion 모델에서 사용됩니다.
"""
class ResnetBlock(Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, dropout = 0.):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, dropout = dropout)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

"""
**LinearAttention**은 선형 복잡도의 어텐션을 구현한 클래스입니다.
구성 요소:
self.scale: dim_head ** -0.5로 설정된 스케일링 값.
self.to_qkv: 입력을 쿼리, 키, 값으로 변환하기 위한 1x1 컨볼루션입니다.
self.mem_kv: 학습 가능한 메모리 키-값 쌍으로, 각 헤드마다 미리 정의된 키와 값을 학습합니다.
self.to_out: 최종 출력을 위한 1x1 컨볼루션과 RMS 정규화.
동작:
쿼리, 키, 값을 생성한 뒤 소프트맥스를 적용하여 키-값 쌍의 맥락을 계산하고, 이를 기반으로 쿼리와 다시 결합해 최종 출력을 얻습니다.
"""
class LinearAttention(Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        num_mem_kv = 4
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, dim_head, num_mem_kv))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h c n -> b h c n', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -1), ((mk, k), (mv, v)))

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)
"""
설명:
**Attention**은 표준 어텐션 메커니즘을 구현한 클래스입니다.
구성 요소:
self.attend: 쿼리-키-값을 사용하여 어텐션을 수행하는 모듈. 여기서 flash 옵션은 효율적인 어텐션 계산을 의미합니다.
mem_kv: 학습 가능한 메모리 키-값 쌍.
to_qkv: 입력을 쿼리, 키, 값으로 변환하는 1x1 컨볼루션.
동작:
쿼리, 키, 값을 계산하고, 주어진 쿼리와 키
"""
class Attention(Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        num_mem_kv = 4,
        flash = False
    ):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.attend = Attend(flash = flash)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h = self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -2), ((mk, k), (mv, v)))

        out = self.attend(q, k, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

# model

class Unet(Module):
    """
    unet은 diffusion model에서 reverse process를 수행합니다, 즉 노이즈 제거 역할을 합니다.
    dim: UNet에서의 기본 차원.
    init_dim: 초기 입력의 차원. 기본적으로 dim과 동일.
    out_dim: 출력의 차원. (이미지 채널 수와 관련)
    dim_mults: 각 해상도 레벨에서 차원을 얼마나 확장할지를 결정하는 배수 값.
    channels: 입력 이미지의 채널 수. 일반적으로 RGB 이미지의 경우 3입니다.
    self_condition: 자기 조건(Self-Conditioning) 여부를 결정합니다. 자기 조건을 사용하는 경우, 입력 이미지가 2배로 증가합니다.
    learned_variance: 출력에 학습된 분산 값을 포함할지 여부.
    learned_sinusoidal_cond / random_fourier_features: 학습 가능한 사인파 위치 임베딩 또는 랜덤 Fourier 특징을 사용할지 여부.
    dropout: 드롭아웃 확률.
    attn_dim_head / attn_heads: 어텐션 블록에서의 헤드 수와 각 헤드의 차원 크기.
    flash_attn: Flash Attention을 사용할지 여부.
    """
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults = (1, 2, 4, 8),
        channels = 3,
        self_condition = False,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        sinusoidal_pos_emb_theta = 10000,
        dropout = 0.,
        attn_dim_head = 32,
        attn_heads = 4,
        full_attn = None,    # defaults to full attention only for inner most layer
        flash_attn = False
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)    # 7x7 커널을 사용하는 컨볼루션 레이어로, 입력을 init_dim 크기로 변환합니다.

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta = sinusoidal_pos_emb_theta)
            fourier_dim = dim

        
        self.time_mlp = nn.Sequential( # 확산 모델에서 시간이 중요한 역할을 하기 때문에, 시간 정보가 네트워크에 포함됩니다. 사인파 기반 임베딩을 사용해 시간 정보를 포함한 다층 퍼셉트론(MLP)을 구성합니다.
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # attention

        if not full_attn:
            full_attn = (*((False,) * (len(dim_mults) - 1)), True)

        num_stages = len(dim_mults)
        full_attn  = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)

        assert len(full_attn) == len(dim_mults)

        # prepare blocks

        FullAttention = partial(Attention, flash = flash_attn)
        resnet_block = partial(ResnetBlock, time_emb_dim = time_dim, dropout = dropout)

        # layers

        self.downs = ModuleList([])
        self.ups = ModuleList([])
        num_resolutions = len(in_out)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            is_last = ind >= (num_resolutions - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.downs.append(ModuleList([
                resnet_block(dim_in, dim_in),
                resnet_block(dim_in, dim_in),
                attn_klass(dim_in, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = resnet_block(mid_dim, mid_dim)
        self.mid_attn = FullAttention(mid_dim, heads = attn_heads[-1], dim_head = attn_dim_head[-1])
        self.mid_block2 = resnet_block(mid_dim, mid_dim)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))):
            is_last = ind == (len(in_out) - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.ups.append(ModuleList([
                resnet_block(dim_out + dim_in, dim_out),
                resnet_block(dim_out + dim_in, dim_out),
                attn_klass(dim_out, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = resnet_block(init_dim * 2, init_dim)
        self.final_conv = nn.Conv2d(init_dim, self.out_dim, 1)
    """
    **downsample_factor**는 UNet의 다운샘플링 단계에서 해상도가 얼마나 줄어드는지 계산하는 속성입니다. 
    여기서는 UNet의 다운샘플링 횟수에 따라 해상도가 2의 제곱 비율로 줄어들므로, 2 ** (len(self.downs) - 1)로 계산합니다.
    """
    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)

    def forward(self, x, time, x_self_cond = None):
        # 입력 이미지의 해상도(x.shape[-2:], 즉 높이와 너비)가 UNet에서 정의된 다운샘플링 비율로 나누어떨어지는지 확인합니다. 나누어떨어지지 않으면 오류를 발생시킵니다.
        assert all([divisible_by(d, self.downsample_factor) for d in x.shape[-2:]]), f'your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet'

        # Self-Conditioning이 활성화된 경우, x_self_cond 입력을 기존 입력 x와 결합합니다. 만약 x_self_cond가 주어지지 않았다면, torch.zeros_like(x)로 초기화된 텐서를 사용합니다.
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        """
        초기 입력에 대해 7x7 컨볼루션을 수행하여 초기 특징 맵을 생성합니다.
        **r**에 원본 입력의 복사본을 저장해 나중에 skip connection으로 사용할 준비를 합니다.
        """
        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        
        """
        다운샘플링 과정에서 두 개의 ResNet 블록을 거친 후 어텐션 블록을 사용합니다. 각 단계에서 중간 결과를 h 리스트에 저장해, 이후 업샘플링에서 skip connection으로 사용합니다.
        **downsample**을 통해 해상도를 줄이며, 마지막 단계에서만 다운샘플링 대신 컨볼루션을 사용합니다. 
        """
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x) + x
            h.append(x)

            x = downsample(x)

        # UNet의 중앙부에서는 두 개의 ResNet 블록과 어텐션을 적용합니다. 여기서 어텐션은 중앙에서 더욱 복잡한 관계를 학습하게끔 돕습니다.
        x = self.mid_block1(x, t)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x) + x

            x = upsample(x)

        """
        초기 입력 r과 최종 업샘플링된 출력을 결합합니다.
        이를 다시 ResNet 블록에 통과시킨 후, 최종 컨볼루션을 통해 출력 이미지를 생성합니다.
        """
        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

"""
역할: 선형 스케줄을 사용하여 beta 값을 계산합니다. 베타 값은 타임스텝에 따라 선형적으로 증가하며, 이는 DDPM(Denoising Diffusion Probabilistic Models) 논문에서 제안된 방식입니다.
결과: 베타 값이 타임스텝이 진행됨에 따라 선형적으로 증가합니다.
사용 시나리오: 간단한 선형 노이즈 추가 방식을 사용할 때 사용됩니다.
"""
def linear_beta_schedule(timesteps):

    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

"""
역할: 코사인 스케줄을 사용하여 beta 값을 계산합니다. 논문 Improved Denoising Diffusion Probabilistic Models에서 제안된 방식입니다.
결과: 베타 값이 코사인 함수를 기반으로 계산되며, 노이즈의 추가가 더 자연스럽게 이루어지도록 합니다.
사용 시나리오: 고해상도 이미지 또는 더 복잡한 이미지 데이터를 처리할 때, 선형 스케줄보다 코사인 스케줄이 더 효율적일 수 있습니다.
"""
def cosine_beta_schedule(timesteps, s = 0.008):
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

"""
역할: 시그모이드 스케줄을 사용하여 beta 값을 계산합니다. 시그모이드 함수는 점진적으로 변화하는 값을 제공하여 선형 또는 코사인 방식보다 더 부드러운 노이즈 추가 과정을 제공합니다.
결과: 베타 값이 시그모이드 함수에 의해 부드럽게 변화합니다.
사용 시나리오: 고해상도 이미지(64x64 이상) 처리에 적합한 방식으로, 시그모이드 곡선을 따라 더 자연스러운 노이즈 추가가 이루어집니다.
"""
def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):

    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class GaussianDiffusion(Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps = 1000,
        sampling_timesteps = None,
        objective = 'pred_v',
        beta_schedule = 'sigmoid',
        schedule_fn_kwargs = dict(),
        ddim_sampling_eta = 0.,
        auto_normalize = True,
        offset_noise_strength = 0.,  # https://www.crosslabs.org/blog/diffusion-with-offset-noise
        min_snr_loss_weight = False, # https://arxiv.org/abs/2303.09556
        min_snr_gamma = 5,
        immiscible = False
    ):
        super().__init__()
        assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        assert not hasattr(model, 'random_or_learned_sinusoidal_cond') or not model.random_or_learned_sinusoidal_cond

        self.model = model

        self.channels = self.model.channels
        self.self_condition = self.model.self_condition

        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        assert isinstance(image_size, (tuple, list)) and len(image_size) == 2, 'image size must be a integer or a tuple/list of two integers'
        """
        image_size는 입력 이미지의 크기를 정의합니다.
        objective는 이 모델이 예측해야 하는 목표를 설정합니다. 여기서는 노이즈(pred_noise), 시작 이미지(pred_x0), 또는 v-파라미터화(pred_v) 중 하나를 선택합니다.
        """
        self.image_size = image_size

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        """
        노이즈 베타 값의 스케줄링 방식을 정의합니다. linear, cosine, sigmoid 중 하나를 선택할 수 있으며, 이는 베타 값이 각 타임스텝에 따라 어떻게 변화하는지를 나타냅니다.
        """
        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        """
        알파 값(alphas)은 베타 값과 관련이 있으며, 노이즈가 추가되는 정도를 나타냅니다.
        **alphas_cumprod**는 알파 값의 누적 곱으로, 확산 과정에서 각 타임스텝에서 얼마나 많은 원본 이미지 정보가 남아 있는지를 나타냅니다.

        """
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        """
        샘플링 과정에서 사용할 타임스텝 수를 설정합니다. 
        기본적으로 훈련된 타임스텝 수와 동일하게 설정되지만, 더 적은 타임스텝으로 DDIM(Deterministic Diffusion Implicit Models) 방식으로 샘플링할 수 있습니다.
        """
        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        """
        버퍼 등록: 모델에 필요한 상수 값을 버퍼로 등록합니다. 베타 값(betas), 알파 값(alphas_cumprod), 그리고 이전 알파 값(alphas_cumprod_prev)을 버퍼로 설정합니다.
        이 값들은 모델이 훈련 및 샘플링할 때 고정된 상수로 사용됩니다.
        """
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        """
        Posterior 분산 및 평균 계수는 확산 과정에서 현재 상태에서 이전 상태로 복원할 때 필요한 값들입니다.
        이 계산은 모델이 노이즈 제거 과정에서 필요한 분포를 정의하는 데 사용됩니다.
        """
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        """
        Posterior 분산 및 평균 계수는 확산 과정에서 현재 상태에서 이전 상태로 복원할 때 필요한 값들입니다.
        이 계산은 모델이 노이즈 제거 과정에서 필요한 분포를 정의하는 데 사용됩니다.
        """
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # immiscible diffusion

        self.immiscible = immiscible

        # offset noise strength - in blogpost, they claimed 0.1 was ideal

        self.offset_noise_strength = offset_noise_strength

        # derive loss weight
        # snr - signal noise ratio

        snr = alphas_cumprod / (1 - alphas_cumprod)

        # https://arxiv.org/abs/2303.09556

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        """
        SNR(Signal-to-Noise Ratio)을 계산하여 손실 함수의 가중치로 사용합니다. 
        이는 노이즈 제거가 효과적으로 이루어지도록 조정하는 데 중요한 역할을 합니다.
        """
        if objective == 'pred_noise':
            register_buffer('loss_weight', maybe_clipped_snr / snr)
        elif objective == 'pred_x0':
            register_buffer('loss_weight', maybe_clipped_snr)
        elif objective == 'pred_v':
            register_buffer('loss_weight', maybe_clipped_snr / (snr + 1))

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False
        """
        데이터를 [-1, 1] 범위로 자동으로 정규화하거나, 이를 비활성화할 수 있습니다.
        """
        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    @property
    def device(self):
        return self.betas.device

    """
    역방향 과정에서는 각 타임스텝 t에서 **현재 이미지 x_t**와 그 이미지에 추가된 노이즈 **ε**를 사용해 **원본 이미지 x_0**를 추정합니다.
    """
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    
    """
    이 함수는 **타임스텝 t에서의 상태 x_t와 원본 이미지 x0**를 사용하여 **노이즈 noise**를 예측합니다.
    """
    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )
    """
    이 함수는 Posterior 분포의 평균과 분산을 계산합니다. 이는 타임스텝 t에서 역방향 과정에서 현재 상태 x_t와 원본 이미지 x_start 간의 관계를 계산하는 데 사용됩니다.
    """
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    """
    이 함수는 모델의 출력을 바탕으로 노이즈를 예측하거나 이미지 시작 상태를 예측합니다. 목표에 따라 세 가지 방식 중 하나로 작동합니다:
    pred_noise: 모델이 노이즈를 예측.
    pred_x0: 모델이 원본 이미지를 직접 예측.
    pred_v: v-파라미터화를 통해 원본 이미지를 예측.
    재정의 옵션:
    clip_x_start가 True인 경우, 모델이 예측한 시작 이미지를 -1에서 1 사이로 클리핑.
    rederive_pred_noise가 True이면 클립 후 노이즈를 다시 예측.
    """
    def model_predictions(self, x, t, x_self_cond = None, clip_x_start = False, rederive_pred_noise = False):
        model_output = self.model(x, t, x_self_cond)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)
    """
    Posterior 분포의 평균 및 분산을 계산합니다.
    이 과정에서:
    model_predictions를 사용해 모델의 예측값을 가져오고, 클리핑 여부에 따라 결과를 정리합니다.
    Posterior는 q_posterior 함수를 사용하여 계산되며, 여기서 posterior_mean과 posterior_variance가 중요한 역할을 합니다.
    이 함수는 p-sample에서 사용되며, 역방향 확산 과정에서 필요한 중간값들을 제공합니다.
    """
    def p_mean_variance(self, x, t, x_self_cond = None, clip_denoised = True):
        preds = self.model_predictions(x, t, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    """
    이 함수는 역방향 확산 과정에서 **현재 상태 x_t**에서 **이전 상태 x_{t-1}**로 샘플링합니다.
    p_mean_variance를 통해 계산된 Posterior 평균 및 분산을 바탕으로 다음 샘플을 생성합니다.
    노이즈는 마지막 타임스텝(t == 0)에서는 추가되지 않습니다. 이는 마지막 타임스텝에서 복원된 이미지를 안정적으로 유지하기 위함입니다.
    """
    @torch.inference_mode() # 모델이 학습 모드가 아닌 추론 모드, 백워드 패스와 그레이디언트 계산을 비활성화하여 메모리 사용을 줄이고 성능을 향상
    def p_sample(self, x, t: int, x_self_cond = None):
        b, *_, device = *x.shape, self.device
        """
        **torch.full((b,), t)**는 b개의 타임스텝 **t**를 저장한 1차원 텐서를 생성합니다. 
        예를 들어 **b = 4**이고, **t = 5**인 경우 **[5, 5, 5, 5]**가 됩니다.
        """
        batched_times = torch.full((b,), t, device = device, dtype = torch.long)
        """
        model_mean: 모델이 예측한 평균값.
        model_log_variance: 모델이 예측한 로그 분산값. 이를 통해 분산을 계산할 수 있습니다.
        x_start: 모델이 예측한 원본 이미지 x_0(시작 상태).
        """
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, x_self_cond = x_self_cond, clip_denoised = True)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0 # 타임스텝 **t > 0**인 경우, 랜덤한 가우시안 노이즈를 생성, 다음 타임스텝의 이미지에 추가
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise # model_mean + (표준 편차 * noise)
        return pred_img, x_start

    """
    타임스텝이 감소하는 루프를 사용하여 이미지를 점진적으로 복원합니다.
    처음에는 랜덤한 노이즈에서 시작하여, 각 타임스텝마다 p_sample을 호출해 점진적으로 노이즈를 제거하고 이미지를 복원합니다.
    return_all_timesteps 옵션을 사용해 전체 타임스텝 동안의 중간 샘플을 반환할 수도 있습니다.
    마지막 결과는 self.unnormalize 함수를 통해 0에서 1 사이로 변환됩니다.
    """
    @torch.inference_mode()
    def p_sample_loop(self, shape, return_all_timesteps = False):
        batch, device = shape[0], self.device

        img = torch.randn(shape, device = device) # 랜덤 노이즈
        imgs = [img]

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps): # **self.num_timesteps**에서 0으로 역순으로 반복하는 루프
            self_cond = x_start if self.self_condition else None # 모델이 이전 타임스텝에서 복원한 이미지 정보를 다시 조건으로 사용
            img, x_start = self.p_sample(img, t, self_cond) # p_sample 함수는 **타임스텝 t에서 현재 이미지 img**와 조건 **self_cond**를 사용하여 다음 타임스텝의 이미지를 생성합니다. 이 과정에서 노이즈가 점진적으로 제거되며, 원본 이미지로 점차 다가갑니다.
            imgs.append(img)

        """
        최종 이미지 반환: return_all_timesteps 옵션이 설정되지 않으면, **최종 타임스텝에서의 이미지 img**만 반환합니다. 만약 **return_all_timesteps**가 **True**로 설정되었다면, 모든 타임스텝의 이미지를 반환합니다.
        self.unnormalize(ret): 마지막으로 반환된 이미지를 unnormalize합니다. 이는 보통 [-1, 1] 범위에서 [0, 1] 범위로 변환하는 과정입니다. 이 과정을 통해 샘플링된 이미지를 시각적으로 볼 수 있는 상태로 변환합니다.
        """
        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret) # 보통 [-1, 1] 범위에서 [0, 1] 범위로 변환하는 과정입니다. 이 과정을 통해 샘플링된 이미지를 시각적으로 볼 수 있는 상태로 변환
        return ret

    @torch.inference_mode()
    def ddim_sample(self, shape, return_all_timesteps = False):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start = True, rederive_pred_noise = True)

            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret
    """
    목적: Diffusion 모델을 이용해 샘플링을 수행합니다.
    batch_size는 샘플링할 이미지의 수, return_all_timesteps는 중간 타임스텝의 이미지들을 모두 반환할지 결정하는 옵션
    출력: 샘플링된 이미지(최종 타임스텝 또는 모든 타임스텝에 걸친 이미지).
    """
    @torch.inference_mode()
    def sample(self, batch_size = 16, return_all_timesteps = False):
        (h, w), channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, channels, h, w), return_all_timesteps = return_all_timesteps)

    """
     이미지 **x1**과 **x2**를 **타임스텝 t에서 보간(interpolate)**하여 새로운 이미지를 생성합니다. 이는 두 이미지 사이의 중간 상태를 표현하는 데 사용됩니다.
    """
    @torch.inference_mode()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    """
    목적: 노이즈와 원본 이미지 사이의 대응 관계를 정의하는 함수입니다. 이는 immiscible(혼합 불가능한) 설정이 켜져 있을 때 사용됩니다.
    동작: 원본 이미지 **x_start**와 노이즈의 거리를 계산하고, linear_sum_assignment 함수를 사용하여 두 개를 가장 잘 맞도록 정렬합니다.
    출력: 정렬된 노이즈 인덱스.
    """
    def noise_assignment(self, x_start, noise):
        x_start, noise = tuple(rearrange(t, 'b ... -> b (...)') for t in (x_start, noise))
        dist = torch.cdist(x_start, noise)
        _, assign = linear_sum_assignment(dist.cpu())
        return torch.from_numpy(assign).to(dist.device)

    """
    목적: **원본 이미지 x_start**에 대해 타임스텝 t에서의 노이즈가 추가된 이미지를 생성하는 함수입니다. 이는 forward process의 일부로, 순방향 과정에서 노이즈를 추가하는 역할을 합니다.
    입력: x_start는 원본 이미지, t는 타임스텝, noise는 추가할 노이즈입니다.
    동작:
    기본적으로 noise는 torch.randn_like(x_start)로 무작위로 생성됩니다.
    immiscible 설정이 활성화된 경우, 노이즈를 재배열합니다.
    이미지에 노이즈를 추가하여 타임스텝 t에서의 상태로 변환합니다.
    출력: 타임스텝 t에서의 노이즈가 추가된 이미지.
    """
    @autocast('cuda', enabled = False)
    def q_sample(self, x_start, t, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        if self.immiscible:
            assign = self.noise_assignment(x_start, noise)
            noise = noise[assign]

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
    """
    목적: 손실 함수를 계산하는 함수로, 모델이 현재 타임스텝 t에서 정확하게 노이즈 또는 원본 이미지를 예측했는지 평가합니다.
    동작:
    노이즈를 추가하여 타임스텝 t에서의 이미지를 생성합니다.
    self-conditioning을 사용할 경우, 이전 타임스텝에서 예측된 **x_start**를 조건부로 사용합니다.
    모델이 예측한 **model_out**을 실제 노이즈 **noise**나 원본 이미지 **x_start**와 비교하여 MSE 손실을 계산합니다.
    계산된 손실은 가중치가 적용되어 반환됩니다.
    """
    def p_losses(self, x_start, t, noise = None, offset_noise_strength = None):
        b, c, h, w = x_start.shape

        noise = default(noise, lambda: torch.randn_like(x_start))

        # offset noise - https://www.crosslabs.org/blog/diffusion-with-offset-noise

        offset_noise_strength = default(offset_noise_strength, self.offset_noise_strength)

        if offset_noise_strength > 0.:
            offset_noise = torch.randn(x_start.shape[:2], device = self.device)
            noise += offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1')

        # noise sample

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step

        model_out = self.model(x, t, x_self_cond)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()
    """
    목적: 모델의 forward pass로, 주어진 이미지 **img**에 대해 타임스텝 **t**를 랜덤하게 샘플링하여 손실을 계산하는 함수입니다.
    동작:
    입력 이미지 **img**의 크기와 장치를 확인합니다.
    **타임스텝 t**를 무작위로 선택합니다.
    이미지를 정규화한 후 손실 함수를 계산하여 반환합니다.
    """
    def forward(self, img, *args, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size[0] and w == img_size[1], f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = self.normalize(img)
        return self.p_losses(img, t, *args, **kwargs)

# dataset classes

class Dataset(Dataset):
    """
    folder: 이미지 파일들이 있는 폴더 경로입니다. 이 폴더에서 주어진 확장자(jpg, png, tiff 등)를 가진 이미지 파일들을 불러옵니다.
    image_size: 이미지를 리사이즈할 크기입니다.
    exts: 허용되는 이미지 확장자 목록입니다. 기본값은 jpg, jpeg, png, tiff입니다.
    augment_horizontal_flip: 이미지의 수평 반전 증강을 할지 여부를 나타내는 플래그입니다. True이면 이미지가 수평으로 무작위로 뒤집힙니다.
    convert_image_to: 이미지를 특정 모드(예: RGB, L, CMYK 등)로 변환할 수 있습니다. 변환 함수는 convert_image_to_fn에서 처리됩니다.
    **self.paths**는 주어진 폴더에서 확장자별로 이미지 파일 경로를 리스트로 저장합니다.
    glob 함수는 주어진 패턴(예: **/*.jpg)과 일치하는 파일들을 찾습니다.
    """
    def __init__(
        self,
        folder,
        image_size,
        exts = ['jpg', 'jpeg', 'png', 'tiff'],
        augment_horizontal_flip = False,
        convert_image_to = None
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        #  만약 **convert_image_to**가 주어졌다면, 해당 모드로 이미지를 변환하는 함수(convert_image_to_fn)를 설정합니다. 그렇지 않으면 **nn.Identity()**로 변환을 하지 않습니다.
        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()

        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])
    
    """
    데이터셋 크기를 반환합니다.
    """
    def __len__(self):
        return len(self.paths)

    """
    **index**에 해당하는 이미지 파일을 읽어옵니다.
    """
    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

# trainer class

class Trainer:
    """
    diffusion_model: 학습할 Diffusion 모델입니다.
    folder: 학습할 데이터가 저장된 폴더입니다.
    train_batch_size: 학습 시 사용되는 배치 크기입니다. 기본값은 16입니다.
    gradient_accumulate_every: 기울기 누적을 위한 설정입니다. 여러 배치에 걸쳐 기울기를 누적하여 업데이트를 줄이도록 할 수 있습니다.
    train_lr: 학습률(learning rate)입니다. 기본값은 1e-4입니다.
    train_num_steps: 전체 학습 스텝 수입니다.
    ema_update_every: EMA(Exponential Moving Average) 업데이트 주기입니다.
    save_and_sample_every: 주기적으로 모델을 저장하고 샘플을 생성하는 스텝 수입니다.
    calculate_fid: FID를 계산할지 여부입니다.
    max_grad_norm: 기울기 클리핑을 위한 최대 노름입니다.
    num_fid_samples: FID 계산에 사용할 샘플 개수입니다.
    """
    def __init__(
        self,
        diffusion_model,
        folder,
        *,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        augment_horizontal_flip = True,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 25,
        results_folder = './results',
        amp = False,
        mixed_precision_type = 'fp16',
        split_batches = True,
        convert_image_to = None,
        calculate_fid = True,
        inception_block_idx = 2048,
        max_grad_norm = 1.,
        num_fid_samples = 50000,
        save_best_and_latest_only = False
    ):
        super().__init__()

        # accelerator
        """
        **Accelerator**는 모델을 병렬 처리하거나 혼합 정밀도 학습을 위한 도구입니다. 여기서는 배치 분할과 혼합 정밀도를 설정합니다.
        """
        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = mixed_precision_type if amp else 'no'
        )
        # model

        self.model = diffusion_model
        self.channels = diffusion_model.channels
        is_ddim_sampling = diffusion_model.is_ddim_sampling

        # default convert_image_to depending on channels

        if not exists(convert_image_to):
            convert_image_to = {1: 'L', 3: 'RGB', 4: 'RGBA'}.get(self.channels)

        # sampling and training hyperparameters

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        assert (train_batch_size * gradient_accumulate_every) >= 16, f'your effective batch size (train_batch_size x gradient_accumulate_every) should be at least 16 or above'

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        self.max_grad_norm = max_grad_norm

        # dataset and dataloader
        """
        Dataset 클래스는 이미지를 불러와 전처리하는 역할을 합니다. 불러온 데이터를 DataLoader로 반복 사용되도록 설정합니다.
        """
        self.ds = Dataset(folder, self.image_size, augment_horizontal_flip = augment_horizontal_flip, convert_image_to = convert_image_to)

        assert len(self.ds) >= 100, 'you should have at least 100 images in your folder. at least 10k images recommended'

        dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = cpu_count())

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        # FID-score computation

        self.calculate_fid = calculate_fid and self.accelerator.is_main_process

        if self.calculate_fid:
            from denoising_diffusion_pytorch.fid_evaluation import FIDEvaluation

            if not is_ddim_sampling:
                self.accelerator.print(
                    "WARNING: Robust FID computation requires a lot of generated samples and can therefore be very time consuming."\
                    "Consider using DDIM sampling to save time."
                )
            self.fid_scorer = FIDEvaluation(
                batch_size=self.batch_size,
                dl=self.dl,
                sampler=self.ema.ema_model,
                channels=self.channels,
                accelerator=self.accelerator,
                stats_dir=results_folder,
                device=self.device,
                num_fid_samples=num_fid_samples,
                inception_block_idx=inception_block_idx
            )

        if save_best_and_latest_only:
            assert calculate_fid, "`calculate_fid` must be True to provide a means for model evaluation for `save_best_and_latest_only`."
            self.best_fid = 1e10 # infinite

        self.save_best_and_latest_only = save_best_and_latest_only
    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': __version__
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))
    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])
    def train(self):
        accelerator = self.accelerator
        device = accelerator.device
        """
        모델 학습을 관리하는 핵심 메서드입니다. 학습 루프를 돌면서 매 스텝마다 모델을 학습시키고, 주기적으로 샘플을 생성하며, FID 점수를 계산할 수 있습니다.
        """
        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:
                self.model.train()

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl).to(device)

                    with self.accelerator.autocast():
                        loss = self.model(data)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                pbar.set_description(f'loss: {total_loss:.4f}')

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and divisible_by(self.step, self.save_and_sample_every):
                        """
                        ema.ema_model.sample: 학습한 모델을 사용해 샘플을 생성합니다.
                        생성한 샘플 이미지를 **utils.save_image**를 사용하여 주기적으로 저장합니다.
                        """
                        self.ema.ema_model.eval()

                        with torch.inference_mode():
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))

                        all_images = torch.cat(all_images_list, dim = 0)

                        utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'), nrow = int(math.sqrt(self.num_samples)))

                        # whether to calculate fid

                        if self.calculate_fid:
                            fid_score = self.fid_scorer.fid_score()
                            accelerator.print(f'fid_score: {fid_score}')

                        if self.save_best_and_latest_only:
                            if self.best_fid > fid_score:
                                self.best_fid = fid_score
                                self.save("best")
                            self.save("latest")
                        else:
                            self.save(milestone)

                pbar.update(1)

        accelerator.print('training complete')