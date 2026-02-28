# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Qwen3 model configuration"""

from typing import Optional

from ...configuration_utils import PreTrainedConfig, layer_type_validation
from ...modeling_rope_utils import RopeParameters #Qwen 系列使用 RoPE (旋转位置编码)。这里专门引入了一个辅助类来管理 RoPE 的复杂参数（比如 theta 值，缩放因子等）
from ...utils import logging


logger = logging.get_logger(__name__)


class Qwen3Config(PreTrainedConfig):

    r"""
        这是用于存储 [`Qwen3Model`] 配置的配置类。它用于根据指定的参数实例化一个 Qwen3 模型，
        从而定义模型的架构。使用默认参数实例化配置将生成与 Qwen3-8B 类似的配置
        [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)。

        配置对象继承自 [`PreTrainedConfig`]，可用于控制模型输出。
        阅读 [`PreTrainedConfig`] 的文档以获取更多信息.

        参数 (Args):
            vocab_size (`int`, *可选*, 默认为 151936):
                Qwen3 模型的词表大小。定义了在调用 [`Qwen3Model`] 时传递的 `inputs_ids` 
                可以表示的不同 token（词元）的数量。

            hidden_size (`int`, *可选*, 默认为 4096):
                隐藏状态表示的维度（即每一层神经元的宽度）。

            intermediate_size (`int`, *可选*, 默认为 22016):
                MLP（多层感知机/前馈网络）表示的维度。通常比 hidden_size 大很多。

            num_hidden_layers (`int`, *可选*, 默认为 32):
                Transformer 编码器中的隐藏层数量（即有多少层楼）。

            num_attention_heads (`int`, *可选*, 默认为 32):
                Transformer 编码器中每个注意力层的注意力头数量。

            num_key_value_heads (`int`, *可选*, 默认为 32):
                这是用于实现分组查询注意力 (GQA) 的 Key-Value 头数量。
                - 如果 `num_key_value_heads = num_attention_heads`，模型将使用多头注意力 (MHA)。
                - 如果 `num_key_value_heads = 1`，模型将使用多查询注意力 (MQA)。
                - 否则使用 GQA。
                当将多头 (Multi-head) 权重转换为 GQA 权重时，每个组的 Key 和 Value 头
                应通过对该组内的所有原始头进行平均池化 (meanpooling) 来构造。
                更多详情，请查看 [这篇论文](https://huggingface.co/papers/2305.13245)。
                如果未指定，默认为 `32`。

            head_dim (`int`, *可选*, 默认为 128):
                每个注意力头的维度大小。

            hidden_act (`str` 或 `function`, *可选*, 默认为 `"silu"`):
                解码器中的非线性激活函数（可以是函数对象或字符串名称，如 "silu", "gelu"）。

            max_position_embeddings (`int`, *可选*, 默认为 32768):
                该模型可能使用的最大序列长度（即上下文窗口大小，32k）。

            initializer_range (`float`, *可选*, 默认为 0.02):
                用于初始化所有权重矩阵的截断正态初始化器 (truncated_normal_initializer) 的标准差。

            rms_norm_eps (`float`, *可选*, 默认为 1e-06):
                RMS 归一化层 (RMS Normalization) 使用的 epsilon 值（防止除以零的微小数值）。

            use_cache (`bool`, *可选*, 默认为 `True`):
                模型是否应返回最后的 Key/Values 注意力状态（并非所有模型都使用）。
                仅当 `config.is_decoder=True` 时相关。这是为了加速生成过程。

            tie_word_embeddings (`bool`, *可选*, 默认为 `False`):
                模型的输入词嵌入 (Input Embeddings) 和输出词嵌入 (Output Embeddings) 是否共享权重。

            rope_parameters (`RopeParameters`, *可选*):
                包含 RoPE (旋转位置编码) 配置参数的字典。
                该字典应包含 `rope_theta` 的值，以及可选的缩放参数（如果你想使用比 
                `max_position_embeddings` 更长的序列时需要用到）。

            attention_bias (`bool`, 默认为 `False`, *可选*, 默认为 `False`):
                是否在自注意力机制中的 Query、Key、Value 和 Output 投影层中使用偏置项 (Bias)。

            use_sliding_window (`bool`, *可选*, 默认为 `False`):
                是否启用滑动窗口注意力 (Sliding Window Attention)。

            sliding_window (`int`, *可选*, 默认为 4096):
                滑动窗口注意力 (SWA) 的窗口大小。如果未指定，默认为 `4096`。

            max_window_layers (`int`, *可选*, 默认为 28):
                使用全注意力 (Full Attention) 的层数。
                前 `max_window_layers` 层将使用全注意力，而之后的任何附加层将使用 SWA（滑动窗口注意力）。
                (注：这是 Qwen3 特有的混合注意力机制)。

            layer_types (`list`, *可选*):
                每一层的具体注意力模式列表（例如 ["full_attention", ..., "sliding_attention"]）。

            attention_dropout (`float`, *可选*, 默认为 0.0):
                注意力概率的 Dropout 比率（用于防止过拟合，通常推理时为 0）。

    ```python
    >>> from transformers import Qwen3Model, Qwen3Config

    >>> # Initializing a Qwen3 style configuration
    >>> configuration = Qwen3Config()

    >>> # Initializing a model from the Qwen3-8B style configuration
    >>> model = Qwen3Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "qwen3"
    #在模型做推理（Inference，即生成文本）的时候，如果需要保存模型的输出结果，请忽略掉 past_key_values （kv cache）这个东西。
    keys_to_ignore_at_inference = ["past_key_values"]

    # Default tensor parallel plan for base model `Qwen3`
    #多显卡计算的数据切分方案 
    # Tensor Parallelism
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }

    #流水线并行 (Pipeline Parallelism, PP)
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    #Qwen3Config 类的初始化函数
    def __init__(
        self,
        vocab_size: Optional[int] = 151936,
        hidden_size: Optional[int] = 4096,#这句代码意味着你可以传一个整数给 hidden_size。如果你没传，它就自动等于 4096。
        intermediate_size: Optional[int] = 22016,
        num_hidden_layers: Optional[int] = 32,
        num_attention_heads: Optional[int] = 32,
        num_key_value_heads: Optional[int] = 32,
        head_dim: Optional[int] = 128,
        hidden_act: Optional[str] = "silu",
        max_position_embeddings: Optional[int] = 32768,
        initializer_range: Optional[float] = 0.02,
        rms_norm_eps: Optional[int] = 1e-6,
        use_cache: Optional[bool] = True,
        tie_word_embeddings: Optional[bool] = False,
        rope_parameters: Optional[RopeParameters | dict[str, RopeParameters]] = None,
        attention_bias: Optional[bool] = False,
        use_sliding_window: Optional[bool] = False,
        sliding_window: Optional[int] = 4096,
        max_window_layers: Optional[int] = 28,
        layer_types: Optional[list[str]] = None,
        attention_dropout: Optional[float] = 0.0,
        **kwargs,                            #负责接收所有未被明确定义的命名参数。
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
      

      # 目的：确保“开关”与“数值”逻辑自洽，防止数据矛盾。
      # 解释：如果用户显式关闭了滑动窗口功能 (use_sliding_window=False)，
      # 那么无论传入的窗口大小 (sliding_window) 是多少，都强制将其重置为 None。
      # 这避免了“功能已关闭，却残留着窗口参数”导致的潜在 Bug。

        self.sliding_window = sliding_window if self.use_sliding_window else None
        self.max_window_layers = max_window_layers

      # 目的：兼容传统的 Multi-Head Attention (MHA) 架构。
      # 解释：用户如果没传 num_key_value_heads (KV头数)，说明他可能只想要普通的注意力机制。
      #       此时我们将 KV 头数设为与 Attention 头数一致，模型回退到标准 MHA 模式。
      #       只有当用户显式指定较小的 KV 头数时，才会激活 GQA (分组查询注意力) 以节省显存。
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        self.layer_types = layer_types



        # 目的：实现 Qwen3 特有的“全局-局部”混合注意力策略。
        # 解释：通过列表推导式，自动为 32 层网络生成每一层的类型描述：
        #       1. 底层 (0 ~ 27层): 使用 "full_attention"，确保模型能理解长文的全局依赖。
        #       2. 顶层 (28 ~ 31层): 切换为 "sliding_attention"，只关注局部上下文，减少计算量。
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention"
                # 如果开启了滑动窗口 且 当前层数超过了分界线 (max_window_layers)，则切为滑动窗口模式
                if self.sliding_window is not None and i >= self.max_window_layers
                # 否则，默认保持全注意力模式
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        layer_type_validation(self.layer_types, self.num_hidden_layers)

        self.rope_parameters = rope_parameters

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ["Qwen3Config"]
