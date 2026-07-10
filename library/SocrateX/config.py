"""
sx.Config — Model architecture configuration for SocrateX.
Contains only Transformer + SocratePool parameters.
"""

class Config:
    """
    Unified architecture config for SOCRATE.
    Covers the Transformer layers and SocratePool.
    
    Pass to sx.init() to build a fully custom model:
        config = sx.Config(d_model=256, nhead=4, num_layers=3)
        model = sx.init(config=config, tokenizer=tokenizer)
    """

    def __init__(
        self,
        # ─── Transformer ────────────────────────────────────────────
        d_model: int = 640,
        max_len: int = 512,
        nhead: int = 10,
        dim_feedforward: int = 2560,
        activation: str = "gelu",
        norm_first: bool = True,
        num_layers: int = 12,

        # ─── SocratePool ────────────────────────────────────────────
        pool_height: int = 4,   # target_height in nn.AdaptiveMaxPool2d((pool_height, None))
    ):
        self.d_model = d_model
        self.max_len = max_len
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.activation = activation
        self.norm_first = norm_first
        self.num_layers = num_layers
        self.pool_height = pool_height

    def __repr__(self):
        return (
            f"sx.Config(\n"
            f"  d_model={self.d_model}, nhead={self.nhead}, num_layers={self.num_layers},\n"
            f"  dim_feedforward={self.dim_feedforward}, activation='{self.activation}',\n"
            f"  norm_first={self.norm_first}, max_len={self.max_len},\n"
            f"  pool_height={self.pool_height}  # AdaptiveMaxPool2d((pool_height, None))\n"
            f")"
        )
