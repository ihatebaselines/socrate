from transformers import PretrainedConfig

class SocrateConfig(PretrainedConfig):
    model_type = "socrate"

    def __init__(
        self,
        d_model=640,
        max_len=512,
        nhead=10,
        dim_feedforward=2560,
        activation="gelu",
        norm_first=True,
        num_layers=12,
        vocab_size=1000,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        **kwargs
    ):
        self.d_model = d_model
        self.max_len = max_len
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.activation = activation
        self.norm_first = norm_first
        self.num_layers = num_layers
        
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        
        super().__init__(**kwargs)
