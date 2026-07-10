"""
SocrateX - Custom OCR library based on Transformer architecture.

This library was developed to facilitate training, inference,
and experimentation with SOCRATE models. Everything is modular and customizable.

Quick Start:
-----------------------------------
    import SocrateX as sx

    # 1. Build a custom model architecture using sx.Config:
    config = sx.Config(
        d_model=640,
        nhead=10,
        num_layers=12,
        dim_feedforward=2560,
        pool_height=4   # nn.AdaptiveMaxPool2d((pool_height, None))
    )
    tokenizer = sx.load_tokenizer("ocr_bpe_tokenizer.json")
    model = sx.init(config=config, tokenizer=tokenizer)

    # 2. Build a dataset (height and max_length go here, not in Config):
    train_set = model.make_dataset(images, labels, height=32, max_length=64)

    # 3. Train:
    model.fit(dataloader, optimizer, criterion, epochs=50)

    # 4. Predict (inference params go here):
    results = model.predict(
        image_paths=["document.jpg"],
        function="generate",
        temp=0.5,
        max_iter=64,
        penalty=1.15,
        top_k=5
    )
"""

from .configuration_socrate import SocrateConfig
from .config import Config
from .model import SOCRATE, cat, rat, mice, ResidualBlock, PositionalEncoding, SocratePool
from .dataset import Makeset, SmartBatchSampler, load_dataset
from .trainer import train, Trainer
from .inference import predict, generate, generate_fast, beam_search, extract_crops_from_image
from .tokenizer import init_tokenizer, SocrateXTokenizer
from .synthetic import generate_silly_training_set, generate_silly_testing_set

__all__ = [
    "Config",
    "SOCRATE",
    "SocrateConfig",
    "cat",
    "rat",
    "mice",
    "ResidualBlock",
    "PositionalEncoding",
    "SocratePool",
    "Makeset",
    "SmartBatchSampler",
    "train",
    "Trainer",
    "predict",
    "generate",
    "generate_fast",
    "beam_search",
    "init_tokenizer",
    "load_tokenizer",
    "SocrateXTokenizer",
    "generate_silly_training_set",
    "generate_silly_testing_set",
    "load_dataset",
]

def load_tokenizer(path="ocr_bpe_tokenizer.json"):
    """
    Alias to easily load a tokenizer from a JSON file.
    """
    return SocrateXTokenizer.from_file(path)

def load(model_type="cat", weights=None, tokenizer_path="ocr_bpe_tokenizer.json", device="cuda"):
    """
    Automatically loads the desired model and tokenizer.
    Returns (model, tokenizer).
    """
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    if model_type == "cat":
        model = cat(tokenizer=tokenizer, weights=weights if weights else cat.pretrained, device=device)
    elif model_type == "rat":
        model = rat(tokenizer=tokenizer, weights=weights if weights else rat.pretrained, device=device)
    elif model_type == "mice":
        model = mice(tokenizer=tokenizer, weights=weights if weights else mice.pretrained, device=device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    return model, tokenizer

def init(tokenizer=None, config=None, device="cuda"):
    """
    Initializes a SOCRATE model from scratch.

    Pass an sx.Config() object to fully control the architecture:
        config = sx.Config(d_model=256, nhead=4, num_layers=3, pool_height=4)
        model = sx.init(config=config, tokenizer=tokenizer)

    If config is None, defaults to the cat (158M) architecture.
    """
    if tokenizer is None:
        raise ValueError("You must provide a tokenizer (sx.init_tokenizer() or sx.load_tokenizer()).")

    if config is None:
        # Default: cat architecture
        config = Config()

    hf_config = SocrateConfig(
        d_model=config.d_model,
        max_len=config.max_len,
        nhead=config.nhead,
        dim_feedforward=config.dim_feedforward,
        activation=config.activation,
        norm_first=config.norm_first,
        num_layers=config.num_layers,
        vocab_size=tokenizer.get_vocab_size(),
        pad_id=tokenizer.token_to_id("<pad>"),
        bos_id=tokenizer.token_to_id("<bos>"),
        eos_id=tokenizer.token_to_id("<eos>"),
    )
    model = SOCRATE(hf_config, tokenizer=tokenizer, sx_config=config).to(device)
    return model
