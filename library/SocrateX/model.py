import math
import torch
import torch.nn as nn
try:
    from transformers import PreTrainedModel
    from .configuration_socrate import SocrateConfig
    HAS_TRANSFORMERS = True
except ImportError:
    class PreTrainedModel(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
    HAS_TRANSFORMERS = False

class PositionalEncoding(nn.Module):
    def __init__(self,d_model,max_len):
        super().__init__()
        pe = torch.zeros(max_len,d_model)

        position = torch.arange(0,max_len,dtype=torch.float32).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0,d_model,2,dtype=torch.float32   ) * (-1) * math.log(10000)/d_model
        )

        pe[:,::2] = torch.sin(div_term * position)
        pe[:,1::2] = torch.cos(div_term * position)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe",pe)
        
    def forward(self,x):
        return x + self.pe[:,:x.size(1),:]

class ResidualBlock(nn.Module):
    def __init__(self,in_,out_,stride_=1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_,out_,kernel_size=3,stride=stride_,padding=1),
            nn.BatchNorm2d(out_),
            nn.ReLU(),
        )

        self.relu = nn.ReLU()
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_,out_,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_),
        )

        self.id = nn.Identity()
        if in_ != out_ or stride_ != 1:
            self.id = nn.Sequential(
            nn.Conv2d(in_,out_,kernel_size=1,stride=stride_),
            nn.BatchNorm2d(out_),
        )
    def forward(self,x):
        identity = self.id(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + identity
        x = self.relu(x)
        return x 

class SocratePool(nn.Module):
    """
    SocratePool ensures that the feature map's height is always compressed to a fixed dimension,
    while the width dynamically adapts based on the input sequence.
    target_height controls the vertical resolution after pooling.
    """
    def __init__(self, target_height=4):
        super().__init__()
        self.target_height = target_height
        self.pool = nn.AdaptiveMaxPool2d((target_height, None))
        
    def forward(self, x):
        return self.pool(x)

class SOCRATE(PreTrainedModel):
    if HAS_TRANSFORMERS:
        config_class = SocrateConfig
    else:
        config_class = None

    def __init__(self, config, tokenizer=None, sx_config=None):
        super().__init__(config)
        self.tokenizer = tokenizer
        # Store the unified sx.Config if provided (used for inference defaults)
        self.sx_config = sx_config
        
        # If we have an active tokenizer (during inference/local training), use its values
        if tokenizer is not None:
            self.vocab_size = tokenizer.get_vocab_size()
            self.pad_id = tokenizer.token_to_id("<pad>")
            self.bos_id = tokenizer.token_to_id("<bos>")
            self.eos_id = tokenizer.token_to_id("<eos>")
        else:
            # Fallback to config (when loaded from HuggingFace without an initial tokenizer passed)
            self.vocab_size = config.vocab_size
            self.pad_id = config.pad_id
            self.bos_id = config.bos_id
            self.eos_id = config.eos_id
            
        self.d_model = config.d_model
        
        # Resolve pool_height: prefer sx_config, then HF config, then default 4
        _pool_h = getattr(sx_config, "pool_height", None) or getattr(config, "pool_height", 4)
        
        self.convolution = nn.Sequential(
            ResidualBlock(3, 32, 2),
            ResidualBlock(32, 64),
            ResidualBlock(64, 128, 2),
            ResidualBlock(128, 256),
            ResidualBlock(256, self.d_model, 2),
        )
        self.project = nn.Sequential(
            nn.Linear(self.d_model * _pool_h, self.d_model),
            nn.LayerNorm(self.d_model),
        )
        self.pool = SocratePool(target_height=_pool_h)
        self.pe = PositionalEncoding(self.d_model, config.max_len)
        self.norm_image = nn.LayerNorm(self.d_model)
        self.embedding = nn.Embedding(self.vocab_size, self.d_model, padding_idx=self.pad_id)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=config.nhead, dim_feedforward=config.dim_feedforward, 
            batch_first=True, norm_first=config.norm_first, activation=config.activation
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model, nhead=config.nhead, dim_feedforward=config.dim_feedforward, 
            batch_first=True, norm_first=config.norm_first, activation=config.activation
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.num_layers)

        self.output = nn.Linear(self.d_model, self.vocab_size)
        self.last_norm = nn.LayerNorm(self.d_model) 

    def encode(self, image):
        """Separate encode method (custom requirement)"""
        image = self.convolution(image)
        image = self.pool(image)
        B, C, H, W = image.shape
        image = image.permute(0, 3, 1, 2).contiguous()
        image = image.view(B, W, C * H)
        image = self.project(image)
        image = self.pe(image)
        image = self.encoder(image)
        return image

    def decode(self, memory_image, text):
        """Separate decode method (custom requirement)"""
        tgt_key_padding_mask = (text == self.pad_id)
        tgt_mask = (nn.Transformer.generate_square_subsequent_mask(text.size(1), device=text.device) != 0.0)
        
        text_emb = self.embedding(text)
        text_emb = self.pe(text_emb * math.sqrt(self.d_model))
        
        output = self.decoder(text_emb, memory_image, tgt_key_padding_mask=tgt_key_padding_mask, tgt_mask=tgt_mask)
        output = self.last_norm(output)
        return self.output(output)

    def forward(self, image, text):
        memory_image = self.encode(image)
        return self.decode(memory_image, text)

    # ==========================================
    # High-Level Methods (Keras-like)
    # ==========================================
    
    def fit(self, dataloader, optimizer, criterion, scheduler=None, scaler=None, device="cuda", best_loss=float('inf'), epochs=1, save_dir=".", save_name="socrate", save_interval=100):
        """
        Trains the model for the given number of epochs on the given dataloader.
        Returns the best_loss.
        """
        from .trainer import Trainer
        trainer = Trainer(self, optimizer, scheduler, criterion, device, scaler, save_dir, save_name, save_interval)
        for e in range(1, epochs + 1):
            best_loss = trainer.train_epoch(dataloader, best_loss, epoch_num=e, total_epochs=epochs)
        return best_loss

    def predict(self, image_paths, tokenizer=None, wpb=16, function="generate", doctr_model=None,
                bos_id=None, eos_id=None, device="cuda",
                # generate() params
                temp=0.5, max_iter=64, penalty=1.15, top_k=5,
                # generate_fast() params
                fast_max_iter=32,
                # beam_search() params
                beam_width=4, beam_max_iter=64):
        """
        Performs end-to-end inference on images.

        Args:
            image_paths (str | List[str]): Path(s) to input images.
            tokenizer: Optional tokenizer override.
            wpb (int): Words per batch. Default: 16.
            function (str): 'generate', 'generate_fast', 'beam_search', or a custom callable.
            doctr_model: Pre-loaded doctr detection model (avoids reloading).
            bos_id (int): Override BOS token ID.
            eos_id (int): Override EOS token ID.
            device (str): Target device. Default: 'cuda'.
            temp (float): Temperature for generate(). Default: 0.5.
            max_iter (int): Max tokens for generate(). Default: 64.
            penalty (float): Repetition penalty for generate(). Default: 1.15.
            top_k (int): Top-k candidates per step in generate(). Default: 5.
            fast_max_iter (int): Max tokens for generate_fast(). Default: 32.
            beam_width (int): Number of beams for beam_search(). Default: 4.
            beam_max_iter (int): Max tokens per beam for beam_search(). Default: 64.
        """
        from .inference import predict as inf_predict
        tk = tokenizer if tokenizer is not None else self.tokenizer

        return inf_predict(
            self, tk, image_paths, wpb, function, doctr_model, bos_id, eos_id, device,
            temp=temp, max_iter=max_iter, penalty=penalty, top_k=top_k,
            fast_max_iter=fast_max_iter,
            beam_width=beam_width, beam_max_iter=beam_max_iter
        )

    def load_parameters(self, path, strict=False):
        """
        Loads state dict from path with strict fallback.
        Returns the best_loss if it was saved, otherwise float('inf').
        """
        import os
        import torch
        if not os.path.exists(path):
            print(f"Warning: The weights file {path} was not found.")
            return float('inf')
            
        try:
            checkpoint = torch.load(path, map_location=next(self.parameters()).device, weights_only=False)
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
                best_loss = checkpoint.get("best_loss", float('inf'))
            else:
                state_dict = checkpoint
                best_loss = float('inf')
                
            self.load_state_dict(state_dict, strict=strict)
            return best_loss
        except Exception as e:
            print(f"Warning: Could not load parameters completely due to: {e}")
            return float('inf')

    def load(self, path, strict=False):
        """
        Alias for load_parameters.
        """
        return self.load_parameters(path, strict=strict)

    def make_dataset(self, images, labels=None, transform=None, height=None, max_length=None):
        """
        Creates a Makeset object for this model.
        height and max_length default to values from sx.Config if provided during init,
        otherwise fall back to 32 and 64 respectively.
        """
        from .dataset import Makeset
        sx_cfg = getattr(self, "sx_config", None)
        _height     = height     if height     is not None else (sx_cfg.height     if sx_cfg else 32)
        _max_length = max_length if max_length is not None else (sx_cfg.max_length if sx_cfg else 64)
        return Makeset(
            images=images,
            labels=labels,
            transform=transform,
            tokenizer=self.tokenizer,
            pad_id=self.pad_id,
            bos_id=self.bos_id,
            eos_id=self.eos_id,
            height=_height,
            max_length=_max_length
        )

    def freeze_encoder(self):
        """
        Freezes the encoder weights (CNN + TransformerEncoder).
        Useful if you want to fine-tune only the decoder.
        """
        for param in self.convolution.parameters():
            param.requires_grad = False
        for param in self.project.parameters():
            param.requires_grad = False
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("Encoder (CNN + TransformerEncoder) has been frozen.")

    def unfreeze_encoder(self):
        """
        Unfreezes the encoder weights.
        """
        for param in self.convolution.parameters():
            param.requires_grad = True
        for param in self.project.parameters():
            param.requires_grad = True
        for param in self.encoder.parameters():
            param.requires_grad = True
        print("Encoder has been unfrozen.")
        
    def summary(self):
        """
        Prints a summary of the model's parameters.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"=== SOCRATE Model Summary ===")
        print(f"Total parameters: {total_params / 1e6:.2f} M")
        print(f"Trainable parameters: {trainable_params / 1e6:.2f} M")
        print(f"d_model: {self.d_model}")
        print("=============================")

# Factory functions for models

def cat(tokenizer, weights=None, device="cuda"):
    """
    SOCRATE Cat - The original full-sized model.
    """
    if HAS_TRANSFORMERS:
        config = SocrateConfig(
            d_model=640,
            max_len=512,
            nhead=10,
            dim_feedforward=2560,
            activation="gelu",
            norm_first=True,
            num_layers=12,
            vocab_size=tokenizer.get_vocab_size() if tokenizer else 1000,
            pad_id=tokenizer.token_to_id("<pad>") if tokenizer else 0,
            bos_id=tokenizer.token_to_id("<bos>") if tokenizer else 1,
            eos_id=tokenizer.token_to_id("<eos>") if tokenizer else 2,
        )
    else:
        # Fallback dummy config if transformers is not installed
        class DummyConfig: pass
        config = DummyConfig()
        config.d_model = 640
        config.max_len = 512
        config.nhead = 10
        config.dim_feedforward = 2560
        config.activation = "gelu"
        config.norm_first = True
        config.num_layers = 12

    model = SOCRATE(config, tokenizer=tokenizer).to(device)
    
    if weights:
        if weights.startswith("http://") or weights.startswith("https://"):
            import torch.hub
            checkpoint = torch.hub.load_state_dict_from_url(weights, map_location=device)
        else:
            import os
            if os.path.exists(weights):
                checkpoint = torch.load(weights, map_location=device, weights_only=False)
            else:
                print(f"Warning: The weights file {weights} was not found locally.")
                checkpoint = None

        if checkpoint is not None:
            if "model" in checkpoint:
                model.load_state_dict(checkpoint["model"])
            else:
                model.load_state_dict(checkpoint)
            
    return model

cat.pretrained = "best_socrate_1.3.pt"

def rat(tokenizer, weights=None, device="cuda"):
    """
    SOCRATE Rat - The medium-sized variant.
    """
    if HAS_TRANSFORMERS:
        config = SocrateConfig(
            d_model=512,
            max_len=512,
            nhead=8,
            dim_feedforward=2048,
            activation="gelu",
            norm_first=True,
            num_layers=8,
            vocab_size=tokenizer.get_vocab_size() if tokenizer else 1000,
            pad_id=tokenizer.token_to_id("<pad>") if tokenizer else 0,
            bos_id=tokenizer.token_to_id("<bos>") if tokenizer else 1,
            eos_id=tokenizer.token_to_id("<eos>") if tokenizer else 2,
        )
    else:
        class DummyConfig: pass
        config = DummyConfig()
        config.d_model = 512
        config.max_len = 512
        config.nhead = 8
        config.dim_feedforward = 2048
        config.activation = "gelu"
        config.norm_first = True
        config.num_layers = 8

    model = SOCRATE(config, tokenizer=tokenizer).to(device)
    
    if weights:
        if weights.startswith("http://") or weights.startswith("https://"):
            import torch.hub
            checkpoint = torch.hub.load_state_dict_from_url(weights, map_location=device)
        else:
            import os
            if os.path.exists(weights):
                checkpoint = torch.load(weights, map_location=device, weights_only=False)
            else:
                print(f"Warning: The weights file {weights} was not found locally.")
                checkpoint = None

        if checkpoint is not None:
            if "model" in checkpoint:
                model.load_state_dict(checkpoint["model"])
            else:
                model.load_state_dict(checkpoint)
            
    return model

rat.pretrained = None

def mice(tokenizer, weights=None, device="cuda"):
    """
    SOCRATE Mice - The tiny variant for performance and edge devices.
    """
    if HAS_TRANSFORMERS:
        config = SocrateConfig(
            d_model=256,
            max_len=512,
            nhead=4,
            dim_feedforward=1024,
            activation="gelu",
            norm_first=True,
            num_layers=4,
            vocab_size=tokenizer.get_vocab_size() if tokenizer else 1000,
            pad_id=tokenizer.token_to_id("<pad>") if tokenizer else 0,
            bos_id=tokenizer.token_to_id("<bos>") if tokenizer else 1,
            eos_id=tokenizer.token_to_id("<eos>") if tokenizer else 2,
        )
    else:
        class DummyConfig: pass
        config = DummyConfig()
        config.d_model = 256
        config.max_len = 512
        config.nhead = 4
        config.dim_feedforward = 1024
        config.activation = "gelu"
        config.norm_first = True
        config.num_layers = 4

    model = SOCRATE(config, tokenizer=tokenizer).to(device)
    
    if weights:
        if weights.startswith("http://") or weights.startswith("https://"):
            import torch.hub
            checkpoint = torch.hub.load_state_dict_from_url(weights, map_location=device)
        else:
            import os
            if os.path.exists(weights):
                checkpoint = torch.load(weights, map_location=device, weights_only=False)
            else:
                print(f"Warning: The weights file {weights} was not found locally.")
                checkpoint = None

        if checkpoint is not None:
            if "model" in checkpoint:
                model.load_state_dict(checkpoint["model"])
            else:
                model.load_state_dict(checkpoint)
            
    return model

mice.pretrained = None
