import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from torch.utils.data import DataLoader
from .dataset import Makeset

def generate(model, image, bos_id, eos_id, device="cuda", temp=0.5, max_iter=64, penalty=1.15, top_k=5):
    """
    Prediction function using greedy search / top-k sampling with repetition penalty.
    
    Args:
        model: SOCRATE model instance.
        image (Tensor): Single image tensor [1, C, H, W].
        bos_id (int): Begin-of-sequence token ID.
        eos_id (int): End-of-sequence token ID.
        device (str): Target device. Default: "cuda".
        temp (float): Temperature for sampling. Lower = more greedy. Default: 0.5.
        max_iter (int): Max number of tokens to generate. Default: 64.
        penalty (float): Repetition penalty applied to already-seen tokens. Default: 1.15.
        top_k (int): Number of top candidates to sample from at each step. Default: 5.
    """
    model.eval()
    current_text = [bos_id]
    generated = []
    already_seen = set()
    
    with torch.inference_mode():
        memory_image = model.encode(image)
        
        for i in range(max_iter):
            x = torch.tensor([current_text], dtype=torch.long).to(device)
            output = model.decode(memory_image, x)
            output = output[:, -1, :]
            
            for token_id in already_seen:
                if output[0, token_id] < 0:
                    output[0, token_id] *= penalty
                else:
                    output[0, token_id] /= penalty
                    
            output = output / temp
            topk_vals, topk_idx = torch.topk(output, top_k, dim=-1)
            probs = F.softmax(topk_vals, dim=-1)
            idx = torch.multinomial(probs, 1)

            idx = topk_idx.gather(-1, idx).item()
            if idx == eos_id:
                break

            generated.append(idx)
            current_text.append(idx)
            already_seen.add(idx)

    return generated

def generate_fast(model, image, bos_id, eos_id, device="cuda", max_iter=32):
    """
    Super-fast prediction using only argmax (no sampling).

    Args:
        model: SOCRATE model instance.
        image (Tensor): Single image tensor [1, C, H, W].
        bos_id (int): Begin-of-sequence token ID.
        eos_id (int): End-of-sequence token ID.
        device (str): Target device. Default: "cuda".
        max_iter (int): Max number of tokens to generate. Default: 32.
    """
    model.eval()
    current_text = [bos_id]
    generated = []
    
    with torch.inference_mode():
        memory_image = model.encode(image)
        for _ in range(max_iter):
            x = torch.tensor([current_text], dtype=torch.long, device=device)
            output = model.decode(memory_image, x)
            logits = output[:, -1, :]
            idx = logits.argmax(dim=-1).item()
            
            if idx == eos_id:
                break
                
            generated.append(idx)
            current_text.append(idx)
            
    return generated

def beam_search(model, image, bos_id, eos_id, device="cuda", beam_width=4, max_iter=64):
    """
    Beam search decoding.

    Args:
        model: SOCRATE model instance.
        image (Tensor): Single image tensor [1, C, H, W].
        bos_id (int): Begin-of-sequence token ID.
        eos_id (int): End-of-sequence token ID.
        device (str): Target device. Default: "cuda".
        beam_width (int): Number of beams. Default: 4.
        max_iter (int): Max tokens per beam. Default: 64.
    
    Note: Full beam search is coming soon. Currently uses generate_fast as a fallback.
    """
    print("WARNING: Beam search is not fully implemented yet. Using generate_fast as a fallback.")
    return generate_fast(model, image, bos_id, eos_id, device, max_iter=max_iter)

def extract_crops_from_image(image_path, doctr_model=None):
    """
    Extracts words (crops) using doctr and sorts them
    correctly from top-to-bottom and left-to-right.
    """
    if doctr_model is None:
        from doctr.models import detection_predictor
        doctr_model = detection_predictor(arch="db_resnet50", pretrained=True)
        
    from doctr.io import DocumentFile
    
    doc = DocumentFile.from_images(image_path)
    result = doctr_model(doc)
    
    boxes = result[0]["words"]
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
        
    H, W = image.shape[:2]

    # Sort by lines a la SOCRATE
    boxes_info = []
    for b in boxes:
        xmin, ymin, xmax, ymax, score = b
        cy = (ymin + ymax) / 2.0
        h = ymax - ymin
        boxes_info.append({'box': b, 'cy': cy, 'h': h, 'x': xmin})

    boxes_info.sort(key=lambda item: item['cy'])

    lines = []
    current_line = []

    for b in boxes_info:
        if not current_line:
            current_line.append(b)
        else:
            tolerance = current_line[0]['h'] * 0.5
            if abs(b['cy'] - current_line[0]['cy']) < tolerance:
                current_line.append(b)
            else:
                lines.append(current_line)
                current_line = [b]
    if current_line:
        lines.append(current_line)

    sorted_boxes = []
    for line in lines:
        line.sort(key=lambda item: item['x'])
        for item in line:
            sorted_boxes.append(item['box'])

    crops = []
    for b in sorted_boxes:
        xmin, ymin, xmax, ymax, score = b
        x1 = int(xmin * W)
        y1 = int(ymin * H)
        x2 = int(xmax * W)
        y2 = int(ymax * H)

        crop = image[y1:y2, x1:x2]
        h, w = crop.shape[:2]

        if h == 0 or w == 0:
            continue
            
        crops.append(crop)
        
    return crops

def predict(model, tokenizer, image_paths, wpb=16, function="generate_fast", doctr_model=None, bos_id=None, eos_id=None, device="cuda",
            # generate() params
            temp=None, max_iter=None, penalty=None, top_k=None,
            # generate_fast() params
            fast_max_iter=None,
            # beam_search() params
            beam_width=None, beam_max_iter=None):
    """
    The main prediction function of the library. 
    Takes images and returns the text read from them.
    
    Inference parameters (temp, max_iter, penalty, top_k, fast_max_iter, beam_width, beam_max_iter)
    can be set here directly, OR they will be read from model.sx_config if you created the model via sx.init(config=...).
    
    Args:
        model: SOCRATE model instance.
        tokenizer: SocrateXTokenizer instance.
        image_paths (str | List[str]): Path(s) to the image(s).
        wpb (int): Words per batch. Default: 16.
        function (str | callable): 'generate', 'generate_fast', 'beam_search', or a custom callable.
        doctr_model: Pre-loaded doctr model (avoids re-loading on each call).
        bos_id (int): Override BOS token ID.
        eos_id (int): Override EOS token ID.
        device (str): Target device. Default: "cuda".
        temp (float): Temperature for generate(). Default: from config or 0.5.
        max_iter (int): Max tokens for generate(). Default: from config or 64.
        penalty (float): Repetition penalty for generate(). Default: from config or 1.15.
        top_k (int): Top-k for generate(). Default: from config or 5.
        fast_max_iter (int): Max tokens for generate_fast(). Default: from config or 32.
        beam_width (int): Number of beams for beam_search(). Default: from config or 4.
        beam_max_iter (int): Max tokens for beam_search(). Default: from config or 64.
    """
    model.eval()

    # Pull inference defaults from sx_config if they were set
    sx_cfg = getattr(model, "sx_config", None)

    _temp        = temp        if temp        is not None else (sx_cfg.temp        if sx_cfg else 0.5)
    _max_iter    = max_iter    if max_iter    is not None else (sx_cfg.max_iter    if sx_cfg else 64)
    _penalty     = penalty     if penalty     is not None else (sx_cfg.penalty     if sx_cfg else 1.15)
    _top_k       = top_k       if top_k       is not None else (sx_cfg.top_k       if sx_cfg else 5)
    _fast_max    = fast_max_iter if fast_max_iter is not None else (sx_cfg.fast_max_iter if sx_cfg else 32)
    _beam_width  = beam_width  if beam_width  is not None else (sx_cfg.beam_width  if sx_cfg else 4)
    _beam_max    = beam_max_iter if beam_max_iter is not None else (sx_cfg.beam_max_iter if sx_cfg else 64)
    
    # Resolve tokens (default to tokenizer if not provided)
    if bos_id is None:
        bos_id = tokenizer.token_to_id("<bos>")
    if eos_id is None:
        eos_id = tokenizer.token_to_id("<eos>")
        
    results = {}
    
    if isinstance(image_paths, str):
        image_paths = [image_paths]
        
    for image_path in image_paths:
        crops = extract_crops_from_image(image_path, doctr_model)
        if not crops:
            results[image_path] = ""
            continue
            
        dataset = Makeset(images=crops)
        dataloader = DataLoader(dataset, batch_size=wpb, shuffle=False, collate_fn=dataset.collate_fn)
        
        doc_text = []
        for batch in dataloader:
            batch = batch.to(device)
            for img in batch:
                img = img.unsqueeze(0) # [1, C, H, W]
                
                if function == "generate":
                    pred_ids = generate(model, img, bos_id=bos_id, eos_id=eos_id, device=device,
                                        temp=_temp, max_iter=_max_iter, penalty=_penalty, top_k=_top_k)
                elif function == "generate_fast":
                    pred_ids = generate_fast(model, img, bos_id=bos_id, eos_id=eos_id, device=device,
                                             max_iter=_fast_max)
                elif function == "beam_search":
                    pred_ids = beam_search(model, img, bos_id=bos_id, eos_id=eos_id, device=device,
                                           beam_width=_beam_width, max_iter=_beam_max)
                else:
                    if callable(function):
                        pred_ids = function(model, img, bos_id, eos_id, device)
                    else:
                        raise ValueError(f"Unknown function: {function}")
                
                text = tokenizer.decode(pred_ids)
                doc_text.append(text)
                
        results[image_path] = " ".join(doc_text)
        
    return results
