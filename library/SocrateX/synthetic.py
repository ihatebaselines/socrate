import os
import random
import csv
import urllib.request
from PIL import Image, ImageDraw, ImageFont

def _download_or_read_text(source):
    if source.startswith("http://") or source.startswith("https://"):
        req = urllib.request.Request(source, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            text = resp.read().decode('utf-8', errors='ignore')
    else:
        with open(source, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
    return text

def _get_words_from_text(text):
    words = text.split()
    words = [w.strip('.,!?"\'()[]{}') for w in words]
    words = [w for w in words if w]
    return words

def _render_text(text, bg_color=255, font_size=32):
    try:
        # Use a default Windows font if available
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
        
    # Measure text
    tmp = Image.new("L", (16, 16), 255)
    draw = ImageDraw.Draw(tmp)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    
    pad_x = random.randint(10, 20)
    pad_y = random.randint(5, 15)
    
    cw = max(32, tw + pad_x * 2)
    ch = max(32, th + pad_y * 2)
    
    image = Image.new("RGB", (cw, ch), (bg_color, bg_color, bg_color))
    draw = ImageDraw.Draw(image)
    
    x = pad_x
    y = pad_y
    
    draw.text((x, y), text, fill=(0, 0, 0), font=font)
    
    # Slight rotation (beta silly distortion)
    angle = random.uniform(-2, 2)
    image = image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(bg_color,bg_color,bg_color))
    
    return image

def generate_silly_training_set(source, count, output_dir="silly_train"):
    """
    Generates images containing individual words extracted from 'source'.
    'Beta' utility for quick training on custom datasets.
    Returns the path to the generated labels.csv file.
    """
    os.makedirs(output_dir, exist_ok=True)
    text = _download_or_read_text(source)
    words = _get_words_from_text(text)
    
    if not words:
        raise ValueError("Source has no valid words.")
        
    labels_file = os.path.join(output_dir, "labels.csv")
    
    with open(labels_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label"])
        
        for i in range(count):
            word = random.choice(words)
            img = _render_text(word)
            
            img_name = f"train_{i:06d}.jpg"
            img_path = os.path.join(output_dir, img_name)
            img.save(img_path)
            
            writer.writerow([img_path, word])
            
            if i % 100 == 0:
                print(f"Generated {i}/{count} training samples...", end="\r")
                
    print(f"\nSaved {count} training samples to {labels_file}")
    return labels_file

def generate_silly_testing_set(source, count, output_dir="silly_test"):
    """
    Generates images containing sentences of 2-6 words extracted from 'source'.
    'Beta' utility for testing inference on continuous text.
    Returns the path to the generated labels.csv file.
    """
    os.makedirs(output_dir, exist_ok=True)
    text = _download_or_read_text(source)
    words = _get_words_from_text(text)
    
    if not words:
        raise ValueError("Source has no valid words.")
        
    labels_file = os.path.join(output_dir, "labels.csv")
    
    with open(labels_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label"])
        
        for i in range(count):
            sentence_len = random.randint(2, 6)
            sentence_words = [random.choice(words) for _ in range(sentence_len)]
            sentence = " ".join(sentence_words)
            
            img = _render_text(sentence)
            
            img_name = f"test_{i:06d}.jpg"
            img_path = os.path.join(output_dir, img_name)
            img.save(img_path)
            
            writer.writerow([img_path, sentence])
            
            if i % 100 == 0:
                print(f"Generated {i}/{count} testing samples...", end="\r")
                
    print(f"\nSaved {count} testing samples to {labels_file}")
    return labels_file
