import sys
import argparse
import SocrateX as sx

def interactive_menu():
    print("="*40)
    print("   SocrateX CLI - Interactive Menu")
    print("="*40)
    print("1. Predict on an image")
    print("2. Train a tokenizer")
    print("3. Generate synthetic data (Beta)")
    print("4. Train a model")
    print("5. Exit")
    print("="*40)
    
    choice = input("Select an option (1-5): ")
    
    if choice == '1':
        img = input("Enter image path: ")
        model_type = input("Model type (cat/rat/mice) [cat]: ") or "cat"
        weights = input("Weights file (leave empty for default): ") or None
        tok_path = input("Tokenizer path [ocr_bpe_tokenizer.json]: ") or "ocr_bpe_tokenizer.json"
        
        print("\nLoading model...")
        try:
            model, tokenizer = sx.load(model_type=model_type, weights=weights, tokenizer_path=tok_path)
            print("Predicting...")
            res = model.predict([img])
            print(f"\nResult for {img}:\n{res.get(img, '')}\n")
        except Exception as e:
            print(f"Error: {e}")
        
    elif choice == '2':
        source = input("Enter path to text source (file or url): ")
        vocab = input("Vocab size [1000]: ") or "1000"
        out = input("Output tokenizer path [custom_tokenizer.json]: ") or "custom_tokenizer.json"
        print("Training tokenizer...")
        try:
            tokenizer = sx.init_tokenizer(vocab_size=int(vocab))
            tokenizer.fit(source)
            tokenizer.save(out)
            print(f"Tokenizer saved to {out}\n")
        except Exception as e:
            print(f"Error: {e}")
        
    elif choice == '3':
        mode = input("Type of data: 1=Words (train), 2=Sentences (test) [1]: ") or "1"
        source = input("Source (text file or url): ")
        count = input("Number of images to generate [1000]: ") or "1000"
        out = input("Output directory [silly_data]: ") or "silly_data"
        
        print("Generating data...")
        try:
            if mode == '1':
                sx.generate_silly_training_set(source, int(count), output_dir=out)
            else:
                sx.generate_silly_testing_set(source, int(count), output_dir=out)
        except Exception as e:
            print(f"Error: {e}")
            
    elif choice == '4':
        print("\n[!] For full training, it is highly recommended to write a custom Python script.")
        print("This feature is WIP in the CLI. Check out model.fit() in the documentation!")
        
    elif choice == '5':
        sys.exit(0)
    else:
        print("Invalid choice. Try again.")
        
def main():
    parser = argparse.ArgumentParser(description="SocrateX Command Line Interface")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Predict
    parser_pred = subparsers.add_parser("predict", help="Run inference on an image")
    parser_pred.add_argument("image", help="Path to the image")
    parser_pred.add_argument("--model", default="cat", help="Model type: cat, rat, mice")
    parser_pred.add_argument("--weights", default=None, help="Path to weights file")
    
    # Generate
    parser_gen = subparsers.add_parser("generate", help="Generate synthetic data")
    parser_gen.add_argument("source", help="Path to text source or URL")
    parser_gen.add_argument("--count", type=int, default=1000, help="Number of samples")
    parser_gen.add_argument("--type", choices=["train", "test"], default="train", help="Train (words) or test (sentences)")
    parser_gen.add_argument("--out", default="silly_data", help="Output directory")
    
    # Tokenizer
    parser_tok = subparsers.add_parser("tokenizer", help="Train a custom tokenizer")
    parser_tok.add_argument("source", help="Path to text source or URL")
    parser_tok.add_argument("--vocab-size", type=int, default=1000, help="Vocabulary size")
    parser_tok.add_argument("--out", default="custom_tokenizer.json", help="Output path")

    args = parser.parse_args()

    if args.command is None:
        # Run interactive menu
        while True:
            interactive_menu()
            print()
    elif args.command == "predict":
        print(f"Loading {args.model} model...")
        model, tokenizer = sx.load(model_type=args.model, weights=args.weights)
        res = model.predict([args.image])
        print(f"Prediction: {res.get(args.image, '')}")
    elif args.command == "generate":
        if args.type == "train":
            sx.generate_silly_training_set(args.source, args.count, output_dir=args.out)
        else:
            sx.generate_silly_testing_set(args.source, args.count, output_dir=args.out)
    elif args.command == "tokenizer":
        print("Training tokenizer...")
        tokenizer = sx.init_tokenizer(vocab_size=args.vocab_size)
        tokenizer.fit(args.source)
        tokenizer.save(args.out)
        print(f"Saved to {args.out}")

if __name__ == "__main__":
    main()
