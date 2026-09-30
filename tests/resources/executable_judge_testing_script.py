import argparse
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--color", type=str, default="blue", help="color to save to metadata"
    )
    parser.add_argument("output_file", help="path to output file")
    parser.add_argument("--invalid_json", default=False, action="store_true")
    parser.add_argument("--print-message", type=str, help="message to print to stdout")
    args = parser.parse_args()

    if args.print_message:
        print(args.print_message)

    with open("model_save.txt") as f:
        score = int(f.read())
    # number must be converted to str by executable judge
    metadata = {"color": args.color, "number": 123}

    with open(args.output_file, "w") as f:
        if args.invalid_json:
            json.dump({"invalid": "fail"}, f)
        else:
            json.dump({"score": score, "metadata": metadata}, f)
