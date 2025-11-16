from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from transformers.pipelines.pt_utils import KeyDataset
import datasets
from tqdm.auto import tqdm
import os
import csv

def load_text_dataset(dataset_name):
    return datasets.load_dataset("text", data_files=dataset_name)["train"]

def load_model_and_pipe(model_name):
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, max_len=512)
    return pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=0,
        truncation=True,
        padding=True,
        max_length=512
    )

def run_binary_classifier(pipe, dataset, positive_label, score_threshold=0.8, weight_map=None):
    """Generic pass for yes/no, risk/neutral, etc."""
    total = 0
    count = 0

    for out in tqdm(pipe(KeyDataset(dataset, "text"))):
        if out["score"] >= score_threshold:
            total += 1
            if out["label"] in weight_map:
                count += weight_map[out["label"]]

    return count, total

def relatedness(model_name, dataset_name, output_file="related_data.txt"):
    dataset = load_text_dataset(dataset_name)
    pipe = load_model_and_pipe(model_name)

    total = 0
    count = 0
    kept = []

    for text, out in zip(dataset["text"], tqdm(pipe(KeyDataset(dataset, "text")))):
        if out["score"] >= 0.8:
            total += 1
            if out["label"] == "yes":
                count += 1
                kept.append(text)

    with open(output_file, "w", encoding="utf-8") as f:
        for line in kept:
            f.write(f"\"{line}\"\n")

    return count / total if total > 0 else 0


def specificity(_, dataset_name):
    dataset = load_text_dataset(dataset_name)
    pipe = load_model_and_pipe("climatebert/distilroberta-base-climate-specificity")

    count, total = run_binary_classifier(pipe, dataset, positive_label="spec")
    return count / total if total > 0 else 0


def sentiment(_, dataset_name):
    dataset = load_text_dataset(dataset_name)
    pipe = load_model_and_pipe("climatebert/distilroberta-base-climate-sentiment")

    # weight risk=2, neutral=1
    weight_map = {"risk": 2, "neutral": 1}

    count, total = run_binary_classifier(pipe, dataset, positive_label="risk", weight_map=weight_map)
    return round(count / (total * 2), 2) if total > 0 else 0


def commitment(_, dataset_name):
    dataset = load_text_dataset(dataset_name)
    pipe = load_model_and_pipe("climatebert/distilroberta-base-climate-commitment")

    count, total = run_binary_classifier(pipe, dataset, positive_label="yes")
    return round(count / total, 2) if total > 0 else 0


def climatetcfd(model_name, dataset_name):
    dataset = load_text_dataset(dataset_name)
    pipe = load_model_and_pipe(model_name)

    categories = {"metrics": 0, "strategy": 0, "governance": 0, "risk": 0}

    for out in tqdm(pipe(KeyDataset(dataset, "text"))):
        if out["score"] >= 0.8:
            if out["label"] in categories:
                categories[out["label"]] += 1

    return categories

def run_all_metrics_for_file(dataset_name):
    name = os.path.splitext(os.path.basename(dataset_name))[0]

    filtered_file = f"{name}_related.txt"

    relate = relatedness(
        "climatebert/distilroberta-base-climate-detector",
        dataset_name,
        filtered_file,
    )

    spec = specificity(
        "climatebert/distilroberta-base-climate-specificity",
        filtered_file,
    )

    senti = sentiment(
        "climatebert/distilroberta-base-climate-sentiment",
        filtered_file,
    )

    commit = commitment(
        "climatebert/distilroberta-base-climate-commitment",
        filtered_file,
    )

    tcfd = climatetcfd(
        "climatebert/distilroberta-base-climate-tcfd",
        filtered_file,
    )

    return {
        "name": name,
        "relate": relate,
        "spec": spec,
        "senti": senti,
        "commit": commit,
        "metrics": tcfd.get("metrics", 0),
        "strategy": tcfd.get("strategy", 0),
        "governance": tcfd.get("governance", 0),
        "risk": tcfd.get("risk", 0),
    }


if __name__ == "__main__":
    import sys

    input_files = sys.argv[1:]
    if not input_files:
        print("Usage: python main.py file1.txt [file2.txt ...]")
        raise SystemExit(1)

    fieldnames = [
        "name",
        "relate",
        "spec",
        "senti",
        "commit",
        "metrics",
        "strategy",
        "governance",
        "risk",
    ]

    with open("results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for path in input_files:
            print(f"Processing {path} ...")
            row = run_all_metrics_for_file(path)
            writer.writerow(row)

    print("results.csv written.")


