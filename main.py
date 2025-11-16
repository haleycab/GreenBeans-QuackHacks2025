from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from transformers.pipelines.pt_utils import KeyDataset
import datasets
from tqdm.auto import tqdm

def relatedness(model_name, dataset_name, output_file="related_data.txt"):
    dataset = datasets.load_dataset("text", data_files=dataset_name)["train"]

    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, max_len=512)

    pipe = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=0,
        truncation=True,
        padding=True,
        max_length=512
    )

    count = 0
    total = 0
    kept_lines = []
    for text, out in zip(dataset["text"], tqdm(pipe(KeyDataset(dataset, "text")))):
        # print the raw pipeline output
        # print(out)

        if out["score"] >= 0.8:
            total += 1
            if out["label"] == "yes":
                count += 1
                kept_lines.append(text)

    # print(f"\nhigh confidence lines: {total}")
    # print(f"related lines: {count}")
    with open(output_file, "w", encoding="utf-8") as f:
        for line in kept_lines:
            f.write(f'"{line}"\n')
    return count/total if total > 0 else 0 # return p of related lines

def specificity(model_name, dataset_name):
    dataset = datasets.load_dataset("text", data_files=dataset_name)["train"]

    model_name = "climatebert/distilroberta-base-climate-specificity"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, max_len=512)

    pipe = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=0,
        truncation=True,
        padding=True,
        max_length=512
    )

    count = 0
    total = 0
    for out in tqdm(pipe(KeyDataset(dataset, "text"))):
        if out['score'] >= 0.8:
            total += 1
            if out['label'] == 'spec':
                count += 1
        # print(out)
    # print(f"Specificity count: {count} out of {total} with score >= 0.8")
    # print(f"Specificity percentage: {count/total if total > 0 else 0}")
    return count/total if total > 0 else 0 # return p of specific lines

def sentiment(model_name, dataset_name):
    dataset = datasets.load_dataset("text", data_files=dataset_name)["train"]

    model_name = "climatebert/distilroberta-base-climate-sentiment"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, max_len=512)

    pipe = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=0,
        truncation=True,
        padding=True,
        max_length=512
    )
    total = 0
    count = 0
    for out in tqdm(pipe(KeyDataset(dataset, "text"), padding=True, truncation=True)):
        if out['score'] >= 0.8:
            total += 1
            if out['label'] == 'risk':
                count += 2
            elif out['label'] == 'neutral':
                count += 1
        # print(out)
    # print(f"risk score {count/(total*2)} with score >= 0.8")
    return round(count/(total*2), 2) if total > 0 else 0 # return risk score

def commitment(model_name, dataset_name):
    dataset = datasets.load_dataset("text", data_files=dataset_name)["train"]

    model_name = "climatebert/distilroberta-base-climate-commitment"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, max_len=512)

    pipe = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=0,
        truncation=True,
        padding=True,
        max_length=512
    )

    count = 0
    total = 0
    for out in tqdm(pipe(KeyDataset(dataset, "text"))):
        if out['score'] >= 0.8:
            total += 1
            if out['label'] == 'yes':
                count += 1
        # print(out)
    # print(f"Commitment p: {count/total if total > 0 else 0}")
    return round(count/total, 2) if total > 0 else 0 # return p of commitment lines


def climatetcfd(model_name, dataset_name):
    dataset = datasets.load_dataset("text", data_files=dataset_name)["train"]

    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, max_len=512)

    pipe = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=0,
        truncation=True,
        padding=True,
        max_length=512
    )

    categories = {"metrics": 0, "strategy": 0, "governance": 0, "risk": 0}
    for out in tqdm(pipe(KeyDataset(dataset, "text"), padding=True, truncation=True)):
        if out['score'] >= 0.8:
            categories[out['label']] += 1
            # print(out)

    total = sum(categories.values())
    for category, count in categories.items():
        percent = count / total * 100 if total > 0 else 0
        # print(f"{category}: {count} which is {percent:.2f}%")
    return categories

if __name__ == "__main__":
    relate = relatedness("climatebert/distilroberta-base-climate-detector", "sample_data.txt", "sample_output.txt")
    print(f"Relatedness p: {relate}\n")
    spec = specificity("climatebert/distilroberta-base-climate-specificity", "sample_output.txt")
    print(f"Specificity p: {spec}\n")
    senti = sentiment("climatebert/distilroberta-base-climate-sentiment", "sample_output.txt")
    print(f"Sentiment risk score: {senti}\n")
    commit = commitment("climatebert/distilroberta-base-climate-commitment", "sample_output.txt")
    print(f"Commitment p: {commit}\n")
    tcfd = climatetcfd("climatebert/distilroberta-base-climate-tcfd", "sample_output.txt")
    print(f"TCFD categories: {tcfd}\n")