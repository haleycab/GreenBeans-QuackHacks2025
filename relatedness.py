from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from transformers.pipelines.pt_utils import KeyDataset
import datasets
from tqdm.auto import tqdm

dataset_name = "local_spectest.csv"
dataset = datasets.load_dataset("csv", data_files=dataset_name)["train"]

model_name = "climatebert/distilroberta-base-climate-detector"
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
    print(out)

    if out["score"] >= 0.8:
        total += 1
        if out["label"] == "yes":
            count += 1
            kept_lines.append(text)

print(f"\nhigh confidence lines: {total}")
print(f"related lines: {count}")
with open("related_data.txt", "w", encoding="utf-8") as f:
    for line in kept_lines:
        f.write(f'"{line}"\n')

print("Saved")
   
         
