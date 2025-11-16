from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from transformers.pipelines.pt_utils import KeyDataset
import datasets
from tqdm.auto import tqdm

dataset_name = "related_data.txt"
dataset = datasets.load_dataset("text", data_files=dataset_name)["train"]

model_name = "climatebert/distilroberta-base-climate-tcfd"
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
        print(out)

total = sum(categories.values())
for category, count in categories.items():
    percent = count / total * 100 if total > 0 else 0
    print(f"{category}: {count} which is {percent:.2f}%")
