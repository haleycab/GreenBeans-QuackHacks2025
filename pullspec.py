from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from transformers.pipelines.pt_utils import KeyDataset
import datasets
from tqdm.auto import tqdm

#change file name here
dataset_name = "data/processed_reports_text.txt"
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

kept_lines = []
for text, out in zip(dataset["text"], tqdm(pipe(KeyDataset(dataset, "text")))):
    if out['score'] >= 0.8:
        if out['label'] == 'spec':
            kept_lines.append(text)
            print(out)

with open("specific_data.txt", "w", encoding="utf-8") as f:
    for line in kept_lines:
        f.write(f'"{line}"\n')

