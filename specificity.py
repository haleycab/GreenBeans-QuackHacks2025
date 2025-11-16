from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from transformers.pipelines.pt_utils import KeyDataset
import datasets
from tqdm.auto import tqdm

#change file name here
dataset_name = "local_spectest.csv"
dataset = datasets.load_dataset("csv", data_files=dataset_name)["train"]

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
    print(out)
print(f"Specificity count: {count} out of {total} with score >= 0.8")
print(f"Specificity percentage: {count/total*100 if total > 0 else 0}%")
