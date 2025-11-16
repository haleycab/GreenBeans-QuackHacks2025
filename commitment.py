from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from transformers.pipelines.pt_utils import KeyDataset
import datasets
from tqdm.auto import tqdm

dataset_name = "related_data.txt"
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
    print(out)
print(f"Commitment count: {count} out of {total} with score >= 0.8")
print(f"Commitment p: {count/total if total > 0 else 0}")
