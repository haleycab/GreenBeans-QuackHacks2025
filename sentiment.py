from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from transformers.pipelines.pt_utils import KeyDataset
import datasets
from tqdm.auto import tqdm

dataset_name = "local_spectest.csv"
dataset = datasets.load_dataset("csv", data_files=dataset_name)["train"]

# dataset.to_csv("sentiment_test.csv")
# print("Saved to sentiment_test.csv")

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

for out in tqdm(pipe(KeyDataset(dataset, "text"), padding=True, truncation=True)):
   print(out)