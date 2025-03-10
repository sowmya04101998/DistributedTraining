from transformers import BertTokenizerFast, BertForMaskedLM

# Specify the model name
model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"

# Download and save tokenizer and model
tokenizer = BertTokenizerFast.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)

# Save the tokenizer and model locally
tokenizer.save_pretrained("./biomedbert")
model.save_pretrained("./biomedbert")
