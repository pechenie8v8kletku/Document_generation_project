from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
import os
from transformers import DataCollatorForLanguageModeling



os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch._dynamo
torch._dynamo.config.suppress_errors = True

MODEL_NAME = "SBERmodelMedium/rugpt3medium_based_on_gpt2"
JSON_PATH = "GOSTS_JSON_DATASET.json"
OUTPUT_DIR = "SBERGPT2trainedgosts"
MAX_LENGTH = 2048
def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        #torch_dtype=torch.float16,
        trust_remote_code=True
    )
    model.config.use_cache = False
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    dataset = load_dataset("json", data_files=JSON_PATH)["train"]
    dataset = dataset.train_test_split(test_size=0, seed=42)
    train_data, val_data = dataset["train"], dataset["test"]
    print(tokenizer.pad_token_id)
    print(tokenizer.vocab_size)

    def tokenize(examples):
        all_chunks = {
            "input_ids": [],
            "attention_mask": [],
            "labels": []
        }

        for input_text, output_text in zip(examples["input"], examples["output"]):
            prompt = f"### Вопрос:\n{input_text}\n\n### Ответ:\n{output_text}"
            tokens = tokenizer(prompt, truncation=True, max_length=MAX_LENGTH, padding="max_length")


            input_ids = tokens["input_ids"]
            attention_mask = tokens["attention_mask"]

            for i in range(0, len(input_ids), MAX_LENGTH):
                chunk_input_ids = input_ids[i:i + MAX_LENGTH]
                chunk_attention_mask = attention_mask[i:i + MAX_LENGTH]
                chunk_labels = [
                    token if mask == 1 else -100
                    for token, mask in zip(chunk_input_ids, chunk_attention_mask)
                ]

                all_chunks["input_ids"].append(chunk_input_ids)
                all_chunks["attention_mask"].append(chunk_attention_mask)
                all_chunks["labels"].append(chunk_labels)

        return all_chunks

    train_data = train_data.map(
        tokenize,
        batched=True,
        remove_columns=train_data.column_names
    )
    val_data = val_data.map(
        tokenize,
        batched=True,
        remove_columns=val_data.column_names
    )

    model.gradient_checkpointing_enable()

    training_args = TrainingArguments(
        output_dir="../qwen3-fp16-no-lora",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-7,
        num_train_epochs=2,
        fp16=True,
        logging_steps=1,
        save_strategy="steps",
        save_steps= 300,
        save_total_limit= 4,

        report_to="none", max_grad_norm=1.0,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
    data_collator=data_collator
    )

    trainer.train()

    model.save_pretrained(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")

if __name__ == "__main__":
    main()
