from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

class Qwen25Finetuner:
    def __init__(self, model_name="Qwen/Qwen2.5-7B", data_path="path/to/your/dataset.json", output_dir="./qwen2.5-finetuned",
                 lora_r=8, lora_alpha=32, lora_dropout=0.05, train_batch_size=2, gradient_accumulation_steps=4,
                 learning_rate=2e-4, num_epochs=3):
        self.model_name = model_name
        self.data_path = data_path
        self.output_dir = output_dir
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.train_batch_size = train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.dataset = None
        self.tokenizer = None
        self.model = None
        self.tokenized_dataset = None

    def _load_and_preprocess_data(self):
        dataset = load_dataset("json", data_files=self.data_path)["train"]
        def format_prompt(sample):
            instruction = sample["instruction"]
            input_text = sample.get("input", "")
            output = sample["output"]
            if input_text:
                return {"text": f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{instruction}\n{input_text}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"}
            else:
                return {"text": f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"}
        self.dataset = dataset.map(format_prompt)

    def _tokenize_data(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=1024,
                padding="max_length",
                return_tensors="pt"
            )
        self.tokenized_dataset = self.dataset.map(tokenize_function, batched=True, remove_columns=["text", "instruction", "input", "output"])

    def _load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            load_in_4bit=True,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        self.model = prepare_model_for_kbit_training(self.model)

    def _configure_lora(self):
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def _setup_training_args(self):
        return TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            num_train_epochs=self.num_epochs,
            logging_dir=f"{self.output_dir}/logs",
            logging_steps=10,
            save_strategy="epoch",
            optim="paged_adamw_8bit",
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            report_to="tensorboard",
            fp16=True
        )

    def train(self):
        self._load_and_preprocess_data()
        self._tokenize_data()
        self._load_model()
        self._configure_lora()
        training_args = self._setup_training_args()
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        )
        trainer.train()
        self.model.save_pretrained(f"{self.output_dir}/lora_weights")
        self.tokenizer.save_pretrained(self.output_dir)