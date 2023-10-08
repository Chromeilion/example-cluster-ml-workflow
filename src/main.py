from datasets import load_dataset
from transformers import (
    Trainer,
    BertForSequenceClassification,
    AutoTokenizer,
    TrainingArguments
)
from settings import Settings
from functools import partial


def preprocess(x, tokenizer):
    return tokenizer(x["text"], padding=True, truncation=True),


def main():
    config = Settings()  # Load all settings stored in the environment

    train = load_dataset(config.dataset, split='train')
    test = load_dataset(config.dataset, split='test')
    model = BertForSequenceClassification.from_pretrained(
        config.pretrained_model, num_labels=2
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.pretrained_model,
        model_max_length=128
    )
    preprocess_p = partial(preprocess, tokenizer=tokenizer)
    # Tokenize the entire dataset before training
    train = train.map(lambda x: preprocess_p(x)[0], batched=True)
    test = test.map(lambda x: preprocess_p(x)[0], batched=True)
    trainer = Trainer(
        model,
        TrainingArguments(**config.trainer_args.model_dump()),
        train_dataset=train,
        eval_dataset=test
    )
    trainer.train()


if __name__ == "__main__":  # This is needed to allow for multiprocessing
    main()
