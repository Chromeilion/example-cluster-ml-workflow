from pydantic_settings import BaseSettings, SettingsConfigDict
from transformers import TrainingArguments


class BaseSettingsConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix='mlp_',
        env_file='.env',
        extra="ignore"
    )


class TrainerArgsSettings(BaseSettingsConfig, TrainingArguments):
    """
    Arguments that get passed to the Hugging Face Trainer.
    """
#    output_dir: str
#    seed: int
#    learning_rate: float
#    per_device_eval_batch_size: int
#    per_device_train_batch_size: int
#    warmup_steps: int
#    num_train_epochs: int
#    eval_steps: int
#    ddp_find_unused_parameters: bool
#    evaluation_strategy: str


class Settings(BaseSettingsConfig):
    """
    All settings for the project. These are stored in a .env file in the
    project root or just in the environment. All names start with
    "MLP_". For example, "MLP_DATASET" would be loaded into the dataset
    property of this class.
    """
    pretrained_model: str
    dataset: str
    trainer_args: TrainerArgsSettings = TrainerArgsSettings()
