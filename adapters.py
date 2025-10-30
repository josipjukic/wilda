from peft import LoraConfig, PromptEncoderConfig, PrefixTuningConfig, TaskType


def lora_config():
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )


def ptuning_config():
    return PromptEncoderConfig(
        task_type=TaskType.CAUSAL_LM, num_virtual_tokens=20, encoder_hidden_size=128
    )


def prefix_tuning_config():
    return PrefixTuningConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=False, num_virtual_tokens=20
    )


PEFT_CONFIGS = {
    "lora": lora_config,
    "ptuning": ptuning_config,
    "prefix_tuning": prefix_tuning_config,
}


def peft_factory(name):
    return PEFT_CONFIGS[name]()
