for model in Qwen/Qwen2.5-7B-Instruct open-r1/OpenR1-Distill-7B \
    bigcode/starcoder2-7b microsoft/NextCoder-7B \
    google/codegemma-7b deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    google/gemma-2-9b-it open-r1/OlympicCoder-7B Qwen/Qwen3-8B; do
    python code/agent_qwen3-8B.py $model;
done