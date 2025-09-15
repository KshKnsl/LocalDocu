export interface LocalModelInfo {
    name: string;
    company: string;
    bestAt: string;
}

export const LOCAL_MODELS: LocalModelInfo[] = [
    { name: "llava", company: "Llava.ai", bestAt: "Image understanding" },
    { name: "llama2", company: "Meta", bestAt: "General purpose" },
    { name: "llama3", company: "Meta", bestAt: "General purpose" },
    { name: "mistral", company: "Mistral AI", bestAt: "General purpose" },
    { name: "phi3", company: "Microsoft", bestAt: "General purpose" },
    { name: "codellama", company: "Meta", bestAt: "Code generation" },
    { name: "mixtral", company: "Mistral AI", bestAt: "General purpose" },
    { name: "qwen2", company: "Alibaba", bestAt: "General purpose" },
    { name: "starcoder2", company: "BigCode", bestAt: "Code generation" },
    { name: "gpt-oss", company: "Open Source", bestAt: "General purpose" },
    { name: "dolphin-mixtral", company: "Cognitive Computation", bestAt: "General purpose" },
    { name: "openhermes", company: "OpenHermes", bestAt: "General purpose" },
    { name: "vicuna", company: "LMSYS", bestAt: "General purpose" },
    { name: "zephyr", company: "HuggingFace", bestAt: "General purpose" },
    { name: "orca-mini", company: "Microsoft", bestAt: "General purpose" },
    { name: "nous-hermes", company: "Nous Research", bestAt: "General purpose" },
    { name: "wizardlm", company: "WizardLM Team", bestAt: "General purpose" },
    { name: "deepseek-coder", company: "DeepSeek", bestAt: "Code generation" },
    { name: "deepseek-llm", company: "DeepSeek", bestAt: "General purpose" },
    { name: "stablelm-2", company: "Stability AI", bestAt: "General purpose" },
    { name: "yi-34b", company: "01.AI", bestAt: "General purpose" },
];
