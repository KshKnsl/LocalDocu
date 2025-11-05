export interface LocalModelInfo {
    name: string;
    company: string;
    bestAt: string;
}

export const LOCAL_MODELS: LocalModelInfo[] = [
    { name: "remote", company: "Google", bestAt: "Cloud-based Gemini" },
    { name: "llama3", company: "Meta", bestAt: "General purpose, reasoning" },
    { name: "mistral", company: "Mistral AI", bestAt: "General purpose, fast" },
    { name: "phi3", company: "Microsoft", bestAt: "Efficient, small model" },
    { name: "codellama", company: "Meta", bestAt: "Code generation" },
    { name: "mixtral", company: "Mistral AI", bestAt: "Advanced reasoning" },
    { name: "qwen2", company: "Alibaba", bestAt: "Multilingual, general" },
    { name: "deepseek-coder", company: "DeepSeek", bestAt: "Code & debugging" },
    { name: "llava", company: "Llava.ai", bestAt: "Vision & images" },
    { name: "gemma", company: "Google", bestAt: "Efficient reasoning" },
];
