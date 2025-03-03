from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
system_prompt="You are a senior physician specializing in HIV. You are assisting a junior clinician answering medical questions. Keep your answers brief and clear. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
question = 'What are thes symptoms of HIV?'
messages = [
    {"role":"system", "content":system_prompt},
    {"role":"user", "content":question}
]

tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.3-70B-Instruct",
)
messages_templated = tokenizer.apply_chat_template(messages, tokenize=False)

custom_cache_dir = "/cluster/scratch/gcardenal/LLM_models"
model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.1-8B",
            cache_dir = custom_cache_dir,
            device_map="auto",
            torch_dtype="auto")

pipe = pipeline("text-generation",model=model, tokenizer = tokenizer, model_kwargs={"torch_dtype": torch.bfloat16},
device_map="auto")
messages = [
    {"role":"system", "content":system_prompt},
    {"role":"user", "content":question}
]
messages_templated = tokenizer.apply_chat_template(messages, tokenize=False)
#answer = pipe(messages_templated, max_new_tokens = 1024)[0]['generated_text'][-1]
answer = pipe(messages_templated, max_new_tokens = 1024)[0]
print(type(answer), answer)