from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
from llama_cpp import Llama
import json
from ziga import append_to_json_file
json_path= "../just_md_ibm.json"
with open(json_path, "r", encoding="utf-8") as jf:
     data = json.load(jf)


out_path="yandex_worked_on_gosts"
model = Llama(model_path="../yandex/YandexGPT-5-Lite-8B-instruct.Q4_K_S.gguf", n_ctx=25000, n_gpu_layers=-1, verbose=True)

entries=[]
for a in data:
    txt1=a["input"]
    txt2=a["output"]
    response = model(
        f"Попробуй немного сжать текст этого госта и убрать мусор, но сохрани всю важную техническую информацию так как при работе с гостом потеря кусочка важной информации критична, убери не несущие полезной информации текст,но оставь каждый раздел с всей несущей смысл в нем информацией, но при этом он все еще должен иметь важные значения и данные из исходного документа,не стоит сжимать более чем в 2 или 3 раза{txt2}\n",
        max_tokens=25000)
    entr = {
        "input": txt1,
        "output": response["choices"][0]["text"]
    }
    entries.append(entr)


append_to_json_file(out_path,entries)

#
# MODEL_NAME = "yandex/YandexGPT-5-Lite-8B-instruct.Q4_K_S.gguf"
#
# with open("output_text.txt", "r", encoding="utf-8") as f:
#     content=f.read()
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
#
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME,
#     device_map="auto",
#     trust_remote_code=True
# )
#
# model.eval()  # режим инференса
# prompt = f"Раздели этот текст на смысловые части и верни результат по типу раздел 1 Введение ...  раздел 2 Нормативные ссылки...  и так далее  сам текст:{content}\n"
#
# inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#
# with torch.no_grad():
#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=64000,
#         eos_token_id=tokenizer.eos_token_id
#     )
#
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))
