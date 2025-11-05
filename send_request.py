import time
from openai import OpenAI
from PIL import Image
from vllm.multimodal.utils import encode_image_base64
client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
    timeout=3600
)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image_pil",
                "image_pil": encode_image_base64(Image.open("/data/chitb/VLM/data_superpod/sglang/funui_alipay_0000.png"))
            },
            {
                "type": "text",
                "text": "Extract elements and return json structured output" 
            }
        ]
    }
]

start = time.time()
response = client.chat.completions.create(
    model="Qwen/Qwen3-VL-32B-Instruct",
    messages=messages,
    max_tokens=2048
)
print(f"Response costs: {time.time() - start:.2f}s")
print(f"Generated text: {response.choices[0].message.content}")
