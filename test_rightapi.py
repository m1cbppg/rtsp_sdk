# test_rightapi.py

import requests
import json

API_KEY = "sk-912ace9284c24a50a8976c763ae0e776"

# 模型名直接从 https://www.rightapi.ai/models 复制
MODEL = "gpt-5.6-sol"

url = "https://www.rightapi.ai/codex/v1/responses"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

data = {
    "model": MODEL,
    "input": [
        {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": """
请分析下面这个问题：

一个企业内部 Agent 有 200 个 API Tool。
如果把 200 个 Tool 全部放进 LLM context，会有什么问题？
应该如何设计动态 Tool Retrieval？

请从准确率、Token、延迟、工程复杂度四个方面回答。
"""
                }
            ],
        }
    ],
    "stream": False,
}

resp = requests.post(
    url,
    headers=headers,
    json=data,
    timeout=180,
)

print("status:", resp.status_code)

if resp.ok:
    result = resp.json()

    print("\n===== RAW RESPONSE =====")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    # 尝试直接提取文本
    print("\n===== MODEL OUTPUT =====")
    for item in result.get("output", []):
        if item.get("type") == "message":
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    print(content.get("text"))
else:
    print(resp.text)