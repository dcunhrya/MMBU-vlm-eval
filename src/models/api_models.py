import base64
import os
from io import BytesIO

import requests

from .base import BaseVLMAdapter


def _image_to_base64_png(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _text_from_content(content):
    if isinstance(content, str):
        return content
    return "\n".join(part["text"] for part in content if part.get("type") == "text")


class OpenAIVisionAdapter(BaseVLMAdapter):
    def load(self):
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY must be set to use the OpenAI API adapter.")
        from openai import OpenAI

        return OpenAI(), None

    def prepare_inputs(self, messages, processor, model):
        requests_batch = []
        for message_group in messages:
            request_messages = []
            for message in message_group:
                content = message["content"]
                if message["role"] == "system":
                    request_messages.append(
                        {"role": "system", "content": _text_from_content(content)}
                    )
                    continue

                api_content = []
                for part in content:
                    if part["type"] == "text":
                        api_content.append({"type": "text", "text": part["text"]})
                    elif part["type"] == "image":
                        encoded = _image_to_base64_png(part["image"])
                        api_content.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{encoded}"},
                            }
                        )
                request_messages.append({"role": message["role"], "content": api_content})
            requests_batch.append(request_messages)
        return requests_batch

    def infer(self, model, processor, inputs, max_new_tokens):
        outputs = []
        for messages in inputs:
            response = model.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=0,
            )
            outputs.append(response.choices[0].message.content or "")
        return outputs


class GeminiVisionAdapter(BaseVLMAdapter):
    def load(self):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY must be set to use the Gemini API adapter.")
        return api_key, None

    def prepare_inputs(self, messages, processor, model):
        requests_batch = []
        for message_group in messages:
            system_text = ""
            user_parts = []
            for message in message_group:
                content = message["content"]
                if message["role"] == "system":
                    system_text = _text_from_content(content)
                    continue

                for part in content:
                    if part["type"] == "text":
                        user_parts.append({"text": part["text"]})
                    elif part["type"] == "image":
                        user_parts.append(
                            {
                                "inline_data": {
                                    "mime_type": "image/png",
                                    "data": _image_to_base64_png(part["image"]),
                                }
                            }
                        )

            payload = {
                "contents": [{"role": "user", "parts": user_parts}],
                "generationConfig": {"temperature": 0},
            }
            if system_text:
                payload["systemInstruction"] = {"parts": [{"text": system_text}]}
            requests_batch.append(payload)
        return requests_batch

    def infer(self, model, processor, inputs, max_new_tokens):
        outputs = []
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
        for payload in inputs:
            payload["generationConfig"]["maxOutputTokens"] = max_new_tokens
            response = requests.post(
                url,
                params={"key": model},
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()
            parts = data["candidates"][0]["content"].get("parts", [])
            outputs.append("".join(part.get("text", "") for part in parts))
        return outputs
