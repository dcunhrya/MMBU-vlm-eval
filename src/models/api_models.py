import base64
import os
from concurrent.futures import ThreadPoolExecutor
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


def _chat_messages_to_responses_input(messages):
    responses_input = []
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            responses_input.append(
                {
                    "role": message["role"],
                    "content": [{"type": "input_text", "text": content}],
                }
            )
            continue

        response_content = []
        for part in content:
            if part["type"] == "text":
                response_content.append({"type": "input_text", "text": part["text"]})
            elif part["type"] == "image_url":
                response_content.append(
                    {"type": "input_image", "image_url": part["image_url"]["url"]}
                )
        responses_input.append({"role": message["role"], "content": response_content})
    return responses_input


def _max_workers(batch_size):
    if batch_size <= 0:
        return 1
    raw_value = os.environ.get("FRONTIER_MAX_WORKERS")
    if raw_value is None:
        return batch_size
    try:
        return max(1, min(batch_size, int(raw_value)))
    except ValueError:
        return batch_size


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

    def _infer_one(self, model, messages, max_new_tokens):
        if self.model_name.startswith("gpt-5"):
            response = model.responses.create(
                model=self.model_name,
                input=_chat_messages_to_responses_input(messages),
                max_output_tokens=max_new_tokens,
            )
            return response.output_text or ""

        response = model.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=0,
        )
        return response.choices[0].message.content or ""

    def infer(self, model, processor, inputs, max_new_tokens):
        with ThreadPoolExecutor(max_workers=_max_workers(len(inputs))) as executor:
            return list(
                executor.map(
                    lambda messages: self._infer_one(model, messages, max_new_tokens),
                    inputs,
                )
            )


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

    def _infer_one(self, api_key, url, payload, max_new_tokens):
        payload["generationConfig"]["maxOutputTokens"] = max_new_tokens
        response = requests.post(
            url,
            params={"key": api_key},
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        parts = data["candidates"][0]["content"].get("parts", [])
        return "".join(part.get("text", "") for part in parts)

    def infer(self, model, processor, inputs, max_new_tokens):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
        with ThreadPoolExecutor(max_workers=_max_workers(len(inputs))) as executor:
            return list(
                executor.map(
                    lambda payload: self._infer_one(model, url, payload, max_new_tokens),
                    inputs,
                )
            )
