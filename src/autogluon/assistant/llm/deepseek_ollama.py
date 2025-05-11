import requests
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from langchain.schema import AIMessage, BaseMessage
import logging

logger = logging.getLogger(__name__)

class AssistantChatDeepSeek(BaseModel):
    """DeepSeek LLM remote API adapter."""
    api_url: str
    api_key: Optional[str] = None
    model_name: str = "deepseek-chat"
    temperature: float = 0.7
    max_tokens: int = 512
    verbose: bool = False
    history_: List[Dict[str, Any]] = Field(default_factory=list)
    input_: int = Field(default=0)
    output_: int = Field(default=0)

    def describe(self) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "api_url": self.api_url,
            "history": self.history_,
            "prompt_tokens": self.input_,
            "completion_tokens": self.output_,
        }

    def invoke(self, messages: List[BaseMessage], **kwargs) -> AIMessage:
        payload = {
            "model": self.model_name,
            "messages": [{"role": m.type, "content": m.content} for m in messages],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.verbose:
            logger.info(f"DeepSeek API request: {payload}")
        resp = requests.post(self.api_url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        # Assume OpenAI-compatible response
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        self.input_ += usage.get("prompt_tokens", 0)
        self.output_ += usage.get("completion_tokens", 0)
        result = AIMessage(content=content, usage_metadata=usage)
        self.history_.append({
            "input": payload["messages"],
            "output": content,
            "prompt_tokens": self.input_,
            "completion_tokens": self.output_,
        })
        return result

class AssistantChatOllama(BaseModel):
    """Ollama LLM remote API adapter."""
    api_url: str = "http://localhost:11434/api/chat"
    model_name: str = "llama3"
    temperature: float = 0.7
    max_tokens: int = 512
    verbose: bool = False
    history_: List[Dict[str, Any]] = Field(default_factory=list)
    input_: int = Field(default=0)
    output_: int = Field(default=0)

    def describe(self) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "api_url": self.api_url,
            "history": self.history_,
            "prompt_tokens": self.input_,
            "completion_tokens": self.output_,
        }

    def invoke(self, messages: List[BaseMessage], **kwargs) -> AIMessage:
        payload = {
            "model": self.model_name,
            "messages": [{"role": m.type, "content": m.content} for m in messages],
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens},
        }
        if self.verbose:
            logger.info(f"Ollama API request: {payload}")
        resp = requests.post(self.api_url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        content = data["message"]["content"]
        # Ollama may not return token usage; set as 0 or parse if available
        usage = data.get("usage", {})
        self.input_ += usage.get("prompt_tokens", 0)
        self.output_ += usage.get("completion_tokens", 0)
        result = AIMessage(content=content, usage_metadata=usage)
        self.history_.append({
            "input": payload["messages"],
            "output": content,
            "prompt_tokens": self.input_,
            "completion_tokens": self.output_,
        })
        return result
