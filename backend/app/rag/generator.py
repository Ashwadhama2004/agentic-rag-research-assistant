"""
Answer Generator
Generates answers using LLM with retrieved context.
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time

from backend.app.config import config
from backend.app.rag.retriever import RetrievalResult


@dataclass
class GeneratedAnswer:
    """Represents a generated answer with metadata."""
    answer: str
    sources: List[Dict[str, Any]]
    latency: float
    model: str
    tokens_used: int = 0


class AnswerGenerator:
    """
    Generates answers using LLM with retrieved context.
    Supports Groq and Ollama backends.
    """
    
    RAG_PROMPT_TEMPLATE = """You are a helpful research assistant. Answer the user's question based ONLY on the provided context. If the context doesn't contain enough information to answer the question, say so clearly.

CONTEXT:
{context}

USER QUESTION:
{question}

INSTRUCTIONS:
1. Answer based only on the provided context
2. If citing information, mention the source document
3. Be concise but thorough
4. If you cannot answer from the context, say "I cannot find this information in the provided documents."

ANSWER:"""

    CONVERSATION_TEMPLATE = """You are a helpful AI assistant. Continue the conversation naturally while being helpful and informative.

PREVIOUS CONVERSATION:
{history}

USER: {question}

ASSISTANT:"""

    def __init__(self):
        self.provider = config.LLM_PROVIDER
        self.model = config.LLM_MODEL
        self._client = None
    
    @property
    def client(self):
        """Get the LLM client."""
        if self._client is None:
            self._client = config.get_llm_client()
        return self._client
    
    def _format_context(self, results: List[RetrievalResult]) -> str:
        """Format retrieval results as context string."""
        context_parts = []
        
        for i, result in enumerate(results, 1):
            source_info = f"[Source {i}: {result.document_name}"
            if result.page_number:
                source_info += f", Page {result.page_number}"
            source_info += "]"
            
            context_parts.append(f"{source_info}\n{result.content}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def _extract_sources(self, results: List[RetrievalResult]) -> List[Dict[str, Any]]:
        """Extract source citations from results."""
        sources = []
        
        for result in results:
            source = {
                "document": result.document_name,
                "chunk_id": result.id,
                "relevance_score": round(result.score, 3)
            }
            
            if result.page_number:
                source["page"] = result.page_number
            
            sources.append(source)
        
        return sources
    
    def generate(
        self,
        query: str,
        context_results: List[RetrievalResult],
        max_tokens: int = 1024,
        temperature: float = 0.7
    ) -> GeneratedAnswer:
        """
        Generate an answer using retrieved context.
        
        Args:
            query: User's question
            context_results: Retrieved documents for context
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
        
        Returns:
            GeneratedAnswer with answer and metadata
        """
        start_time = time.time()
        
        # Format context
        context = self._format_context(context_results) if context_results else "No relevant documents found."
        
        # Build prompt
        prompt = self.RAG_PROMPT_TEMPLATE.format(
            context=context,
            question=query
        )
        
        # Generate response
        answer, tokens = self._call_llm(prompt, max_tokens, temperature)
        
        latency = time.time() - start_time
        
        return GeneratedAnswer(
            answer=answer,
            sources=self._extract_sources(context_results),
            latency=round(latency, 3),
            model=self.model,
            tokens_used=tokens
        )
    
    def generate_conversation(
        self,
        query: str,
        history: List[Dict[str, str]] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7
    ) -> GeneratedAnswer:
        """
        Generate a conversational response.
        
        Args:
            query: User's message
            history: Previous conversation messages
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
        """
        start_time = time.time()
        
        # Format history
        history_text = ""
        if history:
            for msg in history[-10:]:  # Last 10 messages
                role = msg.get("role", "user").upper()
                content = msg.get("content", "")
                history_text += f"{role}: {content}\n"
        
        prompt = self.CONVERSATION_TEMPLATE.format(
            history=history_text or "No previous conversation.",
            question=query
        )
        
        answer, tokens = self._call_llm(prompt, max_tokens, temperature)
        
        latency = time.time() - start_time
        
        return GeneratedAnswer(
            answer=answer,
            sources=[],
            latency=round(latency, 3),
            model=self.model,
            tokens_used=tokens
        )
    
    def _call_llm(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> tuple:
        """Call the LLM and return response with token count."""
        try:
            if self.provider == "groq":
                return self._call_groq(prompt, max_tokens, temperature)
            elif self.provider == "ollama":
                return self._call_ollama(prompt, max_tokens, temperature)
            else:
                raise ValueError(f"Unsupported LLM provider: {self.provider}")
        except Exception as e:
            return f"Error generating response: {str(e)}", 0
    
    def _call_groq(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> tuple:
        """Call Groq API."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        answer = response.choices[0].message.content
        tokens = response.usage.total_tokens if response.usage else 0
        
        return answer, tokens
    
    def _call_ollama(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> tuple:
        """Call Ollama API."""
        response = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={
                "num_predict": max_tokens,
                "temperature": temperature
            }
        )
        
        answer = response["message"]["content"]
        tokens = response.get("eval_count", 0)
        
        return answer, tokens
    
    def summarize(self, text: str, max_length: int = 200) -> str:
        """Generate a summary of the given text."""
        prompt = f"""Summarize the following text in {max_length} words or less. 
Be concise and capture the main points.

TEXT:
{text}

SUMMARY:"""
        
        answer, _ = self._call_llm(prompt, max_tokens=max_length * 2, temperature=0.5)
        return answer


# Global generator instance
_generator: Optional[AnswerGenerator] = None


def get_generator() -> AnswerGenerator:
    """Get the global answer generator instance."""
    global _generator
    if _generator is None:
        _generator = AnswerGenerator()
    return _generator
