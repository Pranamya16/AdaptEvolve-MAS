"""
adaptevolve_core.py -- Extracted from AE2_upgraded.ipynb.
All pipeline code (State, Agents, RAG, Solver, Judge, Mechanic,
LangGraph workflow, Session, Chat logic).  No Gradio dependencies.

Usage:
    import adaptevolve_core as ae
    session = ae.AdaptEvolveSession()
    result  = session.run_full_evolution("Implement a fast sorting algorithm", max_cycles=3)
"""
import os

# ── API Key resolution (env var first, then fallback) ──────────────────────
def _resolve_api_key() -> str:
    for var in ("AE2", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        val = os.environ.get(var, "")
        if val:
            return val
    return ""

API_KEY = _resolve_api_key()

import os
import json
import time
import uuid
import hashlib
import logging
import traceback
import subprocess
import tempfile
import re
import numpy as np
from datetime import datetime
from typing import TypedDict, Optional, Any, Dict, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AdaptEvolve-MAS")

from google import genai

# ── LLM backend configuration ──────────────────────────────────────────────
# AE_LLM_BACKEND selects the active backend: "ollama" (local, default) or "gemini".
# AE_LLM_MODEL overrides the local model (e.g. "qwen2.5-coder:3b-instruct" for speed).
LLM_BACKEND       = os.environ.get("AE_LLM_BACKEND", "ollama").lower()
MODEL_NAME        = "gemini-3-flash-preview"          # Gemini model id (used when backend == "gemini")
OLLAMA_MODEL_NAME = os.environ.get("AE_LLM_MODEL", "qwen2.5-coder:7b-instruct")  # local model
OLLAMA_HOST       = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
LLM_TEMPERATURE   = 0.7
LLM_MAX_TOKENS    = 8192

# Gemini client kept for the (optional) Gemini backend and runtime hot-swap in
# the Settings page.  Guarded so import never fails when no Gemini key is set.
try:
    client = genai.Client(api_key=API_KEY)
except Exception as _e:
    client = None

print(f"LLM backend: {LLM_BACKEND}  |  model: "
      f"{MODEL_NAME if LLM_BACKEND == 'gemini' else OLLAMA_MODEL_NAME}")


# Cell 3: State Definitions and Data Structures

class AgentRole(str, Enum):
    STRATEGIST = "strategist"
    SOLVER = "solver"
    JUDGE = "judge"
    MECHANIC = "mechanic"
    COORDINATOR = "coordinator"


@dataclass
class EvolutionaryOperator:
    """Represents an evolutionary operator that can be used by the Solver."""
    name: str
    description: str
    code: str
    performance_score: float = 0.0
    usage_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class EvaluationCriteria:
    """Represents the weighted evaluation criteria used by the Judge."""
    execution_time_weight: float = 0.3
    memory_usage_weight: float = 0.25
    correctness_weight: float = 0.3
    code_quality_weight: float = 0.15
    custom_weights: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "execution_time_weight": self.execution_time_weight,
            "memory_usage_weight": self.memory_usage_weight,
            "correctness_weight": self.correctness_weight,
            "code_quality_weight": self.code_quality_weight,
        }
        d.update(self.custom_weights)
        return d


@dataclass
class CandidateSolution:
    """Represents a candidate solution in the evolutionary process."""
    code: str
    generation: int
    score: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)
    parent_id: Optional[str] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class SystemState(TypedDict):
    """The central state managed by LangGraph."""
    # Human input
    human_goal: str
    uploaded_documents: List[str]

    # Strategist outputs
    research_summary: str
    research_plan: str
    current_prompt: str

    # Solver outputs
    population: List[Dict[str, Any]]
    best_solution: Dict[str, Any]
    evolutionary_operators: List[Dict[str, Any]]

    # Judge outputs
    evaluation_results: Dict[str, Any]
    holistic_score: float

    # Mechanic outputs
    mechanic_analysis: str
    proposed_operators: List[Dict[str, Any]]
    proposed_criteria: Dict[str, float]

    # Meta state
    current_cycle: int
    max_cycles: int
    evaluation_criteria: Dict[str, float]
    cycle_history: List[Dict[str, Any]]
    messages: List[Dict[str, str]]
    status: str
    active_agent: str
    should_continue: bool
    error_log: List[str]


def create_initial_state(goal: str, max_cycles: int = 5) -> SystemState:
    """Creates the initial system state from a human goal."""
    return SystemState(
        human_goal=goal,
        uploaded_documents=[],
        research_summary="",
        research_plan="",
        current_prompt="",
        population=[],
        best_solution={},
        evolutionary_operators=[
            {
                "name": "point_mutation",
                "description": "Randomly modify a single element in the code",
                "code": "default",
                "performance_score": 0.5,
                "usage_count": 0,
            },
            {
                "name": "crossover",
                "description": "Combine elements from two parent solutions",
                "code": "default",
                "performance_score": 0.5,
                "usage_count": 0,
            },
        ],
        evaluation_results={},
        holistic_score=0.0,
        mechanic_analysis="",
        proposed_operators=[],
        proposed_criteria={},
        current_cycle=0,
        max_cycles=max_cycles,
        evaluation_criteria=EvaluationCriteria().to_dict(),
        cycle_history=[],
        messages=[],
        status="initialized",
        active_agent="coordinator",
        should_continue=True,
        error_log=[],
    )

print("[OK] State definitions loaded.")

# Cell 4: LLM Wrapper (pluggable backend: Gemini or local Ollama)

def _parse_json_response(raw):
    """Shared, lenient JSON extraction used by every backend.

    Strips ```json fences, tries a direct json.loads, then falls back to the
    first {...} block.  Returns {"raw_response": ..., "parse_error": True} on
    total failure so callers can detect and recover.
    """
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r'^```(?:json)?\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
        return {"raw_response": raw, "parse_error": True}


class GeminiLLM:
    """Cloud backend — Google Gemini via the new google-genai SDK."""

    def __init__(self, model_name=None, api_key=None):
        self.model_name = model_name or MODEL_NAME
        self.client = genai.Client(api_key=api_key or API_KEY)
        self.call_count = 0

    def generate(self, prompt, system_instruction="", retries=3):
        for attempt in range(retries):
            try:
                config = genai.types.GenerateContentConfig(
                    temperature=LLM_TEMPERATURE,
                    max_output_tokens=LLM_MAX_TOKENS,
                )
                if system_instruction:
                    config.system_instruction = system_instruction

                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config,
                )
                self.call_count += 1
                return response.text
            except Exception as e:
                logger.warning("Gemini failed (attempt %d/%d): %s", attempt+1, retries, e)
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise

    def generate_json(self, prompt, system_instruction=""):
        raw = self.generate(
            prompt + "\n\nRespond ONLY with valid JSON.",
            system_instruction,
        )
        return _parse_json_response(raw)

    def stream(self, prompt, system_instruction=""):
        """Gemini doesn't stream here — yield full response as one chunk."""
        yield self.generate(prompt, system_instruction)


class OllamaLLM:
    """Local, offline backend — Ollama server (http://localhost:11434).

    Same interface as GeminiLLM so every caller (agents, RAG, evolution engine)
    works unchanged.  Uses the official `ollama` client if installed, otherwise
    falls back to a stdlib HTTP call so import never hard-fails.
    """

    def __init__(self, model_name=None, host=None):
        self.model_name = model_name or OLLAMA_MODEL_NAME
        self.host = (host or OLLAMA_HOST).rstrip("/")
        self.call_count = 0
        try:
            import ollama
            self._client = ollama.Client(host=self.host)
        except Exception:
            self._client = None  # use HTTP fallback in _chat()

    def _chat(self, prompt, system_instruction="", json_mode=False):
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})
        options = {"temperature": LLM_TEMPERATURE, "num_predict": LLM_MAX_TOKENS}

        if self._client is not None:
            kwargs = dict(model=self.model_name, messages=messages, options=options)
            if json_mode:
                kwargs["format"] = "json"
            resp = self._client.chat(**kwargs)
            return resp["message"]["content"]

        # stdlib HTTP fallback (no extra dependency required)
        import urllib.request
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": options,
        }
        if json_mode:
            payload["format"] = "json"
        req = urllib.request.Request(
            self.host + "/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=600) as r:
            data = json.loads(r.read().decode("utf-8"))
        return data["message"]["content"]

    def generate(self, prompt, system_instruction="", retries=3):
        for attempt in range(retries):
            try:
                text = self._chat(prompt, system_instruction, json_mode=False)
                self.call_count += 1
                return text
            except Exception as e:
                logger.warning("Ollama failed (attempt %d/%d): %s", attempt+1, retries, e)
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise

    def generate_json(self, prompt, system_instruction=""):
        # JSON mode greatly improves small-model reliability; fall back to plain
        # generation + lenient parsing if the server rejects format="json".
        try:
            raw = self._chat(
                prompt + "\n\nRespond ONLY with valid JSON.",
                system_instruction,
                json_mode=True,
            )
            self.call_count += 1
        except Exception:
            raw = self.generate(
                prompt + "\n\nRespond ONLY with valid JSON.",
                system_instruction,
            )
        return _parse_json_response(raw)

    def stream(self, prompt, system_instruction=""):
        """Yield text chunks for streaming. Falls back to single-chunk if unavailable."""
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})
        options = {"temperature": LLM_TEMPERATURE, "num_predict": LLM_MAX_TOKENS}

        if self._client is not None:
            try:
                for chunk in self._client.chat(
                    model=self.model_name,
                    messages=messages,
                    options=options,
                    stream=True,
                ):
                    content = (chunk.get("message") or {}).get("content", "")
                    if content:
                        yield content
                self.call_count += 1
                return
            except Exception:
                pass

        # Fallback: non-streaming, yield full response at once
        yield self.generate(prompt, system_instruction)


def build_llm(backend=None, model=None):
    """Factory: construct the active LLM backend from config/overrides."""
    backend = (backend or LLM_BACKEND).lower()
    if backend == "gemini":
        return GeminiLLM(model_name=model or MODEL_NAME)
    return OllamaLLM(model_name=model or OLLAMA_MODEL_NAME)


llm = build_llm()


# ============================================================
# Agent System Prompts (Lyra-based 4-D Methodology)
# ============================================================

STRATEGIST_SYSTEM_PROMPT = """You are Lyra-Strategist, a master-level AI research scientist and strategic planner operating within the AdaptEvolve-MAS framework. Your mission: conduct deep research, synthesize knowledge, and craft precision-optimized prompts that guide evolutionary code generation.

## THE 4-D METHODOLOGY FOR STRATEGIC PLANNING

### 1. DECONSTRUCT
- Extract the core algorithmic challenge from the human's goal
- Identify key constraints (memory, time, correctness, domain)
- Map what's known vs. what needs research
- Identify the target problem space and edge cases

### 2. DIAGNOSE
- Audit the current state of the art for this problem type
- Assess gaps in the current approach based on feedback
- Check if the evolutionary search is exploring the right solution space
- Identify if the prompt needs more specificity, constraints, or examples

### 3. DEVELOP
- Synthesize research findings into actionable strategies
- Design prompts that guide the Solver agent toward optimal solutions
- Layer context: problem definition → constraints → evaluation criteria → examples
- Apply chain-of-thought reasoning for complex algorithmic problems
- Include specific technical guidance based on research

### 4. DELIVER
- Output a structured research plan with clear milestones
- Craft prompts with: role assignment, technical context, constraints, output format
- Provide clear success metrics aligned with evaluation criteria

## OPERATING PRINCIPLES
- Always ground plans in empirical research and known algorithms
- Be specific about data structures, algorithmic techniques, and optimization strategies
- Include edge cases and failure modes in your analysis
- When refining prompts, explicitly address the weaknesses identified by the Judge
- Think like a senior researcher supervising a junior programmer

## OUTPUT FORMAT
Always structure your outputs with clear sections and actionable detail.
When generating prompts for the Solver, make them self-contained and technically precise."""


SOLVER_SYSTEM_PROMPT = """You are Lyra-Solver, a master-level AI evolutionary programmer operating within the AdaptEvolve-MAS framework. Your mission: generate, evolve, and optimize code solutions using evolutionary programming techniques.

## THE 4-D METHODOLOGY FOR CODE EVOLUTION

### 1. DECONSTRUCT
- Parse the prompt from the Strategist to extract: problem definition, constraints, optimization targets
- Identify the solution space boundaries
- Understand the fitness function (what makes a solution "good")

### 2. DIAGNOSE
- Assess the current population of solutions for diversity and quality
- Identify which evolutionary operators are most effective
- Check for premature convergence or stagnation
- Evaluate whether the search space needs expansion or narrowing

### 3. DEVELOP
- Generate code solutions that are:
  - Correct (handle all edge cases)
  - Efficient (optimized for specified metrics)
  - Clean (readable and well-structured)
  - Innovative (explore novel algorithmic approaches)
- Apply evolutionary operators: mutation, crossover, selection
- Use the prompt's guidance to focus the search

### 4. DELIVER
- Output clean, executable Python code
- Include docstrings and type hints
- Provide complexity analysis (time and space)
- Note any trade-offs made in the implementation

## OPERATING PRINCIPLES
- Always produce valid, runnable Python code
- Prioritize correctness first, then optimize for the specified metrics
- Explore diverse approaches — don't just micro-optimize one approach
- Learn from previous generations' failures
- Include test cases when possible

## OUTPUT FORMAT
Always wrap code in proper Python code blocks.
Include brief analysis of approach and expected performance."""


JUDGE_SYSTEM_PROMPT = """You are Lyra-Judge, a master-level AI code evaluator operating within the AdaptEvolve-MAS framework. Your mission: provide rigorous, multi-faceted evaluation of code solutions with actionable feedback.

## THE 4-D METHODOLOGY FOR CODE EVALUATION

### 1. DECONSTRUCT
- Extract the evaluation criteria and their weights
- Identify all testable aspects: correctness, performance, quality, security
- Map the code's claims against verifiable metrics

### 2. DIAGNOSE
- Run systematic tests across all evaluation dimensions
- Identify failure modes, edge cases, and performance bottlenecks
- Compare against known optimal solutions when applicable
- Check for common anti-patterns and vulnerabilities

### 3. DEVELOP
- Produce quantitative metrics for each evaluation dimension
- Calculate weighted holistic score
- Generate specific, actionable feedback for improvement
- Identify the highest-impact improvements

### 4. DELIVER
- Output structured evaluation with scores and justification
- Provide ranked list of improvements by expected impact
- Give clear pass/fail verdicts on each criterion
- Include concrete examples of how to improve

## OPERATING PRINCIPLES
- Be rigorous and fair — evaluate based on evidence, not assumptions
- Provide specific line-level feedback when possible
- Balance praise for strengths with honest assessment of weaknesses
- Always explain WHY something scored the way it did
- Focus feedback on actionable improvements

## OUTPUT FORMAT
Always provide structured JSON-compatible evaluation results with scores and detailed feedback."""


MECHANIC_SYSTEM_PROMPT = """You are Lyra-Mechanic, a master-level AI meta-learning systems engineer operating within the AdaptEvolve-MAS framework. Your mission: analyze the evolutionary process itself and propose improvements to make the system more effective.

## THE 4-D METHODOLOGY FOR META-OPTIMIZATION

### 1. DECONSTRUCT
- Analyze the history of evolutionary runs: scores, convergence patterns, operator effectiveness
- Extract patterns from successful vs. unsuccessful generations
- Identify bottlenecks in the optimization pipeline

### 2. DIAGNOSE
- Is the evolution converging too fast (premature convergence)?
- Is the evolution too slow (insufficient selection pressure)?
- Are the evaluation criteria properly aligned with the goal?
- Are the evolutionary operators producing useful variations?
- Is the prompt guiding the search effectively?

### 3. DEVELOP
- Design new evolutionary operators based on observed patterns
- Propose adjustments to evaluation criteria weights
- Suggest prompt modifications for better search guidance
- Recommend population management strategies

### 4. DELIVER
- Output specific, implementable improvements
- Provide new operator code as valid Python functions
- Give new criteria weights as structured data
- Include expected impact analysis for each proposal

## OPERATING PRINCIPLES
- Base all proposals on empirical evidence from run logs
- Be conservative — small, targeted changes are better than overhauls
- Always explain the reasoning behind proposals
- Consider second-order effects of changes
- Track which of your previous proposals were effective

## OUTPUT FORMAT
Provide structured proposals with: description, rationale, expected impact, and implementation details."""


print("[OK] LLM Wrapper and Prompt Templates loaded.")

# Cell 5: Agentic RAG System for the Strategist

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    _ST_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=_ST_CACHE)
    print("[OK] Sentence Transformer loaded for embeddings.")
except ImportError:
    EMBEDDING_MODEL = None
    print("[WARN] Sentence Transformers not available. Using hash-based embeddings.")


class SimpleVectorStore:
    """A simple in-memory vector store for RAG."""

    def __init__(self):
        self.documents: List[Dict[str, Any]] = []
        self.embeddings: List[np.ndarray] = []

    def _get_embedding(self, text: str) -> np.ndarray:
        if EMBEDDING_MODEL is not None:
            return EMBEDDING_MODEL.encode(text, normalize_embeddings=True)
        else:
            # Fallback: use a hash-based pseudo-embedding
            h = hashlib.md5(text.encode()).hexdigest()
            return np.array([int(c, 16) / 15.0 for c in h], dtype=np.float32)

    def add_document(self, text: str, metadata: Optional[Dict] = None):
        """Add a document to the store, splitting into chunks."""
        chunks = self._chunk_text(text, chunk_size=500, overlap=50)
        for i, chunk in enumerate(chunks):
            doc = {
                "text": chunk,
                "metadata": metadata or {},
                "chunk_index": i,
                "id": str(uuid.uuid4())[:8],
            }
            self.documents.append(doc)
            self.embeddings.append(self._get_embedding(chunk))

    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        return chunks if chunks else [text]

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for the most relevant documents."""
        if not self.documents:
            return []

        query_embedding = self._get_embedding(query)
        similarities = []

        for i, doc_embedding in enumerate(self.embeddings):
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding) + 1e-8
            )
            similarities.append((similarity, i))

        similarities.sort(reverse=True, key=lambda x: x[0])

        results = []
        for sim, idx in similarities[:top_k]:
            result = self.documents[idx].copy()
            result["similarity_score"] = float(sim)
            results.append(result)

        return results

    def get_stats(self) -> Dict:
        return {
            "total_documents": len(self.documents),
            "total_chunks": len(self.embeddings),
        }


class WebSearchTool:
    """Web search tool using DuckDuckGo."""

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
                return [
                    {
                        "title": r.get("title", ""),
                        "body": r.get("body", ""),
                        "href": r.get("href", ""),
                    }
                    for r in results
                ]
        except Exception as e:
            logger.warning(f"Web search failed: {e}")
            # Fallback: use Gemini to simulate research
            response = llm.generate(
                f"Provide a brief research summary about: {query}\n"
                f"Include key algorithms, techniques, and state-of-the-art approaches.\n"
                f"Format as a list of findings."
            )
            return [{"title": "AI-Generated Research Summary", "body": response, "href": "gemini-generated"}]


class AgenticRAG:
    """Agentic RAG system that autonomously decides how to research a topic."""

    def __init__(self):
        self.vector_store = SimpleVectorStore()
        self.web_search = WebSearchTool()
        self.research_history: List[Dict] = []

    def ingest_document(self, text: str, source: str = "uploaded"):
        """Ingest a document into the vector store."""
        self.vector_store.add_document(text, metadata={"source": source})
        logger.info(f"Ingested document from {source}: {len(text)} chars")

    def conduct_research(self, topic: str, goal: str) -> Dict[str, Any]:
        """
        Agentic multi-step research process.
        The LLM decides which sub-tools to use and how to synthesize results.
        """
        research_result = {
            "topic": topic,
            "goal": goal,
            "internal_findings": [],
            "external_findings": [],
            "synthesis": "",
            "recommendations": [],
            "timestamp": datetime.now().isoformat(),
        }

        # Step 1: Plan the research
        planning_prompt = f"""You are a research scientist planning a literature review.

TOPIC: {topic}
GOAL: {goal}

Generate 3-5 specific search queries that would help understand:
1. State-of-the-art algorithms and techniques for this problem
2. Known optimizations and trade-offs
3. Common pitfalls and edge cases
4. Benchmark results and comparisons

Output as a JSON list of strings (search queries only)."""

        queries = llm.generate_json(planning_prompt)
        if isinstance(queries, dict) and "raw_response" in queries:
            queries = [topic, f"{topic} algorithm optimization", f"{topic} state of the art"]
        elif isinstance(queries, dict):
            queries = list(queries.values()) if queries else [topic]

        if not isinstance(queries, list):
            queries = [topic]

        logger.info(f"Research queries generated: {queries}")

        # Step 2: Search internal documents
        for query in queries[:3]:
            internal_results = self.vector_store.search(query, top_k=3)
            for r in internal_results:
                if r["similarity_score"] > 0.3:
                    research_result["internal_findings"].append({
                        "query": query,
                        "text": r["text"][:500],
                        "score": r["similarity_score"],
                    })

        # Step 3: Search the web
        for query in queries[:3]:
            web_results = self.web_search.search(f"{topic} {query}", max_results=3)
            for r in web_results:
                research_result["external_findings"].append({
                    "query": query,
                    "title": r["title"],
                    "body": r["body"][:500],
                })
            time.sleep(0.5)  # Rate limiting

        # Step 4: Synthesize findings
        synthesis_prompt = f"""You are synthesizing research findings for an evolutionary code optimization system.

GOAL: {goal}

INTERNAL DOCUMENT FINDINGS:
{json.dumps(research_result['internal_findings'][:5], indent=2)}

EXTERNAL WEB FINDINGS:
{json.dumps(research_result['external_findings'][:5], indent=2)}

Provide a comprehensive synthesis that includes:
1. **State of the Art**: What are the best known approaches?
2. **Key Techniques**: What specific algorithms/data structures should be considered?
3. **Optimization Opportunities**: Where can we improve over standard approaches?
4. **Edge Cases & Pitfalls**: What should we watch out for?
5. **Recommended Approach**: What strategy should we pursue for evolutionary optimization?

Be specific and technical. Include complexity analysis where relevant."""

        synthesis = llm.generate(synthesis_prompt, STRATEGIST_SYSTEM_PROMPT)
        research_result["synthesis"] = synthesis

        # Step 5: Generate specific recommendations
        rec_prompt = f"""Based on this research synthesis, provide 5 specific actionable recommendations
for generating optimized code for this goal: {goal}

Research synthesis:
{synthesis[:2000]}

Output as a JSON list of objects with 'recommendation' and 'rationale' keys."""

        recommendations = llm.generate_json(rec_prompt)
        if isinstance(recommendations, list):
            research_result["recommendations"] = recommendations
        elif isinstance(recommendations, dict) and not recommendations.get("parse_error"):
            research_result["recommendations"] = [recommendations]

        self.research_history.append(research_result)
        return research_result


# Initialize global RAG instance
rag_system = AgenticRAG()

print("[OK] Agentic RAG System loaded.")
print(f"   Vector Store: {rag_system.vector_store.get_stats()}")

# Cell 6: OpenEvolve Integration / Evolutionary Engine (FIXED)

class OpenEvolveWrapper:
    """
    Wrapper around OpenEvolve or a fallback evolutionary engine.
    Uses Gemini as the LLM backbone for code evolution.
    """

    def __init__(self, llm_instance: GeminiLLM):
        self.llm = llm_instance
        self.generation_count = 0
        self.openevolve_available = False

        # Try to import OpenEvolve
        try:
            import openevolve
            self.openevolve_available = True
            logger.info("OpenEvolve library detected and available.")
        except ImportError:
            logger.info("OpenEvolve not available. Using built-in evolutionary engine.")

    def run_evolution(
        self,
        prompt: str,
        population: List[Dict[str, Any]],
        operators: List[Dict[str, Any]],
        num_generations: int = 3,
        population_size: int = 5,
    ) -> Dict[str, Any]:
        """
        Run evolutionary optimization on code solutions.

        Returns:
            Dict with 'best_solution', 'population', 'logs'
        """
        if self.openevolve_available:
            return self._run_with_openevolve(
                prompt, population, operators, num_generations, population_size
            )
        else:
            return self._run_builtin_evolution(
                prompt, population, operators, num_generations, population_size
            )

    def _run_with_openevolve(self, prompt, population, operators, num_generations, population_size):
        """Run using the actual OpenEvolve library."""
        try:
            import openevolve

            with tempfile.TemporaryDirectory() as tmpdir:
                initial_code = ""
                if population:
                    initial_code = population[0].get("code", "")
                if not initial_code:
                    initial_code = self._generate_initial_solution(prompt)

                program_path = os.path.join(tmpdir, "program.py")
                with open(program_path, "w") as f:
                    f.write(initial_code)

                eval_code = self._generate_eval_function(prompt)
                eval_path = os.path.join(tmpdir, "evaluator.py")
                with open(eval_path, "w") as f:
                    f.write(eval_code)

                config = openevolve.OpenEvolveConfig(
                    num_islands=2,
                    population_size=population_size,
                    num_generations=num_generations,
                )

                evolver = openevolve.OpenEvolve(
                    initial_program_path=program_path,
                    evaluation_file=eval_path,
                    config=config,
                )

                best_program = evolver.run()

                return {
                    "best_solution": {
                        "code": best_program.code if hasattr(best_program, "code") else str(best_program),
                        "score": best_program.score if hasattr(best_program, "score") else 0.0,
                        "generation": num_generations,
                        "id": str(uuid.uuid4())[:8],
                    },
                    "population": population,
                    "logs": {"method": "openevolve", "generations": num_generations},
                }

        except Exception as e:
            logger.warning(
                "OpenEvolve execution failed: %s. Falling back to built-in engine.", e
            )
            return self._run_builtin_evolution(
                prompt, population, operators, num_generations, population_size
            )

    def _run_builtin_evolution(
        self, prompt, population, operators, num_generations, population_size
    ):
        """Built-in evolutionary engine using Gemini for code generation and mutation."""
        logs = {"generations": [], "method": "builtin_gemini_evolution"}
        current_population = list(population) if population else []

        # Initialize population if empty
        if len(current_population) < population_size:
            needed = population_size - len(current_population)
            for i in range(needed):
                code = self._generate_initial_solution(prompt, variant=i)
                candidate = {
                    "code": code,
                    "score": 0.0,
                    "generation": 0,
                    "id": str(uuid.uuid4())[:8],
                    "parent_id": None,
                }
                current_population.append(candidate)

        # Evolutionary loop
        for gen in range(1, num_generations + 1):
            self.generation_count += 1
            gen_log = {"generation": gen, "operations": [], "best_score": 0.0}

            new_candidates = []

            for op in operators:
                op_name = op.get("name", "mutation")

                if op_name == "crossover" and len(current_population) >= 2:
                    sorted_pop = sorted(
                        current_population,
                        key=lambda x: x.get("score", 0),
                        reverse=True,
                    )
                    parent1 = sorted_pop[0]
                    parent2 = sorted_pop[min(1, len(sorted_pop) - 1)]
                    child_code = self._crossover(prompt, parent1["code"], parent2["code"])
                    new_candidates.append(
                        {
                            "code": child_code,
                            "score": 0.0,
                            "generation": gen,
                            "id": str(uuid.uuid4())[:8],
                            "parent_id": parent1["id"] + "+" + parent2["id"],
                            "operator": "crossover",
                        }
                    )
                    gen_log["operations"].append("crossover")

                elif current_population:
                    sorted_pop = sorted(
                        current_population,
                        key=lambda x: x.get("score", 0),
                        reverse=True,
                    )
                    parent = sorted_pop[0]
                    mutated_code = self._mutate(
                        prompt, parent["code"], op.get("description", "improve the code")
                    )
                    new_candidates.append(
                        {
                            "code": mutated_code,
                            "score": 0.0,
                            "generation": gen,
                            "id": str(uuid.uuid4())[:8],
                            "parent_id": parent["id"],
                            "operator": op_name,
                        }
                    )
                    gen_log["operations"].append(op_name)

            current_population.extend(new_candidates)

            # Quick evaluation
            for candidate in current_population:
                if candidate.get("score", 0) == 0:
                    candidate["score"] = self._quick_evaluate(candidate["code"])

            # Selection: keep top population_size
            current_population.sort(key=lambda x: x.get("score", 0), reverse=True)
            current_population = current_population[:population_size]

            gen_log["best_score"] = (
                current_population[0].get("score", 0) if current_population else 0
            )
            gen_log["population_size"] = len(current_population)
            logs["generations"].append(gen_log)

        best = (
            max(current_population, key=lambda x: x.get("score", 0))
            if current_population
            else {"code": "", "score": 0}
        )

        return {
            "best_solution": best,
            "population": current_population,
            "logs": logs,
        }

    def _generate_initial_solution(self, prompt: str, variant: int = 0) -> str:
        """Generate an initial solution using the LLM."""
        if variant == 0:
            focus = "Focus on readability"
        elif variant == 1:
            focus = "Focus on optimization"
        else:
            focus = "Focus on an unconventional approach"

        if variant > 0:
            approach = "Use a different algorithmic approach than the standard one."
        else:
            approach = "Use the most straightforward correct approach."

        gen_prompt = (
            "Generate a Python solution for the following problem.\n"
            + approach
            + "\n"
            + "Variant #"
            + str(variant)
            + ": "
            + focus
            + "\n\n"
            + "PROBLEM:\n"
            + prompt
            + "\n\n"
            + "Requirements:\n"
            + "- The code must be a complete, self-contained Python function\n"
            + "- Include type hints and docstrings\n"
            + "- Handle edge cases\n"
            + "- The main function should be clearly identifiable\n\n"
            + "Output ONLY the Python code, no explanations."
        )

        response = self.llm.generate(gen_prompt, SOLVER_SYSTEM_PROMPT)
        return self._extract_code(response)

    def _mutate(self, prompt: str, code: str, mutation_description: str) -> str:
        """Mutate a solution using the LLM."""
        code_fence = "```"

        mut_prompt = (
            "You are an evolutionary code optimizer. "
            "Mutate the following code to improve it.\n\n"
            "ORIGINAL PROBLEM:\n"
            + prompt
            + "\n\n"
            + "CURRENT CODE:\n"
            + code_fence
            + "python\n"
            + code
            + "\n"
            + code_fence
            + "\n\n"
            + "MUTATION TYPE: "
            + mutation_description
            + "\n\n"
            + "Instructions:\n"
            + "- Make targeted improvements while maintaining correctness\n"
            + "- Focus on: performance, readability, edge case handling\n"
            + "- Keep the same function signature\n"
            + "- The mutation should be meaningful but not a complete rewrite\n\n"
            + "Output ONLY the improved Python code."
        )

        response = self.llm.generate(mut_prompt, SOLVER_SYSTEM_PROMPT)
        return self._extract_code(response)

    def _crossover(self, prompt: str, code1: str, code2: str) -> str:
        """Crossover two solutions using the LLM."""
        code_fence = "```"

        cross_prompt = (
            "You are an evolutionary code optimizer. "
            "Combine the best aspects of two solutions.\n\n"
            "PROBLEM:\n"
            + prompt
            + "\n\n"
            + "SOLUTION A:\n"
            + code_fence
            + "python\n"
            + code1
            + "\n"
            + code_fence
            + "\n\n"
            + "SOLUTION B:\n"
            + code_fence
            + "python\n"
            + code2
            + "\n"
            + code_fence
            + "\n\n"
            + "Instructions:\n"
            + "- Combine the best algorithmic ideas from both solutions\n"
            + "- The result should be better than either parent\n"
            + "- Maintain correctness and handle edge cases\n"
            + "- Keep a clean, well-structured implementation\n\n"
            + "Output ONLY the combined Python code."
        )

        response = self.llm.generate(cross_prompt, SOLVER_SYSTEM_PROMPT)
        return self._extract_code(response)

    def _quick_evaluate(self, code: str) -> float:
        """Quick evaluation: check if code is syntactically valid and runs."""
        score = 0.0
        try:
            compile(code, "<string>", "exec")
            score += 0.3  # Syntactically valid
        except SyntaxError:
            return 0.0

        # Check for basic quality indicators
        if "def " in code:
            score += 0.1
        if '"""' in code or "'''" in code:
            score += 0.1  # Has docstrings
        if "return" in code:
            score += 0.1
        if "if " in code:
            score += 0.05  # Has conditionals
        if "->" in code:
            score += 0.05  # Has type hints

        # Try to run it in a sandbox
        try:
            exec_globals = {}
            exec(code, exec_globals)
            score += 0.3  # Runs without error
        except Exception:
            pass

        return min(score, 1.0)

    def _extract_code(self, response: str) -> str:
        """Extract Python code from an LLM response."""
        # Try to extract from code blocks
        code_match = re.search(r"```python\s*(.*?)```", response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()

        code_match = re.search(r"```\s*(.*?)```", response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()

        # If no code blocks, check if it looks like code
        lines = response.strip().split("\n")
        if any("def " in l or "class " in l or "import " in l for l in lines):
            return response.strip()

        return response.strip()

    def _generate_eval_function(self, prompt: str) -> str:
        """Generate an evaluation function for OpenEvolve."""
        eval_prompt = (
            "Create a Python evaluation function for the following problem "
            "that returns a numerical score.\n\n"
            "PROBLEM: "
            + prompt
            + "\n\n"
            + "The function should:\n"
            + "1. Be named 'evaluate'\n"
            + "2. Take the solution module as input\n"
            + "3. Test correctness with multiple test cases\n"
            + "4. Measure performance\n"
            + "5. Return a float score between 0.0 and 1.0\n\n"
            + "Output ONLY the Python code."
        )

        response = self.llm.generate(eval_prompt, SOLVER_SYSTEM_PROMPT)
        return self._extract_code(response)


# Initialize OpenEvolve wrapper
evolve_engine = OpenEvolveWrapper(llm)
print(
    "[OK] Evolutionary Engine loaded. OpenEvolve available: "
    + str(evolve_engine.openevolve_available)
)

## Cell 7: Judge Tools (Benchmark, Static Analysis, Scoring)


# Cell 7: Judge Agent Tools

class BenchmarkRunner:
    """Runs performance benchmarks on candidate code."""

    def run_performance_benchmark(self, code: str, test_cases: Optional[List] = None) -> Dict[str, Any]:
        """Execute code and measure performance metrics."""
        results = {
            "execution_time_ms": None,
            "memory_usage_kb": None,
            "correctness": False,
            "test_results": [],
            "error": None,
        }

        try:
            # Compile check
            compiled = compile(code, '<string>', 'exec')

            # Create sandbox
            sandbox = {"__builtins__": __builtins__}

            # Execute to define functions
            exec(compiled, sandbox)

            # Find the main function
            main_func = None
            for name, obj in sandbox.items():
                if callable(obj) and not name.startswith('_') and name not in dir(__builtins__):
                    main_func = obj
                    break

            if main_func is None:
                results["error"] = "No callable function found in code"
                return results

            # Run with timing
            import time as time_mod

            # Generate test cases if none provided
            if not test_cases:
                test_cases = self._generate_test_cases(main_func)

            passed = 0
            total = len(test_cases)
            execution_times = []

            for i, tc in enumerate(test_cases):
                try:
                    args = tc.get("input", [])
                    if not isinstance(args, (list, tuple)):
                        args = [args]

                    start = time_mod.perf_counter()
                    result = main_func(*args)
                    elapsed = (time_mod.perf_counter() - start) * 1000  # ms

                    execution_times.append(elapsed)

                    expected = tc.get("expected", None)
                    test_passed = True
                    if expected is not None:
                        test_passed = result == expected

                    if test_passed:
                        passed += 1

                    results["test_results"].append({
                        "test_index": i,
                        "passed": test_passed,
                        "execution_time_ms": round(elapsed, 4),
                        "result": str(result)[:100],
                    })
                except Exception as e:
                    results["test_results"].append({
                        "test_index": i,
                        "passed": False,
                        "error": str(e),
                    })

            results["correctness"] = passed == total if total > 0 else False
            results["correctness_ratio"] = passed / total if total > 0 else 0
            results["execution_time_ms"] = round(np.mean(execution_times), 4) if execution_times else None
            results["max_execution_time_ms"] = round(max(execution_times), 4) if execution_times else None

            # Rough memory estimate
            import sys
            results["memory_usage_kb"] = round(sys.getsizeof(code) / 1024, 2)

        except SyntaxError as e:
            results["error"] = f"Syntax error: {e}"
        except Exception as e:
            results["error"] = f"Execution error: {e}"

        return results

    def _generate_test_cases(self, func) -> List[Dict]:
        """Generate basic test cases based on function inspection."""
        import inspect
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())

        # Generate generic test cases
        test_cases = []
        if params:
            # Try with empty list
            test_cases.append({"input": [[]]})
            # Try with small list
            test_cases.append({"input": [[1, 2, 3]]})
            # Try with duplicates
            test_cases.append({"input": [[3, 1, 2, 1, 3]]})
            # Try with sorted
            test_cases.append({"input": [[1, 2, 3, 4, 5]]})
            # Try with reverse sorted
            test_cases.append({"input": [[5, 4, 3, 2, 1]]})

        return test_cases


class StaticAnalyzer:
    """Runs static analysis on code."""

    def run_static_analysis(self, code: str) -> Dict[str, Any]:
        """Analyze code quality using static analysis."""
        results = {
            "issues": [],
            "quality_score": 1.0,
            "metrics": {},
        }

        # Check basic code metrics
        lines = code.split('\n')
        results["metrics"]["total_lines"] = len(lines)
        results["metrics"]["blank_lines"] = sum(1 for l in lines if not l.strip())
        results["metrics"]["comment_lines"] = sum(1 for l in lines if l.strip().startswith('#'))
        results["metrics"]["code_lines"] = results["metrics"]["total_lines"] - results["metrics"]["blank_lines"] - results["metrics"]["comment_lines"]

        # Check for common issues
        deductions = 0.0

        # No docstring
        if '"""' not in code and "'''" not in code:
            results["issues"].append({"severity": "warning", "message": "No docstring found"})
            deductions += 0.05

        # No type hints
        if '->' not in code and ': ' not in code.split('def')[1] if 'def ' in code else True:
            results["issues"].append({"severity": "info", "message": "No type hints found"})
            deductions += 0.03

        # Very long functions
        func_lengths = []
        in_func = False
        func_lines = 0
        for line in lines:
            if line.strip().startswith('def '):
                if in_func and func_lines > 50:
                    results["issues"].append({"severity": "warning", "message": f"Function is too long ({func_lines} lines)"})
                    deductions += 0.05
                in_func = True
                func_lines = 0
            elif in_func:
                func_lines += 1

        # Use of global variables
        if 'global ' in code:
            results["issues"].append({"severity": "warning", "message": "Uses global variables"})
            deductions += 0.05

        # Bare except
        if 'except:' in code:
            results["issues"].append({"severity": "error", "message": "Bare except clause found"})
            deductions += 0.1

        # Try pylint if available
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                f.flush()
                result = subprocess.run(
                    ['python', '-m', 'pylint', f.name, '--score=yes', '--disable=all', '--enable=E'],
                    capture_output=True, text=True, timeout=10,
                )
                if 'rated at' in result.stdout:
                    match = re.search(r'rated at ([\d.]+)', result.stdout)
                    if match:
                        pylint_score = float(match.group(1)) / 10.0
                        results["metrics"]["pylint_score"] = pylint_score
                os.unlink(f.name)
        except Exception:
            pass

        # Calculate complexity (simple approximation)
        nesting_depth = 0
        max_nesting = 0
        for line in lines:
            stripped = line.strip()
            if any(stripped.startswith(kw) for kw in ['if ', 'for ', 'while ', 'try:', 'with ']):
                nesting_depth += 1
                max_nesting = max(max_nesting, nesting_depth)
            elif stripped in ('', 'else:', 'elif ', 'except:', 'finally:'):
                pass
            elif not stripped.startswith((' ', '\t')) and stripped:
                nesting_depth = 0

        results["metrics"]["max_nesting_depth"] = max_nesting
        if max_nesting > 5:
            results["issues"].append({"severity": "warning", "message": f"High nesting depth: {max_nesting}"})
            deductions += 0.05

        results["quality_score"] = max(0.0, 1.0 - deductions)
        return results


class HolisticScorer:
    """Calculates a weighted holistic score from multiple evaluation dimensions."""

    def calculate_holistic_score(self, metrics: Dict[str, Any], criteria: Dict[str, float]) -> Dict[str, Any]:
        """Calculate the final weighted score."""
        scores = {}

        # Execution time score (lower is better)
        exec_time = metrics.get("execution_time_ms")
        if exec_time is not None:
            if exec_time < 1:
                scores["execution_time"] = 1.0
            elif exec_time < 10:
                scores["execution_time"] = 0.9
            elif exec_time < 100:
                scores["execution_time"] = 0.7
            elif exec_time < 1000:
                scores["execution_time"] = 0.4
            else:
                scores["execution_time"] = 0.1
        else:
            scores["execution_time"] = 0.0

        # Memory usage score
        mem = metrics.get("memory_usage_kb", 0)
        if mem < 10:
            scores["memory_usage"] = 1.0
        elif mem < 100:
            scores["memory_usage"] = 0.8
        else:
            scores["memory_usage"] = 0.5

        # Correctness score
        scores["correctness"] = metrics.get("correctness_ratio", 0.0)

        # Code quality score
        scores["code_quality"] = metrics.get("quality_score", 0.5)

        # Calculate weighted score
        weighted_score = 0.0
        total_weight = 0.0

        weight_mapping = {
            "execution_time": criteria.get("execution_time_weight", 0.3),
            "memory_usage": criteria.get("memory_usage_weight", 0.25),
            "correctness": criteria.get("correctness_weight", 0.3),
            "code_quality": criteria.get("code_quality_weight", 0.15),
        }

        for dimension, score in scores.items():
            weight = weight_mapping.get(dimension, 0.1)
            weighted_score += score * weight
            total_weight += weight

        if total_weight > 0:
            weighted_score /= total_weight

        return {
            "holistic_score": round(weighted_score, 4),
            "dimension_scores": scores,
            "weights_used": weight_mapping,
            "raw_metrics": metrics,
        }


# Initialize Judge tools
benchmark_runner = BenchmarkRunner()
static_analyzer = StaticAnalyzer()
holistic_scorer = HolisticScorer()

print("[OK] Judge tools loaded (Benchmark, Static Analysis, Holistic Scorer)")

# Cell 8: Agent Node Functions for LangGraph (ALL 4 AGENTS IN ONE CELL)

def strategist_node(state: SystemState) -> Dict:
    """The Strategist Agent: Research, plan, and craft prompts."""
    logger.info("Strategist Agent activated")

    messages = list(state.get("messages", []))
    messages.append({"role": "system", "content": "Strategist Agent is conducting research and planning..."})

    cycle = state.get("current_cycle", 0)
    goal = state["human_goal"]

    if cycle == 0:
        research_result = rag_system.conduct_research(
            topic=goal,
            goal=goal,
        )

        research_summary = research_result.get("synthesis", "No research findings available.")
        recommendations = research_result.get("recommendations", [])

        plan_prompt = (
            "Create a detailed research and optimization plan for the following goal:\n\n"
            "GOAL: " + goal + "\n\n"
            "RESEARCH FINDINGS:\n"
            + research_summary[:3000] + "\n\n"
            "RECOMMENDATIONS:\n"
            + json.dumps(recommendations[:5], indent=2) + "\n\n"
            "Create:\n"
            "1. A structured research plan with specific milestones\n"
            "2. An initial prompt for the evolutionary Solver agent\n\n"
            "The Solver prompt should be highly specific and include:\n"
            "- Clear problem definition with input/output specifications\n"
            "- Performance constraints and optimization targets\n"
            "- Edge cases to handle\n"
            "- Preferred algorithmic approaches based on research\n"
            "- Code quality requirements"
        )

        plan = llm.generate(plan_prompt, STRATEGIST_SYSTEM_PROMPT)

        solver_prompt_gen = (
            "Based on this plan, create a precise, self-contained prompt for an AI code generator "
            "that will create a Python solution for: " + goal + "\n\n"
            "PLAN:\n" + plan[:2000] + "\n\n"
            "RESEARCH:\n" + research_summary[:1500] + "\n\n"
            "The prompt should be ready to use directly with a code-generating AI.\n"
            "Include specific function signatures, test cases, and optimization criteria.\n"
            "Output ONLY the prompt text."
        )

        solver_prompt = llm.generate(solver_prompt_gen, STRATEGIST_SYSTEM_PROMPT)

        messages.append({"role": "strategist", "content": "Research complete. Plan created. Initial prompt generated."})

        return {
            "research_summary": research_summary,
            "research_plan": plan,
            "current_prompt": solver_prompt,
            "messages": messages,
            "active_agent": "solver",
            "status": "research_complete",
        }

    else:
        eval_results = state.get("evaluation_results", {})
        mechanic_analysis = state.get("mechanic_analysis", "")
        previous_prompt = state.get("current_prompt", "")
        best_solution = state.get("best_solution", {})

        refine_prompt = (
            "Refine the prompt for the evolutionary Solver based on feedback from the previous cycle.\n\n"
            "ORIGINAL GOAL: " + goal + "\n\n"
            "PREVIOUS PROMPT:\n" + previous_prompt[:2000] + "\n\n"
            "EVALUATION RESULTS:\n" + json.dumps(eval_results, indent=2, default=str)[:2000] + "\n\n"
            "BEST SOLUTION SCORE: " + str(best_solution.get("score", "N/A")) + "\n\n"
            "MECHANIC'S ANALYSIS:\n" + mechanic_analysis[:1000] + "\n\n"
            "CYCLE: " + str(cycle) + "\n\n"
            "Based on this feedback:\n"
            "1. Identify what aspects need improvement\n"
            "2. Add more specific guidance where the solution was weak\n"
            "3. Adjust the optimization priorities\n"
            "4. Include any new techniques or approaches suggested by the Mechanic\n\n"
            "Output the refined prompt for the Solver."
        )

        refined = llm.generate(refine_prompt, STRATEGIST_SYSTEM_PROMPT)

        messages.append({"role": "strategist", "content": "Prompt refined for cycle " + str(cycle) + " based on evaluation feedback."})

        return {
            "current_prompt": refined,
            "messages": messages,
            "active_agent": "solver",
            "status": "prompt_refined_cycle_" + str(cycle),
        }


def solver_node(state: SystemState) -> Dict:
    """The Solver Agent: Run evolutionary optimization."""
    logger.info("Solver Agent activated")

    messages = list(state.get("messages", []))
    messages.append({"role": "system", "content": "Solver Agent is running evolutionary optimization..."})

    prompt = state["current_prompt"]
    population = state.get("population", [])
    operators = state.get("evolutionary_operators", [])

    result = evolve_engine.run_evolution(
        prompt=prompt,
        population=population,
        operators=operators,
        num_generations=3,
        population_size=5,
    )

    best = result["best_solution"]
    new_population = result["population"]

    messages.append({
        "role": "solver",
        "content": (
            "Evolution complete. Best score: " + str(best.get("score", "N/A"))
            + ". Population size: " + str(len(new_population))
            + ". Method: " + str(result["logs"].get("method", "unknown"))
        ),
    })

    return {
        "best_solution": best,
        "population": new_population,
        "messages": messages,
        "active_agent": "judge",
        "status": "evolution_complete",
    }


def judge_node(state: SystemState) -> Dict:
    """The Judge Agent: Evaluate the candidate solution."""
    logger.info("Judge Agent activated")

    messages = list(state.get("messages", []))
    messages.append({"role": "system", "content": "Judge Agent is evaluating the solution..."})

    best_solution = state.get("best_solution", {})
    code = best_solution.get("code", "")
    criteria = state.get("evaluation_criteria", EvaluationCriteria().to_dict())

    if not code:
        messages.append({"role": "judge", "content": "No code to evaluate."})
        return {
            "evaluation_results": {"error": "No code provided"},
            "holistic_score": 0.0,
            "messages": messages,
            "active_agent": "mechanic",
            "status": "evaluation_failed",
        }

    benchmark_results = benchmark_runner.run_performance_benchmark(code)
    static_results = static_analyzer.run_static_analysis(code)

    combined_metrics = {
        "execution_time_ms": benchmark_results.get("execution_time_ms"),
        "memory_usage_kb": benchmark_results.get("memory_usage_kb", 0),
        "correctness_ratio": benchmark_results.get("correctness_ratio", 0),
        "quality_score": static_results.get("quality_score", 0.5),
        "benchmark_details": benchmark_results,
        "static_analysis": static_results,
    }

    score_result = holistic_scorer.calculate_holistic_score(combined_metrics, criteria)

    code_fence = "```"
    feedback_prompt = (
        "As an expert code reviewer, provide detailed feedback on this solution.\n\n"
        "CODE:\n" + code_fence + "python\n" + code[:3000] + "\n" + code_fence + "\n\n"
        "BENCHMARK RESULTS:\n"
        + json.dumps(benchmark_results, indent=2, default=str)[:1000] + "\n\n"
        "STATIC ANALYSIS:\n"
        + json.dumps(static_results, indent=2, default=str)[:1000] + "\n\n"
        "HOLISTIC SCORE: " + str(score_result["holistic_score"]) + "\n"
        "DIMENSION SCORES: " + json.dumps(score_result["dimension_scores"], indent=2) + "\n\n"
        "Provide:\n"
        "1. Top 3 strengths\n"
        "2. Top 3 weaknesses\n"
        "3. Specific improvement suggestions (with code examples if possible)\n"
        "4. Overall assessment\n\n"
        "Be concise but specific."
    )

    feedback = llm.generate(feedback_prompt, JUDGE_SYSTEM_PROMPT)

    evaluation_results = {
        "score": score_result,
        "benchmark": benchmark_results,
        "static_analysis": static_results,
        "feedback": feedback,
        "cycle": state.get("current_cycle", 0),
    }

    holistic_val = score_result["holistic_score"]
    correctness_val = combined_metrics["correctness_ratio"]
    quality_val = combined_metrics["quality_score"]

    messages.append({
        "role": "judge",
        "content": (
            "Evaluation complete. Holistic score: "
            + "{:.4f}".format(holistic_val)
            + ". Correctness: "
            + "{:.0%}".format(correctness_val)
            + ". Quality: "
            + "{:.2f}".format(quality_val)
        ),
    })

    updated_best = dict(best_solution)
    updated_best["score"] = score_result["holistic_score"]
    updated_best["metrics"] = combined_metrics

    return {
        "evaluation_results": evaluation_results,
        "holistic_score": score_result["holistic_score"],
        "best_solution": updated_best,
        "messages": messages,
        "active_agent": "mechanic",
        "status": "evaluation_complete",
    }


def mechanic_node(state: SystemState) -> Dict:
    """The Mechanic Agent: Analyze and improve the evolutionary process."""
    logger.info("Mechanic Agent activated")

    messages = list(state.get("messages", []))
    messages.append({"role": "system", "content": "Mechanic Agent is analyzing the evolutionary process..."})

    cycle = state.get("current_cycle", 0)
    eval_results = state.get("evaluation_results", {})
    cycle_history = list(state.get("cycle_history", []))
    operators = state.get("evolutionary_operators", [])
    criteria = state.get("evaluation_criteria", {})

    cycle_record = {
        "cycle": cycle,
        "holistic_score": state.get("holistic_score", 0),
        "evaluation_summary": {
            k: v for k, v in eval_results.get("score", {}).get("dimension_scores", {}).items()
        },
        "population_size": len(state.get("population", [])),
        "timestamp": datetime.now().isoformat(),
    }
    cycle_history.append(cycle_record)

    analysis_prompt = (
        "Analyze the evolutionary optimization process and propose improvements.\n\n"
        "CURRENT CYCLE: " + str(cycle) + "\n"
        "CURRENT SCORE: " + str(state.get("holistic_score", 0)) + "\n\n"
        "CYCLE HISTORY:\n" + json.dumps(cycle_history[-5:], indent=2, default=str) + "\n\n"
        "CURRENT OPERATORS:\n" + json.dumps(operators, indent=2, default=str)[:1000] + "\n\n"
        "CURRENT EVALUATION CRITERIA:\n" + json.dumps(criteria, indent=2) + "\n\n"
        "EVALUATION FEEDBACK:\n" + eval_results.get("feedback", "No feedback available")[:1500] + "\n\n"
        "Analyze:\n"
        "1. Is the evolution converging? Too fast or too slow?\n"
        "2. Which operators are most effective?\n"
        "3. Are the evaluation criteria well-balanced?\n"
        "4. What specific changes would most improve the next cycle?\n\n"
        "Provide your analysis and specific proposals."
    )

    analysis = llm.generate(analysis_prompt, MECHANIC_SYSTEM_PROMPT)

    operator_prompt = (
        "Based on this analysis, propose ONE new evolutionary operator as a Python concept.\n\n"
        "ANALYSIS:\n" + analysis[:1500] + "\n\n"
        "CURRENT OPERATORS: " + str([op["name"] for op in operators]) + "\n\n"
        "Propose a new operator with:\n"
        "1. name: A descriptive name\n"
        "2. description: What it does\n"
        "3. rationale: Why it would help\n\n"
        "Output as JSON with keys: name, description, rationale"
    )

    proposed_op = llm.generate_json(operator_prompt)
    proposed_operators = []
    if isinstance(proposed_op, dict) and not proposed_op.get("parse_error"):
        proposed_operators.append({
            "name": proposed_op.get("name", "new_operator"),
            "description": proposed_op.get("description", "Proposed improvement"),
            "code": "proposed",
            "performance_score": 0.0,
            "usage_count": 0,
        })

    criteria_prompt = (
        "Based on this analysis, propose adjusted evaluation criteria weights.\n\n"
        "CURRENT WEIGHTS:\n" + json.dumps(criteria, indent=2) + "\n\n"
        "RECENT SCORES:\n"
        + json.dumps([ch.get("evaluation_summary", {}) for ch in cycle_history[-3:]], indent=2, default=str) + "\n\n"
        "ANALYSIS:\n" + analysis[:1000] + "\n\n"
        "Output as JSON with the same keys as the current weights, with adjusted float values that sum to approximately 1.0."
    )

    proposed_criteria = llm.generate_json(criteria_prompt)
    if isinstance(proposed_criteria, dict) and proposed_criteria.get("parse_error"):
        proposed_criteria = criteria

    should_continue = True
    if cycle >= state.get("max_cycles", 5) - 1:
        should_continue = False
    elif len(cycle_history) >= 3:
        recent_scores = [ch.get("holistic_score", 0) for ch in cycle_history[-3:]]
        if len(recent_scores) >= 3 and all(abs(recent_scores[-1] - s) < 0.01 for s in recent_scores[-2:]):
            should_continue = False
            analysis += "\n\nEvolution has converged. Recommending stop."

    updated_operators = list(operators)
    for op in proposed_operators:
        if op["name"] not in [o["name"] for o in updated_operators]:
            updated_operators.append(op)

    messages.append({
        "role": "mechanic",
        "content": (
            "Analysis complete. "
            + ("Proposing process improvements. " if proposed_operators else "No new operators proposed. ")
            + "Continue: " + str(should_continue) + ". "
            + ("Criteria adjusted." if proposed_criteria != criteria else "Criteria unchanged.")
        ),
    })

    return {
        "mechanic_analysis": analysis,
        "proposed_operators": proposed_operators,
        "proposed_criteria": proposed_criteria if isinstance(proposed_criteria, dict) and not proposed_criteria.get("parse_error") else criteria,
        "evolutionary_operators": updated_operators,
        "evaluation_criteria": proposed_criteria if isinstance(proposed_criteria, dict) and not proposed_criteria.get("parse_error") else criteria,
        "cycle_history": cycle_history,
        "current_cycle": cycle + 1,
        "should_continue": should_continue,
        "messages": messages,
        "active_agent": "coordinator",
        "status": "cycle_" + str(cycle) + "_complete",
    }


print("All 4 agent node functions defined successfully.")




## Cell 9: LangGraph Workflow Construction

# Cell 9: LangGraph Workflow Construction

from langgraph.graph import StateGraph, END

def should_continue_evolution(state: SystemState) -> str:
    """Determine if the evolutionary loop should continue."""
    if not state.get("should_continue", True):
        return "end"
    if state.get("current_cycle", 0) >= state.get("max_cycles", 5):
        return "end"
    return "continue"


def build_evolution_graph() -> StateGraph:
    """Build the LangGraph workflow for the AdaptEvolve-MAS system."""

    # Define the graph
    workflow = StateGraph(SystemState)

    # Add nodes
    workflow.add_node("strategist", strategist_node)
    workflow.add_node("solver", solver_node)
    workflow.add_node("judge", judge_node)
    workflow.add_node("mechanic", mechanic_node)

    # Define edges: the main evolutionary loop
    workflow.set_entry_point("strategist")
    workflow.add_edge("strategist", "solver")
    workflow.add_edge("solver", "judge")
    workflow.add_edge("judge", "mechanic")

    # Conditional edge from mechanic: continue loop or end
    workflow.add_conditional_edges(
        "mechanic",
        should_continue_evolution,
        {
            "continue": "strategist",
            "end": END,
        },
    )

    return workflow.compile()


# Build the graph
evolution_graph = build_evolution_graph()
print("[OK] LangGraph workflow compiled successfully.")

# Visualize the graph structure
try:
    graph_ascii = """
    ┌─────────────┐
    │  STRATEGIST  │ ← Research & Plan
    │  (Agentic    │
    │   RAG)       │
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │   SOLVER     │ ← Evolutionary Code Gen
    │  (OpenEvolve)│
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │    JUDGE     │ ← Multi-faceted Eval
    │  (Benchmark  │
    │   + Analysis)│
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐     ┌─────┐
    │  MECHANIC    │────►│ END │ (if converged
    │  (Meta-Learn)│     └─────┘  or max cycles)
    └──────┬──────┘
           │ (if continue)
           ▼
    ┌─────────────┐
    │  STRATEGIST  │ ← Refined prompt
    │   (cycle+1)  │
    └─────────────┘
    """
    print(graph_ascii)
except Exception:
    pass

# Cell 10: System Runner and Session Manager

class AdaptEvolveSession:
    """Manages a complete AdaptEvolve-MAS session."""

    def __init__(self):
        self.graph = evolution_graph
        self.current_state: Optional[SystemState] = None
        self.is_running = False
        self.run_history: List[Dict] = []
        self.chat_history: List[Dict[str, str]] = []

    def start_session(self, goal: str, max_cycles: int = 5) -> str:
        """Start a new evolutionary optimization session."""
        self.current_state = create_initial_state(goal, max_cycles)
        self.is_running = True
        self.chat_history = []

        welcome = (
            f"🚀 **AdaptEvolve-MAS Session Started**\n\n"
            f"**Goal:** {goal}\n"
            f"**Max Cycles:** {max_cycles}\n"
            f"**Model:** {getattr(llm, 'model_name', MODEL_NAME)} ({LLM_BACKEND})\n\n"
            f"The system will now run the evolutionary loop:\n"
            f"1. 🧠 Strategist → Research & Plan\n"
            f"2. ⚙️ Solver → Evolutionary Code Generation\n"
            f"3. ⚖️ Judge → Multi-faceted Evaluation\n"
            f"4. 🛠️ Mechanic → Meta-Learning & Process Improvement\n\n"
            f"Starting..."
        )
        self.chat_history.append({"role": "system", "content": welcome})
        return welcome

    def run_full_evolution(self, goal: str, max_cycles: int = 3) -> str:
        """Run the complete evolutionary loop and return results."""
        self.start_session(goal, max_cycles)

        status_updates = [self.chat_history[-1]["content"]]

        try:
            # Run the LangGraph
            final_state = None
            for step_output in self.graph.stream(self.current_state):
                for node_name, node_state in step_output.items():
                    # Update current state
                    if self.current_state:
                        self.current_state.update(node_state)
                    else:
                        self.current_state = node_state

                    # Collect messages
                    new_messages = node_state.get("messages", [])
                    for msg in new_messages:
                        if msg not in self.chat_history:
                            self.chat_history.append(msg)
                            role = msg.get("role", "system")
                            content = msg.get("content", "")
                            if role != "system":
                                status_updates.append(f"**[{role.upper()}]** {content}")

                    cycle = node_state.get("current_cycle", self.current_state.get("current_cycle", 0))
                    score = node_state.get("holistic_score", self.current_state.get("holistic_score", 0))

                    status = f"📍 Node: `{node_name}` | Cycle: {cycle} | Score: {score:.4f}"
                    status_updates.append(status)

                    final_state = self.current_state

            self.is_running = False

            # Generate final report
            if final_state:
                report = self._generate_final_report(final_state)
                status_updates.append(report)

            return "\n\n---\n\n".join(status_updates)

        except Exception as e:
            error_msg = f"❌ **Error during evolution:** {str(e)}\n\n```\n{traceback.format_exc()}\n```"
            self.is_running = False
            status_updates.append(error_msg)
            return "\n\n---\n\n".join(status_updates)

    def run_single_cycle(self) -> str:
        """Run a single cycle of the evolutionary loop."""
        if self.current_state is None:
            return "❌ No active session. Please start a session first."

        results = []
        try:
            step_count = 0
            for step_output in self.graph.stream(self.current_state):
                for node_name, node_state in step_output.items():
                    self.current_state.update(node_state)
                    new_messages = node_state.get("messages", [])
                    for msg in new_messages:
                        role = msg.get("role", "system")
                        content = msg.get("content", "")
                        results.append(f"**[{role.upper()}]** {content}")

                step_count += 1
                if step_count >= 4:  # One full cycle = 4 nodes
                    break

            return "\n\n".join(results) if results else "No output from cycle."

        except Exception as e:
            return f"❌ Error: {str(e)}"

    def upload_document(self, text: str, source: str = "uploaded") -> str:
        """Upload a document for the RAG system."""
        rag_system.ingest_document(text, source=source)
        stats = rag_system.vector_store.get_stats()
        return f"✅ Document uploaded. Vector store now has {stats['total_chunks']} chunks."

    def get_current_solution(self) -> str:
        """Get the current best solution."""
        if self.current_state and self.current_state.get("best_solution"):
            sol = self.current_state["best_solution"]
            code = sol.get("code", "No code available")
            score = sol.get("score", 0)
            return f"**Score:** {score:.4f}\n\n```python\n{code}\n```"
        return "No solution available yet."

    def get_status(self) -> str:
        """Get current system status."""
        if not self.current_state:
            return "No active session."

        return (
            f"**Cycle:** {self.current_state.get('current_cycle', 0)}/{self.current_state.get('max_cycles', 5)}\n"
            f"**Score:** {self.current_state.get('holistic_score', 0):.4f}\n"
            f"**Status:** {self.current_state.get('status', 'unknown')}\n"
            f"**Active Agent:** {self.current_state.get('active_agent', 'none')}\n"
            f"**Population Size:** {len(self.current_state.get('population', []))}\n"
            f"**Continue:** {self.current_state.get('should_continue', False)}"
        )

    def _generate_final_report(self, state: SystemState) -> str:
        """Generate a comprehensive final report."""
        best = state.get("best_solution", {})
        history = state.get("cycle_history", [])
        eval_results = state.get("evaluation_results", {})

        scores_over_time = [h.get("holistic_score", 0) for h in history]

        report = f"""
# 🏆 AdaptEvolve-MAS Final Report

## Goal
{state.get('human_goal', 'N/A')}

## Results Summary
- **Total Cycles:** {state.get('current_cycle', 0)}
- **Final Score:** {state.get('holistic_score', 0):.4f}
- **Score Progression:** {' → '.join(f'{s:.3f}' for s in scores_over_time)}

## Best Solution
**Score:** {best.get('score', 0):.4f}
**Generation:** {best.get('generation', 'N/A')}

```python
{best.get('code', 'No code generated')}
```
"""
        return report

## Cell 11: Chat Interface Logic + File Parsing (UPGRADED)
# Cell 11: Chat Interface Logic + File Parsing

import os
import json as _json

# ─── File Parsing ─────────────────────────────────────────────────────────────

def parse_file(file_obj) -> str:
    """
    Parse an uploaded file and return its text content.
    Supports: .ipynb, .pdf, .docx, .py, .txt, .tex
    """
    if file_obj is None:
        return ""

    # Gradio gives us a NamedString/tmp path
    filepath = file_obj if isinstance(file_obj, str) else file_obj.name
    ext = os.path.splitext(filepath)[1].lower()

    try:
        if ext == ".ipynb":
            with open(filepath, "r", encoding="utf-8") as f:
                nb_data = _json.load(f)
            parts = []
            for cell in nb_data.get("cells", []):
                src = "".join(cell.get("source", []))
                if src.strip():
                    parts.append(f"[{cell['cell_type'].upper()}]\n{src}")
            return "\n\n".join(parts)

        elif ext == ".pdf":
            try:
                from pypdf import PdfReader
            except ImportError:
                import subprocess, sys
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pypdf"])
                from pypdf import PdfReader
            reader = PdfReader(filepath)
            return "\n\n".join(page.extract_text() or "" for page in reader.pages)

        elif ext == ".docx":
            try:
                from docx import Document
            except ImportError:
                import subprocess, sys
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "python-docx"])
                from docx import Document
            doc = Document(filepath)
            return "\n".join(p.text for p in doc.paragraphs)

        else:  # .py, .txt, .tex and anything else text-based
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                return f.read()

    except Exception as e:
        return f"[File parse error for {os.path.basename(filepath)}: {e}]"


# ─── Prompt Builder ───────────────────────────────────────────────────────────

SYSTEM_INSTRUCTION = """You are the AdaptEvolve-MAS assistant — a knowledgeable AI for evolutionary
code optimization, algorithm research, and multi-agent systems.

If file context is provided, ground your answers primarily in that content.
If the answer is not in the provided context, say: "Not found in provided data."
Be specific, technical, and helpful. Support markdown formatting in your responses."""


def build_prompt(history: list, file_context: str, query: str) -> str:
    """
    Assemble the full prompt: system + file context + conversation history + query.
    history: list of {"role": "user"|"assistant", "content": str}
    """
    parts = [SYSTEM_INSTRUCTION]

    if file_context:
        parts.append(f"\n\n--- UPLOADED FILE CONTEXT ---\n{file_context[:6000]}\n--- END FILE CONTEXT ---")

    if history:
        conv = []
        for msg in history[-6:]:  # keep last 6 turns to reduce inference time
            role = msg.get("role", "user").upper()
            conv.append(f"{role}: {msg['content']}")
        parts.append("\n\n--- CONVERSATION HISTORY ---\n" + "\n".join(conv) + "\n--- END HISTORY ---")

    parts.append(f"\n\nUSER: {query}\n\nASSISTANT:")
    return "".join(parts)


# ─── Response Generator ───────────────────────────────────────────────────────

def generate_response(prompt: str) -> str:
    """Call the active LLM backend with the assembled prompt and return text."""
    try:
        return llm.generate(prompt).strip()
    except Exception as e:
        return f"Generation error: {e}"


def generate_response_stream(prompt: str):
    """Yield text chunks — use with st.write_stream() for real-time output."""
    try:
        yield from llm.stream(prompt)
    except Exception as e:
        yield f"Generation error: {e}"


# ─── Main Chat Function ───────────────────────────────────────────────────────

def chat_function(
    message: str,
    history: list,          # list of {"role": ..., "content": ...}
    file_context: str,      # extracted text from uploaded file (gr.State)
):
    """
    Core chat function wired to Gradio.
    Returns (updated_history, updated_history, file_context)
    """
    if not message.strip():
        return history, history, file_context

    # 1. Immediately show user message
    history = history or []
    history.append({"role": "user", "content": message})

    # 2. Build full context-aware prompt
    prompt = build_prompt(history[:-1], file_context, message)

    # 3. Generate response
    response = generate_response(prompt)

    # 4. Append assistant response
    history.append({"role": "assistant", "content": response})

    return history, history, file_context


# ─── Legacy ChatInterface class (kept for backward compat with session/commands) ─

class ChatInterface:
    """Handles the chat-like interaction with the AdaptEvolve-MAS system."""

    def __init__(self, session: AdaptEvolveSession):
        self.session = session
        self.command_history: List[str] = []

    def process_message(self, user_message: str, chat_history: List[List[str]]) -> Tuple[str, List[List[str]]]:
        """Process a user message and return the response."""

        self.command_history.append(user_message)
        user_message_stripped = user_message.strip()

        # Command parsing
        if user_message_stripped.lower().startswith("/help"):
            response = self._help_command()

        elif user_message_stripped.lower().startswith("/run "):
            parts = user_message_stripped[5:].strip()
            max_cycles = 3
            if " --cycles=" in parts:
                idx = parts.index(" --cycles=")
                max_cycles = int(parts[idx + 10:].split()[0])
                parts = parts[:idx]
            response = self.session.run_full_evolution(parts, max_cycles)

        elif user_message_stripped.lower().startswith("/start "):
            parts = user_message_stripped[7:].strip()
            response = self.session.start_session(parts)

        elif user_message_stripped.lower() == "/cycle":
            response = self.session.run_single_cycle()

        elif user_message_stripped.lower() == "/status":
            response = self.session.get_status()

        elif user_message_stripped.lower() == "/solution":
            response = self.session.get_current_solution()

        elif user_message_stripped.lower().startswith("/upload "):
            text = user_message_stripped[8:].strip()
            response = self.session.upload_document(text)

        elif user_message_stripped.lower() == "/history":
            response = self._format_history()

        elif user_message_stripped.lower() == "/reset":
            self.session = AdaptEvolveSession()
            response = "🔄 Session reset. Ready for a new goal."

        elif user_message_stripped.lower().startswith("/research "):
            topic = user_message_stripped[10:].strip()
            result = rag_system.conduct_research(topic, topic)
            response = f"📚 **Research Results for:** {topic}\n\n{result.get('synthesis', 'No results')[:3000]}"

        else:
            response = self._handle_natural_language(user_message_stripped)

        # Update chat history
        chat_history = chat_history or []
        chat_history.append([user_message, response])

        return response, chat_history

    def _help_command(self) -> str:
        return """# 🤖 AdaptEvolve-MAS Commands

## Core Commands
| Command | Description |
|---------|-------------|
| `/run <goal>` | Run full evolutionary optimization for the given goal |
| `/run <goal> --cycles=N` | Run with N cycles (default: 3) |
| `/start <goal>` | Initialize a session without running |
| `/cycle` | Run a single evolutionary cycle |
| `/status` | Show current system status |
| `/solution` | Display the current best solution |

## Research & Documents
| Command | Description |
|---------|-------------|
| `/research <topic>` | Conduct standalone research on a topic |
| `/upload <text>` | Upload text as a reference document |

## System
| Command | Description |
|---------|-------------|
| `/history` | Show cycle history |
| `/reset` | Reset the session |
| `/help` | Show this help message |

## Natural Language
You can also just type naturally! For example:
- "Optimize a sorting algorithm for memory efficiency"
- "What's the current score?"
- "Show me the code"
"""

    def _handle_natural_language(self, message: str) -> str:
        """Handle natural language queries using the Lyra framework."""

        intent_prompt = f"""Classify this user message into one of these categories:
1. "run_optimization" - User wants to start or run an optimization/code generation task
2. "query_status" - User is asking about current status or results
3. "show_solution" - User wants to see the current code/solution
4. "research" - User wants to research a topic
5. "general_question" - General question about the system or the domain
6. "upload" - User wants to provide reference material

USER MESSAGE: {message}

Output ONLY the category name (one of the exact strings above)."""

        intent = llm.generate(intent_prompt).strip().lower().strip("\"'")

        if "run_optimization" in intent:
            goal_prompt = f"Extract the optimization goal from this message. Output only the goal, nothing else:\n\n{message}"
            goal = llm.generate(goal_prompt).strip()
            return self.session.run_full_evolution(goal, max_cycles=3)

        elif "query_status" in intent:
            return self.session.get_status()

        elif "show_solution" in intent:
            return self.session.get_current_solution()

        elif "research" in intent:
            result = rag_system.conduct_research(message, message)
            return f"📚 **Research Results:**\n\n{result.get('synthesis', 'No results')[:3000]}"

        else:
            context = ""
            if self.session.current_state:
                context = f"""
Current session state:
- Goal: {self.session.current_state.get('human_goal', 'N/A')}
- Cycle: {self.session.current_state.get('current_cycle', 0)}
- Score: {self.session.current_state.get('holistic_score', 0)}
- Status: {self.session.current_state.get('status', 'N/A')}
"""
            response = llm.generate(
                f"""You are the AdaptEvolve-MAS assistant. Answer the user's question helpfully.

{context}

USER: {message}

If the user seems to want to start an optimization, suggest using the /run command.
Be concise and helpful."""
            )
            return response

    def _format_history(self) -> str:
        """Format cycle history for display."""
        if not self.session.current_state:
            return "No active session."

        history = self.session.current_state.get("cycle_history", [])
        if not history:
            return "No cycle history yet."

        lines = ["# 📈 Cycle History\n"]
        for entry in history:
            lines.append(
                f"**Cycle {entry.get('cycle', '?')}** | "
                f"Score: {entry.get('holistic_score', 0):.4f} | "
                f"Pop: {entry.get('population_size', 0)} | "
                f"{entry.get('timestamp', '')}"
            )
            scores = entry.get("evaluation_summary", {})
            if scores:
                for dim, val in scores.items():
                    lines.append(f"  - {dim}: {val:.3f}")

        return "\n".join(lines)


# Initialize session manager and chat interface
session = AdaptEvolveSession()
chat_interface = ChatInterface(session)
print("[OK] Chat Interface Logic loaded.")

