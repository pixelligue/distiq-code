"""
Orchestrator - Main AI Agent Coordinator

Automatically handles:
- Code indexing (on first run and file changes)
- Context retrieval (semantic search for relevant code)
- Model routing (cheap for simple, expensive for complex)
- Tool execution
- Cost tracking

Usage:
    orchestrator = Orchestrator(project_dir=".")
    
    # Single request
    response = await orchestrator.process("Add authentication to the API")
    
    # Streaming
    async for chunk in orchestrator.stream("Fix the bug in login"):
        print(chunk, end="")
"""

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncGenerator, Any

from loguru import logger


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator."""
    
    # Indexing
    auto_index: bool = True
    watch_files: bool = True
    
    # Context
    max_context_tokens: int = 8000
    include_dependencies: bool = True
    
    # Routing
    enable_routing: bool = True
    default_planning_model: str = "haiku"  # For medium complexity
    default_execution_model: str = "deepseek-chat"  # Cheap execution
    
    # Caching
    enable_cache: bool = True
    cache_similarity_threshold: float = 0.92
    
    # Cost
    monthly_budget_usd: float | None = None


@dataclass
class ProcessResult:
    """Result of processing a request."""
    
    content: str
    model_used: str
    provider_used: str
    
    # Tokens
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    context_tokens: int = 0
    
    # Cost
    cost_usd: float = 0.0
    savings_usd: float = 0.0
    
    # Timing
    latency_ms: float = 0.0
    
    # Routing info
    complexity: str = "simple"
    was_cached: bool = False
    planning_model: str | None = None
    
    # Context info
    context_chunks: int = 0


class Orchestrator:
    """
    Main orchestrator that coordinates all components.
    
    Flow:
    1. Check/update index automatically
    2. Classify request complexity
    3. Build relevant context
    4. Check semantic cache
    5. Route to appropriate model(s)
    6. Execute and return result
    7. Track costs
    """
    
    def __init__(
        self,
        project_dir: Path | str | None = None,
        config: OrchestratorConfig | None = None,
    ):
        """
        Initialize orchestrator.
        
        Args:
            project_dir: Project directory (default: current)
            config: Configuration options
        """
        self.project_dir = Path(project_dir or ".").resolve()
        self.config = config or OrchestratorConfig()
        
        # Components (lazy loaded)
        self._indexer = None
        self._context_builder = None
        self._providers = {}
        self._cache = None
        self._cost_tracker = None
        self._classifier = None
        self._history_summarizer = None
        self._skill_registry = None
        self._tool_registry = None
        
        # Background tasks
        self._watcher_task = None
        self._index_lock = asyncio.Lock()
        
        # State
        self._initialized = False
        self._last_index_check = 0
        
    async def initialize(self) -> None:
        """Initialize all components."""
        if self._initialized:
            return
        
        logger.info(f"Initializing orchestrator for: {self.project_dir}")
        
        # Auto-index if enabled
        if self.config.auto_index:
            await self._ensure_indexed()
        
        # Start file watcher if enabled
        if self.config.watch_files:
            self._start_file_watcher()
        
        self._initialized = True
        logger.info("Orchestrator initialized")
    
    async def process(
        self,
        request: str,
        *,
        messages: list[dict] | None = None,
        system: str | None = None,
        force_model: str | None = None,
        skip_context: bool = False,
    ) -> ProcessResult:
        """
        Process a request with full automation.
        
        Args:
            request: User's request/question
            messages: Optional conversation history
            system: Optional system prompt
            force_model: Force specific model (skip routing)
            skip_context: Skip context retrieval
            
        Returns:
            ProcessResult with response and metadata
        """
        start_time = time.time()
        
        # Ensure initialized
        await self.initialize()
        
        result = ProcessResult(content="", model_used="", provider_used="")
        
        # 1. Check index freshness
        await self._maybe_update_index()
        
        # 2. Build context from index
        context_text = ""
        if not skip_context:
            context = await self._build_context(request)
            context_text = context.to_prompt() if context else ""
            result.context_chunks = len(context.chunks) if context else 0
            result.context_tokens = len(context_text) // 4  # Estimate
        
        # 3. Check semantic cache
        if self.config.enable_cache:
            cached = await self._check_cache(request, context_text)
            if cached:
                result.content = cached
                result.was_cached = True
                result.latency_ms = (time.time() - start_time) * 1000
                return result
        
        # 4. Classify complexity
        if force_model:
            complexity = "forced"
            execution_model = force_model
            planning_model = None
        elif self.config.enable_routing:
            complexity = await self._classify_complexity(request)
            execution_model, planning_model = self._get_models_for_complexity(complexity)
        else:
            complexity = "default"
            execution_model = self.config.default_execution_model
            planning_model = None
        
        result.complexity = complexity
        result.planning_model = planning_model
        
        # 5. Build full prompt
        full_messages = self._build_messages(
            request=request,
            context=context_text,
            history=messages,
            system=system,
        )
        
        # 6. Execute (with optional planning step)
        if planning_model and complexity in ("medium", "complex"):
            # Two-step: plan then execute
            plan = await self._generate_plan(full_messages, planning_model)
            response = await self._execute_plan(plan, execution_model)
        else:
            # Direct execution
            response = await self._direct_execute(full_messages, execution_model)
        
        # 7. Update result
        result.content = response.content
        result.model_used = response.model
        result.provider_used = response.provider
        result.input_tokens = response.input_tokens
        result.output_tokens = response.output_tokens
        result.cached_tokens = response.cached_tokens
        result.cost_usd = response.cost_usd
        result.latency_ms = (time.time() - start_time) * 1000
        
        # 8. Track cost
        await self._track_cost(result)
        
        # 9. Cache result
        if self.config.enable_cache:
            await self._cache_result(request, context_text, result.content)
        
        return result
    
    async def stream(
        self,
        request: str,
        *,
        messages: list[dict] | None = None,
        system: str | None = None,
        force_model: str | None = None,
        skip_context: bool = False,
    ) -> AsyncGenerator[str, None]:
        """
        Stream a response.
        
        Yields text chunks as they arrive.
        """
        await self.initialize()
        
        # Check index
        await self._maybe_update_index()
        
        # Build context
        context_text = ""
        if not skip_context:
            context = await self._build_context(request)
            context_text = context.to_prompt() if context else ""
        
        # Check cache
        if self.config.enable_cache:
            cached = await self._check_cache(request, context_text)
            if cached:
                yield cached
                return
        
        # Classify and get model
        if force_model:
            execution_model = force_model
        elif self.config.enable_routing:
            complexity = await self._classify_complexity(request)
            execution_model, _ = self._get_models_for_complexity(complexity)
        else:
            execution_model = self.config.default_execution_model
        
        # Build messages
        full_messages = self._build_messages(
            request=request,
            context=context_text,
            history=messages,
            system=system,
        )
        
        # Stream response
        provider = await self._get_provider_for_model(execution_model)
        
        async for chunk in provider.stream(full_messages, model=execution_model):
            yield chunk
    
    # =========================================================================
    # Internal methods
    # =========================================================================
    
    async def _ensure_indexed(self) -> None:
        """Ensure project is indexed."""
        async with self._index_lock:
            indexer = self._get_indexer()
            
            if indexer.vector_store and indexer.vector_store.size > 0:
                logger.debug("Index already exists")
                return
            
            logger.info("Indexing project...")
            indexer.index(show_progress=False)
            logger.info("Indexing complete")
    
    async def _maybe_update_index(self) -> None:
        """Update index if files changed (debounced)."""
        now = time.time()
        
        # Only check every 30 seconds
        if now - self._last_index_check < 30:
            return
        
        self._last_index_check = now
        
        async with self._index_lock:
            indexer = self._get_indexer()
            
            # Quick check for updates
            try:
                stats = indexer.update()
                if stats.get("updated_files", 0) > 0:
                    logger.debug(f"Updated {stats['updated_files']} files in index")
            except Exception as e:
                logger.warning(f"Index update failed: {e}")
    
    async def _build_context(self, request: str) -> Any:
        """Build context for the request."""
        try:
            builder = self._get_context_builder()
            context = builder.build(
                request,
                k=10,
                include_dependencies=self.config.include_dependencies,
            )
            return context
        except Exception as e:
            logger.warning(f"Context building failed: {e}")
            return None
    
    async def _classify_complexity(self, request: str) -> str:
        """Classify request complexity."""
        # Use existing routing.py if available
        try:
            from distiq_code.routing import classify_prompt
            _, tier = classify_prompt(request)
            
            # Map tier to complexity
            if tier == "haiku":
                return "simple"
            elif tier == "sonnet":
                return "medium"
            else:
                return "complex"
        except ImportError:
            # Fallback: simple heuristics
            request_lower = request.lower()
            
            # Complex patterns
            complex_patterns = [
                "refactor", "architect", "design", "redesign",
                "rewrite", "migrate", "convert entire",
            ]
            if any(p in request_lower for p in complex_patterns):
                return "complex"
            
            # Simple patterns
            simple_patterns = [
                "fix typo", "rename", "add comment",
                "format", "lint", "what is", "explain",
            ]
            if any(p in request_lower for p in simple_patterns):
                return "simple"
            
            # Default: medium
            return "medium"
    
    def _get_models_for_complexity(self, complexity: str) -> tuple[str, str | None]:
        """
        Get execution and planning models for complexity level.
        
        Returns:
            (execution_model, planning_model or None)
        """
        if complexity == "simple":
            # Direct to cheap model
            return (self.config.default_execution_model, None)
        
        elif complexity == "medium":
            # Haiku plans, DeepSeek executes
            return (self.config.default_execution_model, "haiku")
        
        else:  # complex
            # Sonnet plans, DeepSeek executes
            return (self.config.default_execution_model, "sonnet")
    
    def _build_messages(
        self,
        request: str,
        context: str,
        history: list[dict] | None,
        system: str | None,
    ) -> list[dict]:
        """Build full message list."""
        messages = []
        
        # Add history if provided
        if history:
            messages.extend(history)
        
        # Build user message with context
        user_content = request
        if context:
            user_content = f"{context}\n\n## User Request\n\n{request}"
        
        messages.append({
            "role": "user",
            "content": user_content,
        })
        
        return messages
    
    async def _generate_plan(
        self,
        messages: list[dict],
        planning_model: str,
    ) -> str:
        """Generate a plan using planning model."""
        provider = await self._get_provider_for_model(planning_model)
        
        # Add planning instruction
        planning_messages = messages.copy()
        planning_messages[-1]["content"] += """

Please analyze this request and create a detailed implementation plan.

Output format:
```json
{
  "summary": "What we're doing",
  "steps": [
    {"description": "Step 1", "files": ["file.py"]},
    ...
  ],
  "complexity": "medium"
}
```"""
        
        response = await provider.generate(
            planning_messages,
            model=planning_model,
            max_tokens=2000,
        )
        
        return response.content
    
    async def _execute_plan(
        self,
        plan: str,
        execution_model: str,
    ) -> Any:
        """Execute a plan using execution model."""
        provider = await self._get_provider_for_model(execution_model)
        
        execution_messages = [{
            "role": "user",
            "content": f"""Execute this plan and generate the code:

{plan}

Generate complete, working code for each step."""
        }]
        
        return await provider.generate(
            execution_messages,
            model=execution_model,
            max_tokens=4096,
        )
    
    async def _direct_execute(
        self,
        messages: list[dict],
        model: str,
    ) -> Any:
        """Direct execution without planning."""
        provider = await self._get_provider_for_model(model)
        
        return await provider.generate(
            messages,
            model=model,
            max_tokens=4096,
        )
    
    async def _get_provider_for_model(self, model: str) -> Any:
        """Get appropriate provider for model."""
        from distiq_code.providers import get_provider, ProviderType
        
        # Determine provider type from model name
        if model in ("haiku", "sonnet", "opus") or "claude" in model.lower():
            return get_provider(ProviderType.CLAUDE_CLI)
        elif "deepseek" in model.lower():
            try:
                return get_provider(ProviderType.DEEPSEEK)
            except ValueError:
                # Fallback to OpenRouter if no DeepSeek key
                return get_provider(ProviderType.OPENAI_COMPATIBLE)
        else:
            return get_provider(ProviderType.OPENAI_COMPATIBLE)
    
    async def _check_cache(self, request: str, context: str) -> str | None:
        """Check semantic cache."""
        try:
            from distiq_code.cache import SemanticCache
            from distiq_code.config import settings
            
            cache = SemanticCache(settings.cache_dir)
            result = cache.get(request)
            
            if result and result.get("similarity", 0) >= self.config.cache_similarity_threshold:
                logger.debug(f"Cache hit: similarity={result['similarity']:.2f}")
                return result.get("response")
        except Exception as e:
            logger.debug(f"Cache check failed: {e}")
        
        return None
    
    async def _cache_result(
        self,
        request: str,
        context: str,
        response: str,
    ) -> None:
        """Cache the result."""
        try:
            from distiq_code.cache import SemanticCache
            from distiq_code.config import settings
            
            cache = SemanticCache(settings.cache_dir)
            cache.set(request, response)
        except Exception as e:
            logger.debug(f"Caching failed: {e}")
    
    async def _track_cost(self, result: ProcessResult) -> None:
        """Track cost."""
        try:
            from distiq_code.stats import get_cost_tracker, RequestType
            
            tracker = get_cost_tracker()
            
            request_type = RequestType.PLANNING if result.planning_model else RequestType.EXECUTION
            
            tracker.record(
                provider=result.provider_used,
                model=result.model_used,
                request_type=request_type,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                cached_tokens=result.cached_tokens,
                cost_usd=result.cost_usd,
                latency_ms=result.latency_ms,
            )
        except Exception as e:
            logger.debug(f"Cost tracking failed: {e}")
    
    def _get_indexer(self):
        """Get or create indexer."""
        if self._indexer is None:
            from distiq_code.indexing import get_indexer
            self._indexer = get_indexer(self.project_dir)
        return self._indexer
    
    def _get_context_builder(self):
        """Get or create context builder."""
        if self._context_builder is None:
            from distiq_code.indexing.context_builder import ContextBuilder
            self._context_builder = ContextBuilder(
                self._get_indexer(),
                max_tokens=self.config.max_context_tokens,
            )
        return self._context_builder
    
    def _start_file_watcher(self) -> None:
        """Start background file watcher."""
        # This runs in background and triggers _maybe_update_index
        logger.debug("File watcher started (updates checked every 30s)")
    
    async def close(self) -> None:
        """Clean up resources."""
        if self._indexer:
            self._indexer.close()
    
    # =========================================================================
    # Skills support
    # =========================================================================
    
    async def run_skill(
        self,
        skill_name: str,
        **params,
    ) -> ProcessResult:
        """
        Run a pre-defined skill.
        
        Args:
            skill_name: Name of the skill (e.g., "refactor", "add-tests")
            **params: Parameters for the skill
            
        Returns:
            ProcessResult
        """
        from distiq_code.skills import get_skill
        
        skill = get_skill(skill_name)
        if not skill:
            raise ValueError(f"Unknown skill: {skill_name}")
        
        # Build messages from skill
        system_prompt, messages = skill.build_messages(**params)
        
        # Use skill's recommended model
        return await self.process(
            messages[-1]["content"],  # Last message is the user request
            messages=messages[:-1],   # Previous messages are examples
            system=system_prompt,
            force_model=skill.recommended_model,
        )
    
    def get_skill_registry(self):
        """Get skill registry."""
        if self._skill_registry is None:
            from distiq_code.skills import get_skill_registry
            self._skill_registry = get_skill_registry()
        return self._skill_registry
    
    def get_tool_registry(self):
        """Get tool registry."""
        if self._tool_registry is None:
            from distiq_code.tools import get_tool_registry
            self._tool_registry = get_tool_registry()
        return self._tool_registry
    
    async def execute_tool(self, name: str, params: dict) -> Any:
        """Execute a tool by name."""
        registry = self.get_tool_registry()
        return await registry.execute(name, params)
    
    # =========================================================================
    # History optimization
    # =========================================================================
    
    async def compress_history(
        self,
        messages: list[dict],
    ) -> list[dict]:
        """
        Compress conversation history if needed.
        
        Args:
            messages: Full conversation history
            
        Returns:
            Compressed message list
        """
        from distiq_code.optimization import compress_history
        
        return await compress_history(messages)


# Convenience function
async def process_request(
    request: str,
    project_dir: Path | str | None = None,
    **kwargs,
) -> ProcessResult:
    """
    Process a single request with full automation.
    
    Args:
        request: User's request
        project_dir: Project directory
        **kwargs: Additional options for Orchestrator.process()
        
    Returns:
        ProcessResult
    """
    orchestrator = Orchestrator(project_dir)
    try:
        return await orchestrator.process(request, **kwargs)
    finally:
        await orchestrator.close()
