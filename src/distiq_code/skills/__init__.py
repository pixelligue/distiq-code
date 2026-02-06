"""
Skills System

Skills are pre-compiled, zero-token task templates.
Instead of explaining the task every time, skills provide:
- Pre-written system prompts
- Example inputs/outputs
- Tool configurations
- Validation rules

This saves tokens by not needing to explain common tasks.

Usage:
    skill = skills.get("refactor")
    result = await skill.execute(target_file="main.py")
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass
class Skill:
    """A pre-compiled task skill."""
    
    name: str
    description: str
    
    # Pre-written system prompt
    system_prompt: str
    
    # User prompt template (with {placeholders})
    user_template: str
    
    # Required parameters
    parameters: list[str] = field(default_factory=list)
    
    # Optional parameters with defaults
    defaults: dict[str, Any] = field(default_factory=dict)
    
    # Tools this skill needs
    required_tools: list[str] = field(default_factory=list)
    
    # Recommended model
    recommended_model: str = "sonnet"
    
    # Examples for few-shot learning
    examples: list[dict] = field(default_factory=list)
    
    # Category for organization
    category: str = "general"
    
    def build_messages(self, **params) -> tuple[str, list[dict]]:
        """
        Build messages for this skill.
        
        Args:
            **params: Parameters to fill the template
            
        Returns:
            (system_prompt, messages)
        """
        # Merge with defaults
        all_params = {**self.defaults, **params}
        
        # Check required parameters
        missing = [p for p in self.parameters if p not in all_params]
        if missing:
            raise ValueError(f"Missing required parameters: {missing}")
        
        # Build user prompt from template
        user_prompt = self.user_template.format(**all_params)
        
        # Build messages with examples
        messages = []
        
        for example in self.examples:
            messages.append({
                "role": "user",
                "content": example.get("input", ""),
            })
            messages.append({
                "role": "assistant",
                "content": example.get("output", ""),
            })
        
        messages.append({
            "role": "user",
            "content": user_prompt,
        })
        
        return self.system_prompt, messages
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "system_prompt": self.system_prompt,
            "user_template": self.user_template,
            "parameters": self.parameters,
            "defaults": self.defaults,
            "required_tools": self.required_tools,
            "recommended_model": self.recommended_model,
            "examples": self.examples,
            "category": self.category,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Skill":
        """Create from dictionary."""
        return cls(**data)
    
    @classmethod
    def from_file(cls, path: Path) -> "Skill":
        """Load skill from JSON file."""
        data = json.loads(path.read_text())
        return cls.from_dict(data)


class SkillRegistry:
    """
    Registry of available skills.
    
    Skills are loaded from:
    1. Built-in skills (in this module)
    2. User skills directory (~/.distiq-code/skills/)
    3. Project skills (.distiq-code/skills/)
    """
    
    def __init__(
        self,
        skills_dir: Path | None = None,
    ):
        """
        Initialize skill registry.
        
        Args:
            skills_dir: User skills directory
        """
        self._skills: dict[str, Skill] = {}
        self.skills_dir = skills_dir
        
        # Register built-in skills
        self._register_builtins()
        
        # Load user skills
        if skills_dir:
            self._load_from_directory(skills_dir)
    
    def register(self, skill: Skill) -> None:
        """Register a skill."""
        self._skills[skill.name] = skill
        logger.debug(f"Registered skill: {skill.name}")
    
    def get(self, name: str) -> Skill | None:
        """Get skill by name."""
        return self._skills.get(name)
    
    def list_skills(
        self,
        category: str | None = None,
    ) -> list[Skill]:
        """List all registered skills."""
        skills = list(self._skills.values())
        
        if category:
            skills = [s for s in skills if s.category == category]
        
        return skills
    
    def _load_from_directory(self, directory: Path) -> None:
        """Load skills from directory."""
        if not directory.exists():
            return
        
        for file_path in directory.glob("*.json"):
            try:
                skill = Skill.from_file(file_path)
                self.register(skill)
            except Exception as e:
                logger.warning(f"Failed to load skill {file_path}: {e}")
    
    def _register_builtins(self) -> None:
        """Register built-in skills."""
        
        # Refactor skill
        self.register(Skill(
            name="refactor",
            description="Refactor code to improve quality without changing functionality",
            category="coding",
            system_prompt="""You are an expert code refactoring assistant.

Your job is to improve code quality by:
- Extracting reusable functions
- Improving naming
- Reducing complexity
- Adding type hints
- Improving error handling

RULES:
- Do NOT change functionality
- Preserve all tests
- Keep the same API
- Add comments explaining changes""",
            user_template="""Refactor this code:

```
{code}
```

Focus on: {focus}""",
            parameters=["code"],
            defaults={"focus": "readability and maintainability"},
            recommended_model="sonnet",
            required_tools=["Read", "Write"],
        ))
        
        # Add tests skill
        self.register(Skill(
            name="add-tests",
            description="Generate comprehensive unit tests for code",
            category="testing",
            system_prompt="""You are a testing expert.

Generate comprehensive unit tests that:
- Cover all edge cases
- Test error conditions
- Use proper mocking
- Follow testing best practices

Use pytest for Python, jest for JavaScript.""",
            user_template="""Generate tests for this code:

File: {file_path}
```
{code}
```

Framework: {framework}""",
            parameters=["code", "file_path"],
            defaults={"framework": "pytest"},
            recommended_model="sonnet",
            required_tools=["Read", "Write"],
        ))
        
        # Explain code skill
        self.register(Skill(
            name="explain",
            description="Explain how code works in detail",
            category="documentation",
            system_prompt="""You are a code explainer.

Explain code clearly for someone learning:
- What it does at a high level
- How the main parts work
- Key algorithms or patterns used
- Potential gotchas""",
            user_template="""Explain this code:

```
{code}
```

Audience level: {level}""",
            parameters=["code"],
            defaults={"level": "intermediate"},
            recommended_model="haiku",  # Simple task
        ))
        
        # Fix bug skill
        self.register(Skill(
            name="fix-bug",
            description="Analyze and fix a bug",
            category="debugging",
            system_prompt="""You are a debugging expert.

Approach:
1. Analyze the error/symptom
2. Identify root cause
3. Propose fix
4. Verify fix doesn't break other things

Always explain WHY the bug occurred.""",
            user_template="""Fix this bug:

Error: {error}

Code:
```
{code}
```

Additional context: {context}""",
            parameters=["error", "code"],
            defaults={"context": "none"},
            recommended_model="sonnet",
            required_tools=["Read", "Write", "Bash"],
        ))
        
        # Review PR skill
        self.register(Skill(
            name="review",
            description="Review code changes like a senior engineer",
            category="review",
            system_prompt="""You are a senior engineer doing code review.

Check for:
- Correctness
- Performance issues
- Security vulnerabilities
- Code style
- Missing tests
- Documentation

Be constructive and suggest improvements.""",
            user_template="""Review this code change:

{diff}

Focus areas: {focus}""",
            parameters=["diff"],
            defaults={"focus": "correctness, security, performance"},
            recommended_model="sonnet",
        ))
        
        # Generate docstrings skill
        self.register(Skill(
            name="docstrings",
            description="Add comprehensive docstrings to code",
            category="documentation",
            system_prompt="""You are a documentation expert.

Add docstrings that:
- Describe what the function/class does
- Document all parameters with types
- Document return values
- Include usage examples
- Note any exceptions raised

Use Google-style docstrings for Python.""",
            user_template="""Add docstrings to this code:

```
{code}
```

Style: {style}""",
            parameters=["code"],
            defaults={"style": "google"},
            recommended_model="haiku",
            required_tools=["Write"],
        ))
        
        # Optimize performance skill
        self.register(Skill(
            name="optimize",
            description="Optimize code for performance",
            category="performance",
            system_prompt="""You are a performance optimization expert.

Analyze code for:
- Time complexity improvements
- Memory efficiency
- Caching opportunities
- Unnecessary operations
- Better algorithms

Always explain the performance benefit of each change.""",
            user_template="""Optimize this code for performance:

```
{code}
```

Current metrics: {metrics}
Target: {target}""",
            parameters=["code"],
            defaults={"metrics": "unknown", "target": "general optimization"},
            recommended_model="sonnet",
            required_tools=["Read", "Write"],
        ))


# Global registry
_registry: SkillRegistry | None = None


def get_skill_registry(
    skills_dir: Path | None = None,
) -> SkillRegistry:
    """Get the global skill registry."""
    global _registry
    if _registry is None:
        _registry = SkillRegistry(skills_dir)
    return _registry


def get_skill(name: str) -> Skill | None:
    """Get a skill by name."""
    return get_skill_registry().get(name)


def list_skills(category: str | None = None) -> list[str]:
    """List available skill names."""
    return [s.name for s in get_skill_registry().list_skills(category)]
