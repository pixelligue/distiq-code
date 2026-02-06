"""
Agent System

Two-mode agent system like OpenCode:
- Build Agent: Full access, can edit files and run commands
- Plan Agent: Read-only, analysis and exploration only

Agents are configured via AGENTS.md file in project root.
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from loguru import logger


class AgentMode(str, Enum):
    """Agent operation modes."""
    BUILD = "build"  # Full access - edit files, run commands
    PLAN = "plan"    # Read-only - analysis only


@dataclass
class AgentPermissions:
    """What an agent can do."""
    can_read_files: bool = True
    can_write_files: bool = False
    can_run_commands: bool = False
    can_search_web: bool = True
    can_index_code: bool = True
    requires_confirmation: bool = True
    
    @classmethod
    def build_mode(cls) -> "AgentPermissions":
        """Full access for development work."""
        return cls(
            can_read_files=True,
            can_write_files=True,
            can_run_commands=True,
            can_search_web=True,
            can_index_code=True,
            requires_confirmation=False,  # Auto-approve safe actions
        )
    
    @classmethod
    def plan_mode(cls) -> "AgentPermissions":
        """Read-only for analysis."""
        return cls(
            can_read_files=True,
            can_write_files=False,
            can_run_commands=False,  # Denies by default
            can_search_web=True,
            can_index_code=True,
            requires_confirmation=True,  # Ask for everything
        )


@dataclass
class AgentConfig:
    """Configuration for an agent."""
    name: str
    mode: AgentMode
    permissions: AgentPermissions
    system_prompt: str = ""
    custom_instructions: list[str] = field(default_factory=list)
    tools_enabled: list[str] = field(default_factory=list)
    tools_disabled: list[str] = field(default_factory=list)


class Agent:
    """
    An AI coding agent with specific permissions and behavior.
    
    Usage:
        agent = Agent.build()  # Full access
        agent = Agent.plan()   # Read-only
        
        # Execute with agent
        result = await agent.execute(task, orchestrator)
    """
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.name = config.name
        self.mode = config.mode
        self.permissions = config.permissions
        
        # Confirmation callback
        self._confirm_callback: Callable[[str], bool] | None = None
    
    @classmethod
    def build(cls, custom_instructions: list[str] | None = None) -> "Agent":
        """Create build agent with full access."""
        return cls(AgentConfig(
            name="build",
            mode=AgentMode.BUILD,
            permissions=AgentPermissions.build_mode(),
            system_prompt=BUILD_SYSTEM_PROMPT,
            custom_instructions=custom_instructions or [],
            tools_enabled=["Read", "Write", "Glob", "Grep", "Bash", "Search", "WebSearch", "ReadURL"],
        ))
    
    @classmethod
    def plan(cls, custom_instructions: list[str] | None = None) -> "Agent":
        """Create plan agent for read-only analysis."""
        return cls(AgentConfig(
            name="plan",
            mode=AgentMode.PLAN,
            permissions=AgentPermissions.plan_mode(),
            system_prompt=PLAN_SYSTEM_PROMPT,
            custom_instructions=custom_instructions or [],
            tools_enabled=["Read", "Glob", "Grep", "Search", "WebSearch", "ReadURL"],
            tools_disabled=["Write", "Bash"],  # Explicitly disabled
        ))
    
    def set_confirm_callback(self, callback: Callable[[str], bool]):
        """Set callback for confirmation prompts."""
        self._confirm_callback = callback
    
    def can_use_tool(self, tool_name: str) -> bool:
        """Check if agent can use a specific tool."""
        # Explicitly disabled
        if tool_name in self.config.tools_disabled:
            return False
        
        # Explicitly enabled
        if self.config.tools_enabled and tool_name not in self.config.tools_enabled:
            return False
        
        # Check permissions
        if tool_name == "Write" and not self.permissions.can_write_files:
            return False
        if tool_name == "Bash" and not self.permissions.can_run_commands:
            return False
        
        return True
    
    async def confirm_action(self, action: str) -> bool:
        """
        Request confirmation for an action.
        
        Returns True if:
        - No confirmation required (build mode auto-approve)
        - User confirms
        """
        if not self.permissions.requires_confirmation:
            return True
        
        if self._confirm_callback:
            return self._confirm_callback(action)
        
        # Default: deny if no callback
        logger.warning(f"Action denied (no confirmation callback): {action}")
        return False
    
    def get_filtered_tools(self, all_tools: list[Any]) -> list[Any]:
        """Filter tools based on agent permissions."""
        return [t for t in all_tools if self.can_use_tool(t.name)]
    
    def get_system_prompt(self, project_context: str = "") -> str:
        """Build complete system prompt for agent."""
        parts = [self.config.system_prompt]
        
        # Add project context
        if project_context:
            parts.append(f"\n## Project Context\n{project_context}")
        
        # Add custom instructions
        if self.config.custom_instructions:
            parts.append("\n## Custom Instructions")
            for instruction in self.config.custom_instructions:
                parts.append(f"- {instruction}")
        
        # Add mode-specific notes
        if self.mode == AgentMode.PLAN:
            parts.append("""
## Mode: PLAN (Read-Only)
You are in PLAN mode. You can:
- Read and analyze files
- Search code and web
- Explain architecture
- Suggest changes

You CANNOT:
- Edit or create files
- Run bash commands
- Make any modifications

If user asks to make changes, explain what WOULD need to be changed,
then suggest switching to BUILD mode with: "Switch to BUILD mode to apply these changes."
""")
        
        return "\n".join(parts)


class AgentManager:
    """
    Manages agents and their configurations.
    
    Reads AGENTS.md from project root for custom instructions.
    """
    
    def __init__(self, project_dir: Path | None = None):
        self.project_dir = project_dir or Path.cwd()
        self._agents: dict[str, Agent] = {}
        self._current_agent: Agent | None = None
        
        # Load AGENTS.md if exists
        self._custom_instructions = self._load_agents_md()
        
        # Create default agents
        self._init_default_agents()
    
    def _init_default_agents(self):
        """Initialize default build and plan agents."""
        self._agents["build"] = Agent.build(
            custom_instructions=self._custom_instructions.get("build", [])
        )
        self._agents["plan"] = Agent.plan(
            custom_instructions=self._custom_instructions.get("plan", [])
        )
        
        # Default to build mode
        self._current_agent = self._agents["build"]
    
    def _load_agents_md(self) -> dict[str, list[str]]:
        """Load custom instructions from AGENTS.md."""
        agents_file = self.project_dir / "AGENTS.md"
        
        if not agents_file.exists():
            return {}
        
        try:
            content = agents_file.read_text(encoding="utf-8")
            return self._parse_agents_md(content)
        except Exception as e:
            logger.warning(f"Failed to load AGENTS.md: {e}")
            return {}
    
    def _parse_agents_md(self, content: str) -> dict[str, list[str]]:
        """Parse AGENTS.md content."""
        instructions: dict[str, list[str]] = {
            "build": [],
            "plan": [],
            "all": [],
        }
        
        current_section = "all"
        
        for line in content.split("\n"):
            line = line.strip()
            
            # Section headers
            if line.lower().startswith("## build"):
                current_section = "build"
            elif line.lower().startswith("## plan"):
                current_section = "plan"
            elif line.startswith("## "):
                current_section = "all"
            
            # Instruction lines
            elif line.startswith("- ") or line.startswith("* "):
                instruction = line[2:].strip()
                if instruction:
                    instructions[current_section].append(instruction)
        
        # Merge "all" into both agents
        instructions["build"] = instructions["all"] + instructions["build"]
        instructions["plan"] = instructions["all"] + instructions["plan"]
        
        return instructions
    
    @property
    def current(self) -> Agent:
        """Get current active agent."""
        return self._current_agent
    
    def switch(self, mode: str | AgentMode) -> Agent:
        """Switch to different agent mode."""
        if isinstance(mode, str):
            mode = mode.lower()
        else:
            mode = mode.value
        
        if mode not in self._agents:
            raise ValueError(f"Unknown agent mode: {mode}")
        
        self._current_agent = self._agents[mode]
        logger.info(f"Switched to {self._current_agent.name} agent")
        return self._current_agent
    
    def toggle(self) -> Agent:
        """Toggle between build and plan modes."""
        if self._current_agent.mode == AgentMode.BUILD:
            return self.switch("plan")
        else:
            return self.switch("build")
    
    def get_agent(self, mode: str) -> Agent | None:
        """Get agent by mode name."""
        return self._agents.get(mode.lower())
    
    def list_agents(self) -> list[str]:
        """List available agent modes."""
        return list(self._agents.keys())
    
    def create_agents_md(self) -> str:
        """Generate default AGENTS.md content for a project."""
        return DEFAULT_AGENTS_MD
    
    def save_agents_md(self):
        """Save default AGENTS.md to project."""
        agents_file = self.project_dir / "AGENTS.md"
        
        if agents_file.exists():
            logger.warning("AGENTS.md already exists, not overwriting")
            return
        
        agents_file.write_text(self.create_agents_md(), encoding="utf-8")
        logger.info(f"Created {agents_file}")


# System prompts
BUILD_SYSTEM_PROMPT = """You are an AI coding assistant in BUILD mode.

You have FULL ACCESS to:
- Read and write files
- Run bash/shell commands
- Search code and web
- Make any necessary changes

Your job is to help the developer build and modify their code.
Be proactive - if you see improvements, suggest them.
When making changes, explain what you're doing and why.

Always verify your changes work by checking for errors.
"""

PLAN_SYSTEM_PROMPT = """You are an AI coding assistant in PLAN mode.

You are in READ-ONLY mode for analysis and exploration.
You can:
- Read and analyze code
- Search codebase and web
- Explain architecture and patterns
- Suggest improvements

You CANNOT modify any files or run commands.
This is ideal for understanding unfamiliar code safely.

If the user wants to make changes, explain what would need to change
and suggest switching to BUILD mode.
"""

DEFAULT_AGENTS_MD = """# Project Agent Configuration

This file configures the AI coding agents for this project.
Instructions here apply to all agents unless specified.

## General Instructions

- Follow the existing code style and conventions
- Write tests for new functionality
- Update documentation when making changes
- Use type hints in Python code

## Build Agent

- Auto-format code after changes
- Run tests after making changes
- Commit with descriptive messages

## Plan Agent

- Provide detailed explanations
- Include code examples in suggestions
- Consider edge cases in analysis
"""


# Convenience functions
def get_agent_manager(project_dir: Path | None = None) -> AgentManager:
    """Get or create agent manager for project."""
    return AgentManager(project_dir)


def init_agents(project_dir: Path | None = None) -> AgentManager:
    """Initialize agents for project, creating AGENTS.md if needed."""
    manager = AgentManager(project_dir)
    
    agents_file = (project_dir or Path.cwd()) / "AGENTS.md"
    if not agents_file.exists():
        manager.save_agents_md()
    
    return manager
