"""PreToolUse hook: remind about plan before editing code files."""
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Only trigger for Edit and Write tools
tool_name = os.environ.get("CLAUDE_TOOL_NAME", "")
if tool_name not in ("Edit", "Write"):
    sys.exit(0)

# Parse tool input to get file path
tool_input_raw = os.environ.get("CLAUDE_TOOL_INPUT", "{}")
try:
    tool_input = json.loads(tool_input_raw)
except json.JSONDecodeError:
    tool_input = {}

file_path = tool_input.get("file_path", "")
# Only check for code/config files
if not file_path or not any(
    file_path.endswith(ext) for ext in (".py", ".yaml", ".yml", ".xml", ".sumocfg")
):
    sys.exit(0)

# Check if a plan file was touched recently
project_dir = Path(os.environ.get("CLAUDE_PROJECT_DIR", "."))
plans_dir = project_dir / ".claude" / "plans"
recent = False
if plans_dir.exists():
    cutoff = datetime.now(timezone.utc).timestamp() - 1800  # 30 minutes
    for p in plans_dir.glob("*.md"):
        if p.stat().st_mtime > cutoff:
            recent = True
            break

if not recent:
    print(
        "[coding-workflow] No plan modified in the last 30 minutes.",
        "Consider writing a plan to .claude/plans/ before editing code.",
        file=sys.stderr,
    )

sys.exit(0)  # Never block — soft reminder only
