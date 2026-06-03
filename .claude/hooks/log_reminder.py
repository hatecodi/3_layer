"""PostToolUse hook: remind to append to coding_log after code edits."""
import json
import os
import sys

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
if not file_path:
    sys.exit(0)

# Reminder message
print(
    f"[coding-workflow] Edited: {file_path}",
    "Remember to append to coding_log with:",
    "  [YYYY-MM-DD HH:MM]",
    "  Task: <summary>",
    "  Files: <list with brief descriptions>",
    "  Rationale: <why>",
    "  Status: <DONE|PARTIAL>",
    sep="\n",
    file=sys.stderr,
)

sys.exit(0)
