import os
import json
import subprocess
from typing import Dict, List, Any, Optional
import ollama

# Maximum number of conversation exchanges to keep (system + N*2 messages)
MAX_HISTORY_EXCHANGES = 40

def get_available_models() -> List[str]:
    """Standalone function to query Ollama for available models."""
    try:
        response = ollama.list()
        # Handle both object-style (newer SDK) and dict-style responses
        if hasattr(response, 'models'):
            models = response.models
        elif isinstance(response, dict):
            models = response.get('models', [])
        else:
            return []
        result = []
        for m in models:
            if hasattr(m, 'model'):
                result.append(m.model)
            elif isinstance(m, dict):
                result.append(m.get('name', m.get('model', '')))
        return [r for r in result if r]
    except Exception:
        return []


def get_loaded_models() -> List[str]:
    """Standalone function to query Ollama for currently loaded/running models in memory."""
    try:
        if not hasattr(ollama, 'ps'):
            return []
        response = ollama.ps()
        if hasattr(response, 'models'):
            models = response.models
        elif isinstance(response, dict):
            models = response.get('models', [])
        else:
            return []
        result = []
        for m in models:
            if hasattr(m, 'model'):
                result.append(m.model)
            elif isinstance(m, dict):
                result.append(m.get('name', m.get('model', '')))
        return [r for r in result if r]
    except Exception:
        return []


class LocalAgent:
    def __init__(self, model_name: str = "qwen2.5:7b", cwd: Optional[str] = None, plan_only: bool = False):
        self.model_name = model_name
        self.cwd = cwd or os.getcwd()
        self.plan_only = plan_only
        self.conversation_history: List[Dict[str, Any]] = []
        self._init_system_prompt()

    def _init_system_prompt(self):
        core_tools_str = (
            "1. run_command: Execute a shell command.\n"
            "   Arguments: {\"command\": \"<shell command string>\"}\n"
            "2. read_file: View the contents of a file.\n"
            "   Arguments: {\"path\": \"<absolute or relative file path>\"}\n"
            "3. write_file: Create or write contents to a file.\n"
            "   Arguments: {\"path\": \"<file path>\", \"content\": \"<file content string>\"}\n"
            "4. list_dir: List the files and directories inside a path.\n"
            "   Arguments: {\"path\": \"<directory path, defaults to .>\"}\n"
        )
        
        # Load MCP tools dynamically
        from mcp_client import MCPManager
        mcp_tools = MCPManager.get_instance().get_all_tools()
        mcp_tools_str = ""
        for idx, t in enumerate(mcp_tools, start=5):
            tname = t.get("name", "")
            tdesc = t.get("description", "")
            tschema = json.dumps(t.get("inputSchema", {}))
            mcp_tools_str += f"{idx}. {tname}: {tdesc}\n"
            mcp_tools_str += f"   Arguments schema: {tschema}\n"

        if self.plan_only:
            plan_rules = (
                "STRICT PLAN-ONLY MODE RULES:\n"
                "- You are operating in a STRICT RECONNAISSANCE and PLANNING mode.\n"
                "- Your primary objective is to read files, look around the system, list directories, and locate/investigate issues.\n"
                "- You are STRICTLY FORBIDDEN from modifying any existing files, executing building/installing commands, or deleting items.\n"
                "- You are ONLY permitted to write to exactly one file: 'PLAN.md' (e.g. via write_file to path 'PLAN.md' or relative path in workspace). This file must highlight the issues you found and details of your proposed fix plan.\n"
                "- Do NOT run modification commands or other file writes.\n\n"
            )
        else:
            plan_rules = ""

        self.system_prompt = (
            "You are Superboof, a local AI Agent for Linux systems.\n"
            f"Current working directory: {self.cwd}\n\n"
            f"{plan_rules}"
            "RECOMMENDED WORKFLOW:\n"
            "When executing complex tasks, modifications, or code edits, adopt the following systematic workflow:\n"
            "1. Plan & Understand: Fully understand the user's goals. Outline the plan and architectural changes before modifying anything.\n"
            "2. Read & Analyze: Read files, inspect codebases, and gather necessary context using files/directories lookup or web searches.\n"
            "3. Ask & Align: If requirements are ambiguous or critical trade-offs exist, present option-based questions to align with the user.\n"
            "4. Code & Edit: Perform exact drop-in code modifications or writes once aligned.\n"
            "5. Test & Verify: Execute compile, test, or verification commands to confirm correctness.\n\n"
            "IMPORTANT RULES:\n"
            "- You operate in a fully user-supervised environment. EVERY single tool execution you request is paused and shown to the user, who must manually review and type 'y' or click 'Approve' to allow it to run. Because of this, it is entirely safe and encouraged for you to suggest command executions. You will not cause damage because the user acts as your safety gatekeeper.\n"
            "- When asked for information that requires local system state or network lookup (e.g., local/public IP, hostname, disk usage, active network interfaces, etc.), you MUST execute the command yourself via 'run_command' (e.g., 'ip route' or 'curl -s ifconfig.me') to retrieve the live data instead of telling the user to run it.\n"
            "- satisfying compound requests: If the user asks for multiple pieces of information (such as BOTH local and public IP addresses), make sure to gather all details and fully output all requested items in your final response.\n"
            "- Only perform exactly what the user asks. Do NOT take extra steps beyond the request unless necessary to guarantee success of the request.\n"
            "- Do NOT create, delete, read, or modify files unless the user explicitly asks or it is required to complete the user's objective.\n"
            "- Do NOT run 'exit' or system management commands.\n"
            "- Once a task is complete, respond with a short confirmation message containing the gathered answers. Do not keep calling tools.\n"
            "- Use ONLY the tools listed below. Do not invent tool names.\n\n"
            "TOOL USE & WEB SEARCH GUIDELINES:\n"
            "- When asked to search the web, get docs, look up libraries, find answers, or check external info, you MUST use the corresponding search/fetch tool or use `run_command` (e.g., with curl or python scripts) if no dedicated search tool is active. Never state that you lack internet access; always try to retrieve the information using tools.\n"
            "- To perform any action (such as modifying files, listing directories, executing shell commands, or retrieving web content), you must invoke the appropriate tool rather than explaining to the user how they should do it.\n"
            "- Prefer gathering information and executing actions yourself over requesting the user to perform tasks or give you the info.\n\n"
            "Available tools (respond with a JSON object to call one):\n"
            f"{core_tools_str}"
            f"{mcp_tools_str}\n"
            "Tool call format (respond with ONLY this JSON, no other text):\n"
            "{\n"
            "  \"thought\": \"Brief reasoning\",\n"
            "  \"tool_call\": {\n"
            "    \"name\": \"<tool_name>\",\n"
            "    \"arguments\": {}\n"
            "  }\n"
            "}\n\n"
            "If you do not need to call a tool, just write a normal text response to the user."
        )
        self.conversation_history = [{"role": "system", "content": self.system_prompt}]

    def add_user_message(self, message: str):
        self.conversation_history.append({"role": "user", "content": message})

    def add_tool_result(self, tool_name: str, result: Dict[str, Any]):
        content = f"Tool '{tool_name}' executed. Result:\n{json.dumps(result, indent=2)}"
        self.conversation_history.append({"role": "user", "content": content})

    def add_agent_message(self, message: str):
        self.conversation_history.append({"role": "assistant", "content": message})

    def _trim_history(self):
        """Keep conversation history within bounds to avoid context overflow."""
        # Always keep the system prompt (index 0)
        if len(self.conversation_history) <= MAX_HISTORY_EXCHANGES + 1:
            return
        # Keep system prompt + last N messages
        self.conversation_history = (
            [self.conversation_history[0]] +
            self.conversation_history[-(MAX_HISTORY_EXCHANGES):]
        )

    def _parse_tool_call(self, content: str) -> Optional[Dict[str, Any]]:
        """Robustly parse a tool call JSON from model output."""
        # Try 1: strict full parse
        try:
            parsed = json.loads(content)
            if "tool_call" in parsed and "name" in parsed["tool_call"]:
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass

        # Try 2: extract JSON block from markdown code fences
        import re
        fence_match = re.search(r'```(?:json)?\s*\n?({.*?})\s*\n?```', content, re.DOTALL)
        if fence_match:
            try:
                parsed = json.loads(fence_match.group(1))
                if "tool_call" in parsed and "name" in parsed["tool_call"]:
                    return parsed
            except (json.JSONDecodeError, TypeError):
                pass

        # Try 3: find outermost balanced braces
        depth = 0
        start = -1
        for i, ch in enumerate(content):
            if ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and start != -1:
                    try:
                        parsed = json.loads(content[start:i+1])
                        if "tool_call" in parsed and "name" in parsed["tool_call"]:
                            return parsed
                    except (json.JSONDecodeError, TypeError):
                        pass
                    start = -1

        return None

    def run_step(self) -> Dict[str, Any]:
        """Runs a single step of the agent, calling Ollama and parsing the response."""
        self._trim_history()
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=self.conversation_history,
                options={"temperature": 0.1}
            )
            content = response.get('message', {}).get('content', '').strip()

            parsed = self._parse_tool_call(content)
            if parsed:
                tool_name = parsed["tool_call"]["name"]
                # Reject invented tools immediately at the agent level
                from mcp_client import MCPManager
                mcp_tools = MCPManager.get_instance().get_all_tools()
                valid_tools = {"run_command", "read_file", "write_file", "list_dir"}
                for t in mcp_tools:
                    valid_tools.add(t.get("name"))
                if tool_name not in valid_tools:
                    return {
                        "type": "tool_call",
                        "thought": parsed.get("thought", ""),
                        "tool_name": tool_name,
                        "arguments": parsed["tool_call"].get("arguments", {}),
                        "raw_response": content
                    }
                return {
                    "type": "tool_call",
                    "thought": parsed.get("thought", ""),
                    "tool_name": tool_name,
                    "arguments": parsed["tool_call"].get("arguments", {}),
                    "raw_response": content
                }

            return {
                "type": "message",
                "content": content
            }
        except Exception as e:
            return {
                "type": "error",
                "content": f"Ollama connection error: {str(e)}"
            }


# Tool implementation functions
def run_command(command: str, cwd: Optional[str] = None, timeout: int = 60) -> Dict[str, Any]:
    """Execute a shell command with configurable timeout and working directory."""
    try:
        res = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd
        )
        result = {
            "exit_code": res.returncode,
            "stdout": res.stdout[:10000] if res.stdout else "",
            "stderr": res.stderr[:5000] if res.stderr else ""
        }
        return result
    except subprocess.TimeoutExpired:
        return {"error": f"Command execution timed out after {timeout} seconds."}
    except Exception as e:
        return {"error": str(e)}

def read_file(path: str) -> Dict[str, Any]:
    try:
        if not os.path.exists(path):
            return {"error": f"File '{path}' does not exist."}
        if os.path.isdir(path):
            return {"error": f"'{path}' is a directory. Use list_dir instead."}
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read(20000)
            truncated = len(content) >= 20000
            return {"content": content, "truncated": truncated}
    except Exception as e:
        return {"error": str(e)}

def write_file(path: str, content: str) -> Dict[str, Any]:
    try:
        parent = os.path.dirname(path)
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return {"success": True, "bytes_written": len(content)}
    except Exception as e:
        return {"error": str(e)}

def list_dir(path: str = ".") -> Dict[str, Any]:
    try:
        if not os.path.exists(path):
            return {"error": f"Directory '{path}' does not exist."}
        if not os.path.isdir(path):
            return {"error": f"'{path}' is not a directory."}
        items = os.listdir(path)
        result = []
        for item in sorted(items):
            full_path = os.path.join(path, item)
            try:
                is_dir = os.path.isdir(full_path)
                size = os.path.getsize(full_path) if not is_dir else None
            except OSError:
                # Handle broken symlinks or permission errors
                is_dir = False
                size = None
            result.append({
                "name": item,
                "type": "directory" if is_dir else "file",
                "size_bytes": size
            })
        return {"items": result}
    except Exception as e:
        return {"error": str(e)}
