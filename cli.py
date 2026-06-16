import asyncio
import os
import sys
import json
import time
import shutil
from typing import Dict, Any, List
import ollama
from prompt_toolkit import Application
from prompt_toolkit.application import run_in_terminal
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.layout.containers import HSplit, VSplit, Window, FloatContainer, Float
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.layout.margins import ScrollbarMargin
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.widgets import TextArea
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.layout.menus import CompletionsMenu
import tty
import termios

from agent import LocalAgent, get_available_models, get_loaded_models, run_command, read_file, write_file, list_dir
import config

# ─── Rich console for pre-boot screens only ──────────────────────────────────
from rich.console import Console
console = Console()

# ─── Keyboard Utilities (for pre-boot menus) ─────────────────────────────────

def get_key() -> str:
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == '\x1b':
            ch2 = sys.stdin.read(1)
            ch3 = sys.stdin.read(1)
            if ch2 == '[':
                if ch3 == 'A':
                    return 'up'
                elif ch3 == 'B':
                    return 'down'
        elif ch == '\r' or ch == '\n':
            return 'enter'
        elif ch == '\x03':  # Ctrl+C
            raise KeyboardInterrupt
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

def select_option(title: str, options: List[str], default_idx: int = 0) -> int:
    """Arrow-key driven option selector."""
    selected = default_idx
    num_options = len(options)

    if title:
        console.print(title)

    sys.stdout.write("\x1b[?25l")  # hide cursor
    sys.stdout.flush()

    try:
        while True:
            for idx, option in enumerate(options):
                if idx == selected:
                    sys.stdout.write(f"\r\x1b[K  \x1b[1;36m❯ {option}\x1b[0m\n")
                else:
                    sys.stdout.write(f"\r\x1b[K    {option}\n")
            sys.stdout.flush()

            key = get_key()
            if key == 'up':
                selected = (selected - 1) % num_options
            elif key == 'down':
                selected = (selected + 1) % num_options
            elif key == 'enter':
                # Clear the menu lines + title line
                lines_to_clear = num_options + (2 if title else 0)
                sys.stdout.write(f"\x1b[{num_options}A")
                for _ in range(lines_to_clear):
                    sys.stdout.write("\r\x1b[K\x1b[1A\r\x1b[K")
                sys.stdout.write("\r\x1b[K")
                sys.stdout.flush()
                return selected

            sys.stdout.write(f"\x1b[{num_options}A")
            sys.stdout.flush()
    finally:
        sys.stdout.write("\x1b[?25h")  # show cursor
        sys.stdout.flush()

# ─── Display Helpers ──────────────────────────────────────────────────────────

TOOL_LABELS = {
    "run_command": "Bash",
    "read_file": "Read",
    "write_file": "Write",
    "list_dir": "List",
    "search": "Web Search",
}

def tool_bullet_plain(tool_name: str, args: Dict[str, Any]) -> str:
    """Format a tool call as a compact bullet (plain text, no Rich markup)."""
    label = TOOL_LABELS.get(tool_name, tool_name)
    if tool_name == "run_command":
        detail = args.get("command", "")
    elif tool_name == "read_file":
        detail = args.get("path", "")
    elif tool_name == "write_file":
        detail = args.get("path", "")
    elif tool_name == "list_dir":
        detail = args.get("path", ".")
    elif tool_name == "search":
        detail = args.get("query", "")
    else:
        detail = str(args)
    if len(detail) > 70:
        detail = detail[:67] + "..."
    return f"● {label}({detail})"

def format_tool_result(tool_name: str, tool_res: Dict[str, Any]) -> tuple:
    """Returns (is_success: bool, short_description: str)."""
    if "error" in tool_res:
        return False, tool_res["error"]
    elif tool_name == "run_command":
        code = tool_res.get("exit_code", 0)
        if code == 0:
            return True, "ran successfully"
        else:
            stderr = tool_res.get('stderr', '').strip()
            return False, f"exit code {code}: {stderr[:80]}"
    elif tool_name == "write_file":
        return True, f"wrote {tool_res.get('bytes_written', 0)} bytes"
    elif tool_name == "read_file":
        return True, f"read {len(tool_res.get('content', ''))} chars"
    elif tool_name == "list_dir":
        return True, f"{len(tool_res.get('items', []))} items"
    elif tool_name == "search":
        # Search results typically returns a list of results in a nested content key or standard format
        # Let's count standard entries
        try:
            content = tool_res.get("content", [])
            if isinstance(content, list) and content:
                import json
                text = content[0].get("text", "[]")
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return True, f"found {len(parsed)} search results"
        except Exception:
            pass
        return True, "search completed"
    else:
        return True, "completed"

def parse_direct_tool_call(tool_name: str, arg_string: str) -> Dict[str, Any]:
    """Parse raw argument string into correct JSON arguments for a tool."""
    if tool_name == "run_command":
        return {"command": arg_string}
    elif tool_name in ("read_file", "list_dir"):
        return {"path": arg_string}
    elif tool_name == "write_file":
        try:
            return json.loads(arg_string)
        except Exception:
            pass
        parts = arg_string.split(None, 1)
        return {
            "path": parts[0] if len(parts) > 0 else "",
            "content": parts[1] if len(parts) > 1 else ""
        }
    
    # Handle MCP tools dynamically
    from mcp_client import MCPManager
    mcp_tools = MCPManager.get_instance().get_all_tools()
    target_tool = None
    for t in mcp_tools:
        if t.get("name") == tool_name:
            target_tool = t
            break
            
    if target_tool:
        schema = target_tool.get("inputSchema", {})
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        
        try:
            parsed = json.loads(arg_string)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
            
        if len(properties) == 1:
            prop_name = list(properties.keys())[0]
            return {prop_name: arg_string}
        elif len(properties) > 1:
            for key in ("query", "q", "url", "command", "cmd", "path"):
                if key in properties:
                    return {key: arg_string}
            if required:
                return {required[0]: arg_string}
            prop_name = list(properties.keys())[0]
            return {prop_name: arg_string}
            
    return {}

# ─── Setup Screens (pre-boot, uses Rich) ─────────────────────────────────────

def check_directory_permissions(cwd: str) -> bool:
    if config.is_directory_allowed(cwd):
        return True

    console.print(f"\n[bold]Directory Permission Request[/bold]")
    console.print(f"Superboof wants to operate in: [bold cyan]{cwd}[/bold cyan]")
    console.print("It will be able to read, write, and execute files within this path.\n")

    choice = select_option("Authorize access?", [
        "Allow Always",
        "Allow This Session",
        "Deny and Exit"
    ], default_idx=2)

    if choice == 0:
        config.allow_directory(cwd)
        return True
    elif choice == 1:
        return True
    else:
        console.print("Permission denied. Exiting.")
        sys.exit(0)

async def select_model(available_models: List[str]) -> str:
    if len(available_models) == 1:
        return available_models[0]
    
    # Generate labels containing the best use summary
    import model_summarizer
    console.print("\n[dim]Analyzing models and fetching best uses...[/dim]")
    
    # Fetch capabilities in parallel
    async def fetch_cap(m):
        try:
            show_res = await asyncio.to_thread(ollama.show, m)
            caps = getattr(show_res, 'capabilities', []) or []
            return m, caps
        except Exception:
            return m, []

    cap_results = await asyncio.gather(*(fetch_cap(m) for m in available_models))
    model_caps = {m: caps for m, caps in cap_results}
    
    options = []
    loaded_models = get_loaded_models()
    last_used = config.get_last_used_model()
    
    default_idx = 0
    if last_used in available_models:
        default_idx = available_models.index(last_used)
        
    for m in available_models:
        summary = model_summarizer.get_or_create_model_summary(m)
        caps = model_caps.get(m, [])
        
        features = []
        if "tools" in caps:
            features.append("tools")
        if "vision" in caps:
            features.append("vision")
        feat_str = f" [{'+'.join(features)}]" if features else ""
        
        tags = []
        if m in loaded_models:
            tags.append("loaded in memory")
        if m == last_used:
            tags.append("last used")
        tag_str = f" [{', '.join(tags)}]" if tags else ""
        options.append(f"{m:<20}{feat_str:<16}{tag_str:<18} ({summary})")

    choice = select_option(
        "\n[bold]Select Ollama Model:[/bold]",
        options,
        default_idx=default_idx
    )
    return available_models[choice]

# ─── Tool Execution ──────────────────────────────────────────────────────────

async def execute_tool(tool_name: str, arguments: Dict[str, Any], cwd: str) -> Dict[str, Any]:
    if tool_name == "run_command":
        return await asyncio.to_thread(run_command, arguments.get("command", ""), cwd=cwd)
    elif tool_name == "read_file":
        return await asyncio.to_thread(read_file, arguments.get("path", ""))
    elif tool_name == "write_file":
        return await asyncio.to_thread(write_file, arguments.get("path", ""), arguments.get("content", ""))
    elif tool_name == "list_dir":
        return await asyncio.to_thread(list_dir, arguments.get("path", "."))
    else:
        from mcp_client import MCPManager
        try:
            return await MCPManager.get_instance().call_tool(tool_name, arguments)
        except Exception as e:
            return {"error": f"MCP execution failed: {str(e)}"}

# ─── Full-Screen Chat Application ────────────────────────────────────────────

HELP_TEXT = """\
Commands:
  /help            Show this help
  /models (or /m)  Show and select active Ollama models
  /tools (or /t)   Browse all available tools (core & MCP)
  /newmodel        Search or install new models from Ollama Library (shortcut: /n)
  /mcp             List, view, and add Model Context Protocol (MCP) servers
  /clear           Clear chat history and agent memory
  /exit            Exit (also /quit, /q)

Keyboard:
  ↑ / ↓     Navigate prompt history
  Enter     Submit message
  Ctrl+C    Cancel current operation
  Ctrl+D/X  Exit
"""

ASCII_BANNER = """\
 _                 __
| |               / _|
| |__   ___   ___|  _|
| '_ \\ / _ \\ / _ \\ |
| |_) | (_) | (_) | |
|_.__/ \\___/ \\___/|_|
"""

def build_banner_text(model_name: str, cwd: str) -> str:
    lines = [
        ASCII_BANNER,
        f"  superboof — local ai agent",
        f"  model: {model_name}  workspace: {cwd}",
        f"  /help for commands, ctrl+d or ctrl+x to quit",
        "",
        "─" * 60,
        "",
    ]
    return "\n".join(lines)


class SlashCompleter(Completer):
    def __init__(self, words, ignore_case=True):
        self.words = words
        self.ignore_case = ignore_case

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith('/'):
            return
        if ' ' in text:
            return
        word = text
        word_lower = word.lower() if self.ignore_case else word

        all_words = list(self.words)
        core_tools = ["/run_command", "/read_file", "/write_file", "/list_dir"]
        for ct in core_tools:
            if ct not in all_words:
                all_words.append(ct)
        try:
            from mcp_client import MCPManager
            manager = MCPManager.get_instance()
            mcp_tools = manager.get_all_tools()
            for t in mcp_tools:
                tname = f"/{t.get('name')}"
                if tname not in all_words:
                    all_words.append(tname)
            for sname in manager.sessions.keys():
                sname_cmd = f"/{sname}"
                if sname_cmd not in all_words:
                    all_words.append(sname_cmd)
        except Exception:
            pass

        for w in all_words:
            w_match = w.lower() if self.ignore_case else w
            if w_match.startswith(word_lower):
                yield Completion(w, start_position=-len(word))


class ChatApp:
    """Full-screen CLI chat application using prompt_toolkit."""

    def __init__(self, agent: LocalAgent, session_permitted: bool, welcome_msg: str):
        self.agent = agent
        self.session_permitted = session_permitted
        self._running = True
        self._allowed_tools_this_session = set()
        self._allowed_commands_this_session = set()
        self._current_process_task = None

        # ── Output buffer (read-only, scrollable) ──
        self.output_buffer = Buffer(read_only=True, name="output")

        slash_completer = SlashCompleter([
            '/help', '/models', '/model', '/tools', '/tool', '/newmodel', '/mcp', '/clear', '/exit', '/quit', '/q', '/c', '/h', '/m', '/t', '/new', '/n'
        ], ignore_case=True)

        # ── Input buffer ──
        history_file = os.path.expanduser("~/.config/superboof/history")
        os.makedirs(os.path.dirname(history_file), exist_ok=True)
        self.input_area = TextArea(
            height=1,
            prompt=HTML('<ansicyan>boof ❯ </ansicyan>'),
            multiline=False,
            wrap_lines=False,
            history=FileHistory(history_file),
            completer=slash_completer,
            complete_while_typing=True,
            accept_handler=self._on_submit,
        )

        # ── Status bar text ──
        self._status_text = "idle"

        # ── Layout ──
        output_window = Window(
            content=BufferControl(buffer=self.output_buffer, focusable=False),
            wrap_lines=True,
            right_margins=[ScrollbarMargin(display_arrows=True)],
        )

        def get_statusbar_text():
            cols = shutil.get_terminal_size().columns
            left_text = " superboof"
            status = self._status_text
            
            from mcp_client import MCPManager
            manager = MCPManager.get_instance()
            connected = len(manager.sessions)
            errored = len(manager.errored_servers)
            
            is_active_op = status not in (
                "idle", 
                "waiting for approval", 
                "selecting model...", 
                "browsing tools...", 
                "entering search query", 
                "entering custom model name"
            )
            
            model_part = f"model: {self.agent.model_name} │ "
            mcp_prefix = "mcp: "
            mcp_connected_str = f"{connected} connected"
            mcp_errs_str = f", {errored} error{'s' if errored > 1 else ''}" if errored > 0 else ""
            mcp_hint = " (use /mcp) │ "
            
            status_display = status
            if is_active_op:
                status_display += " (Ctrl+C to cancel)"
                
            plain_right = f"model: {self.agent.model_name} │ mcp: {mcp_connected_str}{mcp_errs_str} (use /mcp) │ {status_display} "
            padding = cols - len(left_text) - len(plain_right)
            if padding < 1:
                padding = 1
                
            tokens = [
                ("", left_text),
                ("", " " * padding),
                ("", model_part),
                ("", mcp_prefix),
                ("fg:#38bdf8", mcp_connected_str)
            ]
            
            if errored > 0:
                tokens.append(("fg:#f87171", mcp_errs_str))
                
            tokens.append(("", " (use "))
            tokens.append(("fg:#34d399 bold", "/mcp"))
            tokens.append(("", ") │ "))
            
            if is_active_op:
                tokens.append(("", f"{status} "))
                tokens.append(("fg:#f87171 bold", "(Ctrl+C to cancel)"))
                tokens.append(("", " "))
            else:
                tokens.append(("", f"{status} "))
                
            return tokens

        statusbar = Window(
            content=FormattedTextControl(get_statusbar_text),
            height=1,
            style="class:statusbar",
        )

        body = HSplit([
            output_window,
            statusbar,
            self.input_area,
        ])

        root = FloatContainer(
            content=body,
            floats=[
                Float(
                    content=CompletionsMenu(max_height=10),
                    attach_to_window=self.input_area.window,
                    allow_cover_cursor=True
                )
            ]
        )

        # ── Keybindings ──
        kb = KeyBindings()

        @kb.add('c-x')
        @kb.add('c-d')
        def _(event):
            self._running = False
            event.app.exit()

        @kb.add('c-c')
        def _(event):
            if self._current_process_task and not self._current_process_task.done():
                self._current_process_task.cancel()
            else:
                self.input_area.buffer.text = ""

        # Scroll output bindings
        @kb.add('pageup')
        def _(event):
            self.output_buffer.cursor_position = max(
                0, self.output_buffer.cursor_position - 1000
            )

        @kb.add('pagedown')
        def _(event):
            self.output_buffer.cursor_position = min(
                len(self.output_buffer.text), self.output_buffer.cursor_position + 1000
            )

        @kb.add('up')
        def _(event):
            # If the user is typing or navigating completions, use normal behavior.
            # Otherwise, allow scrolling up the output buffer.
            buff = event.current_buffer
            if buff == self.input_area.buffer and not buff.text:
                self.output_buffer.cursor_position = max(
                    0, self.output_buffer.cursor_position - 100
                )
            else:
                # Fallback to default behavior
                event.app.key_bindings.get_bindings_for_keys(('up',))
                # Trigger default up arrow logic (history navigate)
                from prompt_toolkit.key_binding.bindings.basic import load_basic_bindings
                # Since prompt-toolkit handles history in TextArea natively, we check if it handles it.
                # Just call history up:
                buff.history_backward()

        @kb.add('down')
        def _(event):
            buff = event.current_buffer
            if buff == self.input_area.buffer and not buff.text:
                self.output_buffer.cursor_position = min(
                    len(self.output_buffer.text), self.output_buffer.cursor_position + 100
                )
            else:
                buff.history_forward()

        # ── Style ──
        style = Style.from_dict({
            "statusbar": "fg:#e2e8f0 bg:#334155 bold",
        })

        self.layout = Layout(root, focused_element=self.input_area)
        self.app = Application(
            layout=self.layout,
            key_bindings=kb,
            style=style,
            full_screen=True,
            mouse_support=True,
        )

        # Seed the output with the banner
        self._append_output(build_banner_text(agent.model_name, agent.cwd))
        self._append_output(f"\nSuperboof ❯ {welcome_msg}")
        self.agent.add_agent_message(welcome_msg)

        # Pending user input queue
        self._input_queue: asyncio.Queue = asyncio.Queue()

    # ── Output helpers ──

    def _append_output(self, text: str):
        """Append text to the scrollable output pane and scroll to bottom."""
        current = self.output_buffer.text
        if current:
            new_text = current + "\n" + text
        else:
            new_text = text
        self.output_buffer.set_document(
            Document(text=new_text, cursor_position=len(new_text)),
            bypass_readonly=True,
        )

    def _update_last_line(self, text: str):
        """Update/overwrite the very last line of the output pane."""
        current = self.output_buffer.text
        if not current:
            self._append_output(text)
            return
        lines = current.split("\n")
        lines[-1] = text
        new_text = "\n".join(lines)
        self.output_buffer.set_document(
            Document(text=new_text, cursor_position=len(new_text)),
            bypass_readonly=True,
        )

    def _set_status(self, text: str):
        self._status_text = text
        self.app.invalidate()

    # ── Input handler ──

    def _on_submit(self, buff):
        """Called when user presses Enter in the input area."""
        text = buff.text.strip()
        if text:
            self._input_queue.put_nowait(text)

    # ── Tool approval (blocking prompt inside the app) ──

    async def _ask_approval(self, tool_name: str, args: Dict[str, Any]) -> bool:
        """Show tool call in output and ask for approval using a styled option selector."""
        if tool_name == "run_command":
            cmd_str = args.get("command", "")
            if cmd_str in self._allowed_commands_this_session:
                return True
            always_allowed_cmds = config.get_always_allowed_commands()
            if cmd_str in always_allowed_cmds:
                return True
        else:
            if tool_name in self._allowed_tools_this_session:
                return True
            always_allowed = config.get_always_allowed_tools()
            if tool_name in always_allowed:
                return True

        bullet = tool_bullet_plain(tool_name, args)
        self._append_output(f"\n{bullet}")
        
        options = [
            "Allow",
            "Allow this session",
            "Always allow (Persistent)",
            "Deny"
        ]
        
        self._set_status("waiting for approval")
        self.app.invalidate()
        
        choice_idx = await run_in_terminal(
            lambda: select_option(f"\n[bold]Tool Execution requested ({tool_name}). Action?[/bold]", options, default_idx=0)
        )
        
        if choice_idx is None or choice_idx < 0 or choice_idx == 3:
            return False
            
        if choice_idx == 0:
            return True
        elif choice_idx == 1:
            if tool_name == "run_command":
                cmd_str = args.get("command", "")
                self._allowed_commands_this_session.add(cmd_str)
                self._append_output(f"  ✓ Approved command for this session: '{cmd_str}'")
            else:
                self._allowed_tools_this_session.add(tool_name)
                self._append_output(f"  ✓ Approved for this session: '{tool_name}'")
            return True
        elif choice_idx == 2:
            if tool_name == "run_command":
                cmd_str = args.get("command", "")
                config.allow_command_persistently(cmd_str)
                self._append_output(f"  ✓ Approved command persistently: '{cmd_str}'")
            else:
                config.allow_tool_persistently(tool_name)
                self._append_output(f"  ✓ Approved persistently: '{tool_name}'")
            return True
            
        return False

    # ── Models screen ──

    async def _show_models(self):
        """List all installed Ollama models with best-use summaries, capabilities, and switch models."""
        import model_summarizer

        self._set_status("fetching models...")
        self.app.invalidate()

        models = await asyncio.to_thread(get_available_models)
        if not models:
            self._append_output("\nNo Ollama models found.")
            self._set_status("idle")
            return

        # Fetch capabilities in parallel
        async def fetch_cap(m):
            try:
                show_res = await asyncio.to_thread(ollama.show, m)
                caps = getattr(show_res, 'capabilities', []) or []
                return m, caps
            except Exception:
                return m, []

        cap_results = await asyncio.gather(*(fetch_cap(m) for m in models))
        model_caps = {m: caps for m, caps in cap_results}

        # Build options list
        loaded_models = await asyncio.to_thread(get_loaded_models)
        last_used = config.get_last_used_model()
        active = self.agent.model_name

        option_labels = []
        default_idx = 0

        for idx, m in enumerate(models):
            summary = await asyncio.to_thread(model_summarizer.get_or_create_model_summary, m)
            caps = model_caps.get(m, [])
            
            features = []
            if "tools" in caps:
                features.append("tools")
            if "vision" in caps:
                features.append("vision")
            feat_str = f" [{'+'.join(features)}]" if features else ""
            
            tags = []
            if m == active:
                tags.append("active")
                default_idx = idx
            if m in loaded_models:
                tags.append("loaded in memory")
            elif m == last_used:
                tags.append("last used")
            tag_str = f" [{', '.join(tags)}]" if tags else ""
            
            option_labels.append(f"{m:<20}{feat_str:<16}{tag_str:<18} ({summary})")

        option_labels.append("Cancel (go back)")

        self._set_status("selecting model...")
        self.app.invalidate()

        selected_idx = await run_in_terminal(
            lambda: select_option("\n[bold]Select Model to Switch to:[/bold]", option_labels, default_idx=default_idx)
        )

        if selected_idx is not None and selected_idx >= 0 and selected_idx < len(models):
            new_model = models[selected_idx]
            if new_model != active:
                self.agent = LocalAgent(model_name=new_model, cwd=self.agent.cwd)
                config.set_last_used_model(new_model)
                self._append_output(f"\nSwitched active model to: {new_model}")
            else:
                self._append_output(f"\nModel {new_model} is already active.")
        else:
            self._append_output("\nModel switch cancelled.")

        self._set_status("idle")
        self.app.invalidate()

    async def _show_mcp(self):
        """Show all currently configured MCP servers and allow adding a new one."""
        from mcp_client import MCPManager
        manager = MCPManager.get_instance()
        
        # Read configured servers
        config_servers = {}
        try:
            if os.path.exists(manager.config_path):
                with open(manager.config_path, "r") as f:
                    cfg = json.load(f)
                    config_servers = cfg.get("mcpServers", {})
        except Exception:
            pass

        # Build options list
        option_labels = []
        servers = list(config_servers.keys())
        
        for sname in servers:
            scfg = config_servers[sname]
            cmd = scfg.get("command", "")
            args = scfg.get("args", [])
            args_str = " ".join(args)
            
            is_active = sname in manager.sessions
            status_tag = "[active]" if is_active else "[configured]"
            
            option_labels.append(f"{sname:<20} {status_tag:<12} ({cmd} {args_str})")
            
        option_labels.append("[Add new MCP server]")
        option_labels.append("Cancel (go back)")

        self._set_status("browsing MCP servers...")
        self.app.invalidate()

        selected_idx = await run_in_terminal(
            lambda: select_option("\n[bold]Configured MCP Servers:[/bold]", option_labels, default_idx=0)
        )

        if selected_idx is None or selected_idx < 0 or selected_idx == len(option_labels) - 1:
            self._append_output("MCP browsing cancelled.")
            self._set_status("idle")
            return

        if selected_idx == len(option_labels) - 2:
            self._append_output("\n[bold]Add New MCP Server[/bold]")
            
            self._append_output("Enter server name (e.g. weather-server):")
            self._set_status("entering MCP server name")
            self.app.invalidate()
            sname = (await self._input_queue.get()).strip()
            if not sname:
                self._append_output("Cancelled.")
                self._set_status("idle")
                return

            self._append_output("Enter command/executable (e.g. npx, node, python3):")
            self._set_status("entering MCP command")
            self.app.invalidate()
            cmd = (await self._input_queue.get()).strip()
            if not cmd:
                self._append_output("Cancelled.")
                self._set_status("idle")
                return

            self._append_output("Enter arguments (optional, space-separated):")
            self._set_status("entering MCP arguments")
            self.app.invalidate()
            args_raw = (await self._input_queue.get()).strip()
            import shlex
            try:
                args = shlex.split(args_raw)
            except Exception:
                args = args_raw.split()

            self._append_output(f"Adding MCP server '{sname}' ({cmd} {' '.join(args)})...")
            self._set_status("saving MCP server...")
            self.app.invalidate()

            try:
                cfg = {"mcpServers": {}}
                if os.path.exists(manager.config_path):
                    with open(manager.config_path, "r") as f:
                        cfg = json.load(f)
                
                if "mcpServers" not in cfg:
                    cfg["mcpServers"] = {}
                
                cfg["mcpServers"][sname] = {
                    "command": cmd,
                    "args": args
                }
                
                with open(manager.config_path, "w") as f:
                    json.dump(cfg, f, indent=2)
                
                await manager.load_servers()
                self.agent._init_system_prompt()
                
                if sname in manager.sessions:
                    self._append_output(f"Successfully loaded and started MCP server '{sname}'!")
                else:
                    self._append_output(f"Saved config, but failed to start MCP server '{sname}'. Check command or args.")
                    err_msg = manager.load_errors.get(sname)
                    if err_msg:
                        self._append_output(f"Error: {err_msg}")
            except Exception as e:
                self._append_output(f"Error saving/loading MCP server: {e}")
                
            self._set_status("idle")
            self.app.invalidate()
        else:
            sname = servers[selected_idx]
            scfg = config_servers[sname]
            cmd = scfg.get("command", "")
            args = scfg.get("args", [])
            is_active = sname in manager.sessions
            
            self._append_output(f"\n[bold]MCP Server Details: {sname}[/bold]")
            self._append_output(f"Status:  {'Active' if is_active else 'Inactive'}")
            self._append_output(f"Command: {cmd}")
            self._append_output(f"Args:    {' '.join(args)}")
            
            if is_active:
                tools = [t for t in manager.sessions[sname].tools]
                self._append_output(f"Provided Tools ({len(tools)}):")
                for t in tools:
                    self._append_output(f"  - {t.get('name')}: {t.get('description')}")
            else:
                self._append_output("This server is configured but not currently running.")
                err_msg = manager.load_errors.get(sname)
                if err_msg:
                    self._append_output(f"Error Log:\n{err_msg}")
            self.app.invalidate()
            
            details_options = [
                "Edit Display Name / Rename",
                "Delete Server",
                "Back to List"
            ]
            
            self._set_status(f"options for {sname}...")
            self.app.invalidate()
            
            action_idx = await run_in_terminal(
                lambda: select_option(f"\n[bold]Action for MCP server '{sname}':[/bold]", details_options, default_idx=0)
            )
            
            if action_idx == 0:
                self._append_output(f"\nEnter new display name for '{sname}':")
                self._set_status("renaming MCP server...")
                self.app.invalidate()
                new_sname = (await self._input_queue.get()).strip()
                if not new_sname or new_sname == sname:
                    self._append_output("Rename cancelled.")
                    self._set_status("idle")
                    return
                
                self._append_output(f"Renaming '{sname}' to '{new_sname}'...")
                self.app.invalidate()
                
                try:
                    cfg = {"mcpServers": {}}
                    if os.path.exists(manager.config_path):
                        with open(manager.config_path, "r") as f:
                            cfg = json.load(f)
                    
                    if "mcpServers" in cfg and sname in cfg["mcpServers"]:
                        cfg["mcpServers"][new_sname] = cfg["mcpServers"].pop(sname)
                        
                        with open(manager.config_path, "w") as f:
                            json.dump(cfg, f, indent=2)
                        
                        if sname in manager.sessions:
                            await manager.sessions[sname].stop()
                            del manager.sessions[sname]
                            
                        await manager.load_servers()
                        self.agent._init_system_prompt()
                        
                        self._append_output(f"Successfully renamed to '{new_sname}'!")
                except Exception as e:
                    self._append_output(f"Error renaming server: {e}")
                    
            elif action_idx == 1:
                self._append_output(f"\nAre you sure you want to delete MCP server '{sname}'? [y/n]:")
                self._set_status("confirming deletion...")
                self.app.invalidate()
                confirm = (await self._input_queue.get()).strip().lower()
                if confirm not in ("y", "yes"):
                    self._append_output("Delete cancelled.")
                    self._set_status("idle")
                    return
                
                self._append_output(f"Deleting '{sname}'...")
                self.app.invalidate()
                
                try:
                    cfg = {"mcpServers": {}}
                    if os.path.exists(manager.config_path):
                        with open(manager.config_path, "r") as f:
                            cfg = json.load(f)
                    
                    if "mcpServers" in cfg and sname in cfg["mcpServers"]:
                        cfg["mcpServers"].pop(sname)
                        
                        with open(manager.config_path, "w") as f:
                            json.dump(cfg, f, indent=2)
                        
                        if sname in manager.sessions:
                            await manager.sessions[sname].stop()
                            del manager.sessions[sname]
                            
                        await manager.load_servers()
                        self.agent._init_system_prompt()
                        
                        self._append_output(f"Successfully deleted MCP server '{sname}'.")
                except Exception as e:
                    self._append_output(f"Error deleting server: {e}")
            
            self._set_status("idle")
            self.app.invalidate()

    async def _show_tools(self):
        """Show all available tools (core + MCP) in an interactive menu and print details on selection."""
        from mcp_client import MCPManager
        
        # 1. Core tools
        tools_list = [
            {
                "name": "run_command",
                "source": "core",
                "description": "Execute a shell command",
                "schema": {"command": "string"}
            },
            {
                "name": "read_file",
                "source": "core",
                "description": "View the contents of a file",
                "schema": {"path": "string"}
            },
            {
                "name": "write_file",
                "source": "core",
                "description": "Create or write contents to a file",
                "schema": {"path": "string", "content": "string"}
            },
            {
                "name": "list_dir",
                "source": "core",
                "description": "List the files and directories inside a path",
                "schema": {"path": "string (optional)"}
            }
        ]
        
        # 2. MCP tools
        mcp_tools = MCPManager.get_instance().get_all_tools()
        for t in mcp_tools:
            tools_list.append({
                "name": t.get("name", ""),
                "source": f"mcp:{t.get('server', '')}",
                "description": t.get("description", ""),
                "schema": t.get("inputSchema", {})
            })
            
        option_labels = []
        for t in tools_list:
            option_labels.append(f"{t['name']:<25} ({t['source']}) - {t['description']}")
            
        option_labels.append("Cancel (go back)")
        
        self._set_status("browsing tools...")
        self.app.invalidate()
        
        selected_idx = await run_in_terminal(
            lambda: select_option("\n[bold]Available Tools (Select to view detailed schema):[/bold]", option_labels, default_idx=0)
        )
        
        if selected_idx is not None and selected_idx >= 0 and selected_idx < len(tools_list):
            t = tools_list[selected_idx]
            lines = []
            lines.append("")
            lines.append("─" * 70)
            lines.append(f"  Tool Details: {t['name']}")
            lines.append("─" * 70)
            lines.append(f"  Source:      {t['source']}")
            lines.append(f"  Description: {t['description']}")
            schema_str = json.dumps(t['schema'], indent=2)
            lines.append(f"  Schema:\n{schema_str}")
            lines.append("─" * 70)
            lines.append("")
            self._append_output("\n".join(lines))
        else:
            self._append_output("\nBrowsing cancelled.")
            
        self._set_status("idle")
        self.app.invalidate()

    async def _search_and_install_model(self, initial_query: str):
        """Search Ollama Library and optionally pull a model."""
        import urllib.request
        import urllib.parse
        import html as html_lib
        import re
        import model_summarizer

        query = initial_query
        if not query:
            self._append_output("\nSearch Ollama Library")
            self._append_output("Enter search term (or press Enter to show top 10 uninstalled models):")
            self._set_status("entering search query")
            self.app.invalidate()
            query = await self._input_queue.get()
            query = query.strip()

        self._set_status("searching ollama.com...")
        self.app.invalidate()

        url = "https://ollama.com/library"
        if query:
            url += f"?q={urllib.parse.quote(query)}"

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Superboof/1.0"})
            resp = await asyncio.to_thread(urllib.request.urlopen, req, timeout=7)
            html_content = resp.read().decode('utf-8')
        except Exception as e:
            self._append_output(f"\nError fetching Ollama library: {e}")
            self._set_status("idle")
            return

        pattern = re.compile(
            r'href=\"/library/([a-zA-Z0-9_\.-]+)\".*?group-hover:underline[^>]*>([^<]+)</span>.*?<p[^>]*break-words[^>]*>([^<]+)</p>',
            re.DOTALL
        )
        results = []
        for match in pattern.finditer(html_content):
            name, display_name, desc = match.groups()
            desc_cleaned = html_lib.unescape(desc.strip())
            results.append((name.strip(), display_name.strip(), desc_cleaned))

        if not results:
            self._append_output(f"\nNo models found on Ollama library for '{query}'.")
            self._set_status("idle")
            return

        # Fetch installed models to filter them out
        installed_models = await asyncio.to_thread(get_available_models)
        installed_bases = set()
        for m in installed_models:
            if ":" in m:
                installed_bases.add(m.split(":")[0])
            installed_bases.add(m)

        uninstalled_results = []
        for name, display_name, desc in results:
            if name not in installed_bases and display_name not in installed_bases:
                uninstalled_results.append((name, display_name, desc))

        if not uninstalled_results:
            if query:
                self._append_output(f"\nAll models matching '{query}' are already installed.")
            else:
                self._append_output(f"\nAll top models from Ollama Library are already installed.")
            self._set_status("idle")
            return

        # Limit to top 10
        displayed_models = uninstalled_results[:10]

        # Calculate system RAM
        import os
        try:
            mem_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
            mem_gb = mem_bytes / (1024 ** 3)
        except Exception:
            mem_gb = 8.0

        lines = []
        lines.append("")
        lines.append("─" * 70)
        title_text = f"  Ollama Library Search: '{query}'" if query else "  Top Uninstalled Ollama Models"
        lines.append(title_text)
        lines.append("─" * 70)

        for idx, (name, display_name, desc) in enumerate(displayed_models):
            # Parse parameters
            param_match = re.search(r'\b(\d+(?:\.\d+)?)\s*[Bb]\b', desc)
            if param_match:
                params = float(param_match.group(1))
                size_str = f"{params}B"
            else:
                # Estimate based on name
                lower_name = name.lower()
                if "plus" in lower_name or "70b" in lower_name:
                    params = 70.0
                elif "35b" in lower_name or "command-r" in lower_name:
                    params = 35.0
                elif "14b" in lower_name or "phi4" in lower_name:
                    params = 14.0
                elif "9b" in lower_name or "gemma2" in lower_name:
                    params = 9.0
                elif "8b" in lower_name or "llama3.1" in lower_name:
                    params = 8.0
                elif "7b" in lower_name or "mistral" in lower_name or "deepseek" in lower_name:
                    params = 7.0
                elif "3b" in lower_name or "llama3.2" in lower_name or "mini" in lower_name:
                    params = 3.0
                elif "1b" in lower_name or "tiny" in lower_name or "smollm" in lower_name:
                    params = 1.5
                else:
                    params = 7.0
                size_str = "~7B"

            required_ram = params * 1.2
            if required_ram < mem_gb * 0.7:
                compat = "Highly Compatible"
            elif required_ram < mem_gb * 1.0:
                compat = "Compatible"
            elif required_ram < mem_gb * 1.5:
                compat = "May run slow"
            else:
                compat = "High RAM required"

            desc_snippet = desc[:50] + "..." if len(desc) > 50 else desc
            lines.append(f"  [{idx + 1}] {name:<20} ({size_str}) - {compat}")
            lines.append(f"      {desc_snippet}")
            lines.append("")

        lines.append("─" * 70)
        self._append_output("\n".join(lines))

        # Build options for selector
        option_labels = []
        for name, display_name, desc in displayed_models:
            # Estimate parameters / compatibility again to show in selector label
            param_match = re.search(r'\b(\d+(?:\.\d+)?)\s*[Bb]\b', desc)
            if param_match:
                params = float(param_match.group(1))
                size_str = f"{params}B"
            else:
                lower_name = name.lower()
                if "plus" in lower_name or "70b" in lower_name:
                    params = 70.0
                elif "35b" in lower_name or "command-r" in lower_name:
                    params = 35.0
                elif "14b" in lower_name or "phi4" in lower_name:
                    params = 14.0
                elif "9b" in lower_name or "gemma2" in lower_name:
                    params = 9.0
                elif "8b" in lower_name or "llama3.1" in lower_name:
                    params = 8.0
                elif "7b" in lower_name or "mistral" in lower_name or "deepseek" in lower_name:
                    params = 7.0
                elif "3b" in lower_name or "llama3.2" in lower_name or "mini" in lower_name:
                    params = 3.0
                elif "1b" in lower_name or "tiny" in lower_name or "smollm" in lower_name:
                    params = 1.5
                else:
                    params = 7.0
                size_str = "~7B"

            required_ram = params * 1.2
            if required_ram < mem_gb * 0.7:
                compat = "Compatible"
            elif required_ram < mem_gb * 1.0:
                compat = "Compatible"
            elif required_ram < mem_gb * 1.5:
                compat = "May run slow"
            else:
                compat = "High RAM required"

            option_labels.append(f"{name:<20} ({size_str}) - {compat}")
        
        option_labels.append("Search again or enter custom model name...")
        option_labels.append("Cancel Pull")

        self._set_status("selecting model...")
        self.app.invalidate()

        selected_idx = await run_in_terminal(
            lambda: select_option("\n[bold]Select Model to Pull:[/bold]", option_labels, default_idx=0)
        )

        if selected_idx is None or selected_idx < 0 or selected_idx == len(option_labels) - 1:
            self._append_output("Pull cancelled.")
            self._set_status("idle")
            return
        elif selected_idx == len(option_labels) - 2:
            self._append_output("Enter custom model name to pull:")
            self._set_status("entering custom model name")
            self.app.invalidate()
            selection = await self._input_queue.get()
            selection = selection.strip()
            if not selection:
                self._append_output("Pull cancelled.")
                self._set_status("idle")
                return
            pulled_model_name = selection
        else:
            pulled_model_name = displayed_models[selected_idx][0]

        self._append_output(f"\nPulling '{pulled_model_name}' from Ollama registry...")
        self._set_status("pulling model...")
        self.app.invalidate()

        # Download with AsyncClient
        try:
            client = ollama.AsyncClient()
            self._append_output(f"Connecting to Ollama daemon...")
            async for progress in await client.pull(pulled_model_name, stream=True):
                status = progress.get('status', '')
                if 'completed' in progress and 'total' in progress:
                    completed = progress['completed']
                    total = progress['total']
                    percent = (completed / total) * 100
                    bar_len = 20
                    filled = int(percent / 100 * bar_len)
                    bar = "=" * filled + "-" * (bar_len - filled)
                    self._update_last_line(f"Pulling {pulled_model_name}: [{bar}] {percent:.1f}% ({status})")
                else:
                    self._update_last_line(f"Pulling {pulled_model_name}: {status}")
                self.app.invalidate()
        except Exception as e:
            self._append_output(f"\nError pulling model: {e}")
            self._set_status("idle")
            return

        self._append_output(f"\nModel '{pulled_model_name}' successfully pulled!")
        self._append_output(f"Generating best-use summary...")
        self._set_status("summarizing...")
        self.app.invalidate()
        summary = await asyncio.to_thread(model_summarizer.get_or_create_model_summary, pulled_model_name)
        self._append_output(f"Best use: {summary}")
        self._append_output(f"You can now select or use '{pulled_model_name}'.")
        self._set_status("idle")
        self.app.invalidate()

    # ── Main agent processing ──

    async def _process_message(self, user_input: str):
        """Process a single user message through the agent loop."""
        # Slash commands
        cmd = user_input.strip()
        cmd_lower = cmd.lower()
        if cmd_lower in ["/exit", "/quit", "/q"]:
            self._append_output("Goodbye!")
            self._running = False
            self.app.exit()
            return
        if cmd_lower in ["/clear", "/c"]:
            self.agent._init_system_prompt()
            self._append_output("Chat history cleared.")
            self._set_status("idle")
            return
        if cmd_lower in ["/help", "/h"]:
            self._append_output(HELP_TEXT)
            self._set_status("idle")
            return
        if cmd_lower in ["/models", "/model", "/m"]:
            await self._show_models()
            return
        if cmd_lower in ["/tools", "/tool", "/t"]:
            await self._show_tools()
            return
        if cmd_lower in ["/mcp"]:
            await self._show_mcp()
            return

        # Handle /newmodel command (possibly with arguments)
        cmd_parts = cmd.split(None, 1)
        if cmd_parts:
            base_cmd = cmd_parts[0].lower()
            if base_cmd in ["/newmodel", "/new", "/n"]:
                cmd_arg = cmd_parts[1].strip() if len(cmd_parts) > 1 else ""
                await self._search_and_install_model(cmd_arg)
                return

        direct_tool_processed = False
        # Handle direct tool execution via /<tool_name> or direct server execution via /<mcp_server_name>
        if cmd_parts:
            base_cmd = cmd_parts[0].lower()
            if base_cmd.startswith("/"):
                potential_name = base_cmd[1:]
                
                from mcp_client import MCPManager
                manager = MCPManager.get_instance()
                mcp_tools = manager.get_all_tools()
                
                # Check if potential_name is an MCP server
                is_mcp_server = potential_name in manager.sessions
                
                # Check if it matches a valid tool
                valid_tools = {"run_command", "read_file", "write_file", "list_dir"}
                for t in mcp_tools:
                    valid_tools.add(t.get("name").lower())
                    
                tool_name = None
                if potential_name in valid_tools:
                    tool_name = potential_name
                    if tool_name not in ("run_command", "read_file", "write_file", "list_dir"):
                        for t in mcp_tools:
                            if t.get("name").lower() == tool_name:
                                tool_name = t.get("name")
                                break
                elif is_mcp_server:
                    # Select the default or primary tool for this server
                    server_session = manager.sessions[potential_name]
                    if server_session.tools:
                        # Rules: 
                        # 1. If only one tool exists, choose it
                        # 2. Try to find a tool with "search" in its name
                        # 3. Fall back to the first tool
                        chosen_tool = server_session.tools[0]
                        for t in server_session.tools:
                            if "search" in t.get("name", "").lower():
                                chosen_tool = t
                                break
                        tool_name = chosen_tool.get("name")
                        
                if tool_name:
                    direct_tool_processed = True
                    # Register user query in history so agent has context
                    self.agent.add_user_message(user_input)
                    
                    arg_string = cmd_parts[1].strip() if len(cmd_parts) > 1 else ""
                    args = parse_direct_tool_call(tool_name, arg_string)
                    
                    logging.info(f"Direct tool execution matched: {tool_name} with args {json.dumps(args)}")
                    self._append_output(f"\n[bold]Executing Tool Directly: {tool_name}[/bold]")
                    self._append_output(f"Arguments: {json.dumps(args)}")
                    
                    # Security check for file/dir tools
                    if tool_name in ["read_file", "write_file", "list_dir"]:
                        path = args.get("path", ".")
                        if not (config.is_directory_allowed(path) or self.session_permitted):
                            logging.warning(f"Direct tool execution BLOCKED: path '{path}' is outside workspace")
                            self._append_output(f"  ✗ Blocked: '{path}' outside workspace")
                            self._set_status("idle")
                            self.app.invalidate()
                            return

                    # Plan-only mode checks for direct executions
                    if self.agent.plan_only:
                        if tool_name == "write_file":
                            target_file_name = os.path.basename(args.get("path", ""))
                            if target_file_name.lower() != "plan.md":
                                logging.warning(f"Direct tool execution BLOCKED: write_file to '{target_file_name}' forbidden in plan-only mode")
                                self._append_output(f"  ✗ Blocked: write_file to '{target_file_name}' forbidden in plan-only mode (PLAN.md only)")
                                self._set_status("idle")
                                self.app.invalidate()
                                return
                        elif tool_name not in ["read_file", "list_dir", "search"]:
                            logging.warning(f"Direct tool execution BLOCKED: tool '{tool_name}' forbidden in plan-only mode")
                            self._append_output(f"  ✗ Blocked: tool '{tool_name}' forbidden in plan-only mode")
                            self._set_status("idle")
                            self.app.invalidate()
                            return
                            
                    approved = await self._ask_approval(tool_name, args)
                    if approved:
                        self._set_status(f"running {tool_name}...")
                        self.app.invalidate()
                        
                        try:
                            tool_res = await execute_tool(tool_name, args, self.agent.cwd)
                            logging.debug(f"Direct execution of {tool_name} returned: {json.dumps(tool_res)}")
                        except Exception as ex:
                            logging.exception(f"Exception while running direct tool {tool_name}")
                            tool_res = {"error": f"Internal execution exception: {str(ex)}"}
                        
                        is_ok, res_desc = format_tool_result(tool_name, tool_res)
                        if is_ok:
                            self._append_output(f"  ✓ {res_desc}")
                        else:
                            self._append_output(f"  ✗ {res_desc}")
                            
                        self.agent.add_tool_result(tool_name, tool_res)
                    else:
                        logging.info(f"Direct tool execution denied by user: {tool_name}")
                        self._append_output("  ✗ denied")
                        self.agent.add_tool_result(tool_name, {"error": "Denied by user."})
                        
                    self._set_status("idle")
                    self.app.invalidate()
 
        steps_taken = []
        start_time = time.time()
        if not direct_tool_processed:
            self.agent.add_user_message(user_input)
        processing = True
        while processing:
            self._set_status("thinking...")
            self.app.invalidate()

            think_start = time.time()
            try:
                step_result = await asyncio.to_thread(self.agent.run_step)
            except Exception as e:
                self._append_output(f"\nError: {str(e)}")
                self._set_status("idle")
                return

            think_elapsed = time.time() - think_start

            if step_result["type"] == "message":
                self._append_output(f"\n▸ Thought for {think_elapsed:.1f}s")
                self._append_output(f"\n{step_result['content']}")

                elapsed = time.time() - start_time
                if steps_taken:
                    ok_count = sum(1 for s in steps_taken if s["success"])
                    fail_count = len(steps_taken) - ok_count
                    parts = []
                    if ok_count:
                        parts.append(f"{ok_count} succeeded")
                    if fail_count:
                        parts.append(f"{fail_count} failed")
                    self._append_output(f"\n{', '.join(parts)} · {elapsed:.1f}s")
                else:
                    self._append_output(f"\n{elapsed:.1f}s")

                self.agent.add_agent_message(step_result["content"])
                processing = False
                self._set_status("idle")

            elif step_result["type"] == "tool_call":
                self.agent.add_agent_message(step_result["raw_response"])

                tool_name = step_result["tool_name"]
                args = step_result["arguments"]
                thought = step_result["thought"]

                self._append_output(f"\n▸ Thought for {think_elapsed:.1f}s")
                if thought:
                    self._append_output(f"  {thought}")

                # Security check for file/dir tools
                if tool_name in ["read_file", "write_file", "list_dir"]:
                    path = args.get("path", ".")
                    if not (config.is_directory_allowed(path) or self.session_permitted):
                        self._append_output(f"  ✗ Blocked: '{path}' outside workspace")
                        self.agent.add_tool_result(tool_name, {"error": "Access denied."})
                        steps_taken.append({"success": False})
                        continue

                # Plan-only strict checks
                if self.agent.plan_only:
                    if tool_name == "write_file":
                        target_file_name = os.path.basename(args.get("path", ""))
                        if target_file_name.lower() != "plan.md":
                            self._append_output(f"  ✗ Blocked: write_file to '{target_file_name}' forbidden in plan-only mode (PLAN.md only)")
                            self.agent.add_tool_result(tool_name, {"error": "Plan-only mode permits write_file to PLAN.md only."})
                            steps_taken.append({"success": False})
                            continue
                    elif tool_name not in ["read_file", "list_dir", "search"]:
                        self._append_output(f"  ✗ Blocked: tool '{tool_name}' forbidden in plan-only mode")
                        self.agent.add_tool_result(tool_name, {"error": f"Tool '{tool_name}' is forbidden in plan-only mode."})
                        steps_taken.append({"success": False})
                        continue

                approved = await self._ask_approval(tool_name, args)

                if approved:
                    self._set_status(f"running {tool_name}...")
                    self.app.invalidate()

                    tool_res = await execute_tool(tool_name, args, self.agent.cwd)

                    is_ok, res_desc = format_tool_result(tool_name, tool_res)
                    if is_ok:
                        self._append_output(f"  ✓ {res_desc}")
                    else:
                        self._append_output(f"  ✗ {res_desc}")

                    self.agent.add_tool_result(tool_name, tool_res)
                    steps_taken.append({"success": is_ok})
                else:
                    self._append_output(f"  ✗ denied")
                    self.agent.add_tool_result(tool_name, {"error": "Denied by user."})
                    steps_taken.append({"success": False})

                self._set_status("thinking...")

            elif step_result["type"] == "error":
                self._append_output(f"\nError: {step_result['content']}")
                processing = False
                self._set_status("idle")

    # ── Background message consumer ──

    async def _message_loop(self):
        """Background task that reads from the input queue and processes messages."""
        while self._running:
            try:
                user_input = await self._input_queue.get()
            except asyncio.CancelledError:
                return

            self._append_output(f"\nyou ❯ {user_input}")
            self.app.invalidate()

            self._current_process_task = asyncio.create_task(self._process_message(user_input))
            try:
                await self._current_process_task
            except asyncio.CancelledError:
                self._append_output("\nAgent execution cancelled.")
                self._set_status("idle")
            finally:
                self._current_process_task = None
            self.app.invalidate()

    # ── Run ──

    async def run(self):
        """Launch the full-screen app with a background message processor."""
        message_task = asyncio.ensure_future(self._message_loop())
        try:
            await self.app.run_async()
        finally:
            message_task.cancel()
            try:
                await message_task
            except asyncio.CancelledError:
                pass


# ─── Main ─────────────────────────────────────────────────────────────────────

async def main():
    cwd = os.getcwd()
    mcp = None
    
    # Simple CLI parsing
    import argparse
    parser = argparse.ArgumentParser(description="Superboof - Local AI Agent")
    parser.add_argument("--plan", "-p", action="store_true", help="Launch in STRICT plan-only/recon mode")
    args_parsed = parser.parse_args()
    plan_only = args_parsed.plan

    try:
        # 1. Query available Ollama models
        with console.status("[dim]Querying Ollama models..."):
            models = get_available_models()

        if not models:
            console.print("[red]No Ollama models found.[/red] Start Ollama and run `ollama pull <model>` first.")
            return

        # 2. Model selection (interactive selection)
        selected_model = await select_model(models)
        config.set_last_used_model(selected_model)

        # 3. Initialize MCP Manager & run permission checks
        from mcp_client import MCPManager
        mcp = MCPManager.get_instance()
        session_permitted = check_directory_permissions(cwd)

        # 4. Load MCP servers and warm up model in parallel
        with console.status(f"[dim]Loading MCP servers and warming up {selected_model}..."):
            async def run_warmup():
                try:
                    system_prompt = (
                        "You are Superboof, a local AI Agent for Linux systems.\n"
                        f"Current working directory: {cwd}\n"
                        "Introduce yourself briefly in one friendly sentence and ask how you can help."
                    )
                    res = await asyncio.to_thread(
                        ollama.generate,
                        model=selected_model,
                        prompt="Hello!",
                        system=system_prompt,
                        options={"temperature": 0.7, "num_predict": 100}
                    )
                    return res.get("response", "").strip()
                except Exception:
                    return f"Hello! I am Superboof, successfully loaded model {selected_model} and authorized for workspace {cwd}. How can I help you today?"

            load_task = asyncio.ensure_future(mcp.load_servers())
            warmup_task = asyncio.ensure_future(run_warmup())
            
            await asyncio.gather(load_task, warmup_task)
            welcome_msg = warmup_task.result()

        # 5. Initialize agent with cwd context
        agent = LocalAgent(model_name=selected_model, cwd=cwd, plan_only=plan_only)

        # 6. Clear pre-boot output and launch full-screen chat
        sys.stdout.write("\033[H\033[2J")
        sys.stdout.flush()

        chat = ChatApp(agent, session_permitted, welcome_msg)
        await chat.run()
    finally:
        if mcp is not None:
            with console.status("[dim]Stopping MCP servers..."):
                await mcp.stop_all()

if __name__ == "__main__":
    import logging
    log_dir = os.path.expanduser("~/.config/superboof")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, "superboof.log"),
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    logging.info("Superboof starting up...")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting.")
    except Exception as e:
        logging.exception("Fatal unhandled exception in main loop")
        raise e
