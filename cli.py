import asyncio
import os
import sys
import json
import time
from typing import Dict, Any, List
import ollama

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.completion import Completer, Completion

from rich.console import Console
from rich.markdown import Markdown
from rich.status import Status
from rich.panel import Panel
from rich.syntax import Syntax

from agent import LocalAgent, get_available_models, get_loaded_models, run_command, read_file, write_file, list_dir
import config
import model_summarizer

console = Console()

# ─── Keyboard Utilities (for menus) ─────────────────────────────────
import tty
import termios

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
    """Arrow-key driven option selector that does not clear the screen."""
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
                # Clear the menu lines so it looks clean
                sys.stdout.write(f"\x1b[{num_options}A")
                for _ in range(num_options):
                    sys.stdout.write("\r\x1b[K\n")
                sys.stdout.write(f"\x1b[{num_options}A")
                sys.stdout.flush()
                return selected

            sys.stdout.write(f"\x1b[{num_options}A")
            sys.stdout.flush()
    except KeyboardInterrupt:
        sys.stdout.write(f"\x1b[{num_options}A")
        for _ in range(num_options):
            sys.stdout.write("\r\x1b[K\n")
        sys.stdout.write(f"\x1b[{num_options}A")
        sys.stdout.flush()
        return -1
    finally:
        sys.stdout.write("\x1b[?25h")  # show cursor
        sys.stdout.flush()


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


def check_directory_permissions(cwd: str) -> bool:
    if config.is_directory_allowed(cwd):
        return True

    console.print(f"\n[bold red]Directory Permission Request[/bold red]")
    console.print(f"Superboof wants to operate in: [bold cyan]{cwd}[/bold cyan]")
    console.print("It will be able to read, write, and execute files within this path.\n")

    # This runs before the event loop, so synchronous is fine here
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
        console.print("[red]Permission denied. Exiting.[/red]")
        sys.exit(0)


async def select_model(available_models: List[str]) -> str:
    if len(available_models) == 1:
        return available_models[0]
    
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

    # To properly strip rich text for the standard sys.stdout plain writer
    from rich.text import Text
    title = Text.from_markup("\n[bold]Select Ollama Model:[/bold]").plain
    choice = await asyncio.to_thread(select_option,
        title,
        options,
        default_idx=default_idx
    )
    if choice < 0:
        return ""
    return available_models[choice]

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

class ChatCLI:
    def __init__(self, agent: LocalAgent, session_permitted: bool, welcome_msg: str):
        self.agent = agent
        self.session_permitted = session_permitted
        self.running = True

        self.allowed_tools_this_session = set()
        self.allowed_commands_this_session = set()

        history_file = os.path.expanduser("~/.config/superboof/history")
        os.makedirs(os.path.dirname(history_file), exist_ok=True)

        completer = SlashCompleter([
            '/help', '/models', '/m', '/clear', '/exit', '/quit', '/q',
            '/mcp', '/tools', '/tool', '/t', '/newmodel', '/new', '/n'
        ])

        self.session = PromptSession(
            history=FileHistory(history_file),
            completer=completer,
            bottom_toolbar=self.get_bottom_toolbar
        )

        console.print(Markdown(f"**Superboof**: {welcome_msg}"))

    def get_bottom_toolbar(self):
        mcp_connected = 0
        try:
            from mcp_client import MCPManager
            mcp_connected = len(MCPManager.get_instance().sessions)
        except Exception:
            pass
        return HTML(f' <b>Superboof</b> | <style bg="ansiblue" fg="ansiwhite"> Model: {self.agent.model_name} </style> | MCP: {mcp_connected} connected ')

    async def _ask_approval(self, tool_name: str, args: Dict[str, Any]) -> bool:
        if tool_name == "run_command":
            cmd_str = args.get("command", "")
            if cmd_str in self.allowed_commands_this_session:
                return True
            always_allowed_cmds = config.get_always_allowed_commands()
            if cmd_str in always_allowed_cmds:
                return True
        else:
            if tool_name in self.allowed_tools_this_session:
                return True
            always_allowed = config.get_always_allowed_tools()
            if tool_name in always_allowed:
                return True

        console.print(Panel(Syntax(json.dumps(args, indent=2), "json", theme="monokai", background_color="default"), title=f"Tool Request: {tool_name}", border_style="yellow"))
        
        options = [
            "Allow",
            "Allow this session",
            "Always allow (Persistent)",
            "Deny"
        ]
        
        from rich.text import Text
        title = Text.from_markup(f"Action for [bold yellow]{tool_name}[/bold yellow]?").plain
        choice_idx = await asyncio.to_thread(select_option, title, options, default_idx=0)
        
        if choice_idx < 0 or choice_idx == 3:
            console.print("[red]✗ Denied[/red]")
            return False
            
        if choice_idx == 0:
            console.print("[green]✓ Approved[/green]")
            return True
        elif choice_idx == 1:
            if tool_name == "run_command":
                cmd_str = args.get("command", "")
                self.allowed_commands_this_session.add(cmd_str)
            else:
                self.allowed_tools_this_session.add(tool_name)
            console.print(f"[green]✓ Approved for this session[/green]")
            return True
        elif choice_idx == 2:
            if tool_name == "run_command":
                cmd_str = args.get("command", "")
                config.allow_command_persistently(cmd_str)
            else:
                config.allow_tool_persistently(tool_name)
            console.print(f"[green]✓ Approved persistently[/green]")
            return True
            
        return False


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

        from rich.text import Text
        title = Text.from_markup("\n[bold]Configured MCP Servers:[/bold]").plain
        selected_idx = await asyncio.to_thread(select_option, title, option_labels, default_idx=0)

        if selected_idx < 0 or selected_idx == len(option_labels) - 1:
            console.print("MCP browsing cancelled.")
            return

        if selected_idx == len(option_labels) - 2:
            console.print("\n[bold]Add New MCP Server[/bold]")
            
            sname = await self.session.prompt_async("Enter server name (e.g. weather-server): ")
            sname = sname.strip()
            if not sname:
                console.print("Cancelled.")
                return

            cmd = await self.session.prompt_async("Enter command/executable (e.g. npx, node, python3): ")
            cmd = cmd.strip()
            if not cmd:
                console.print("Cancelled.")
                return

            args_raw = await self.session.prompt_async("Enter arguments (optional, space-separated): ")
            args_raw = args_raw.strip()
            import shlex
            try:
                args = shlex.split(args_raw)
            except Exception:
                args = args_raw.split()

            console.print(f"Adding MCP server '{sname}' ({cmd} {' '.join(args)})...")

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
                    console.print(f"[green]Successfully loaded and started MCP server '{sname}'![/green]")
                else:
                    console.print(f"[yellow]Saved config, but failed to start MCP server '{sname}'. Check command or args.[/yellow]")
                    err_msg = manager.load_errors.get(sname)
                    if err_msg:
                        console.print(f"[red]Error: {err_msg}[/red]")
            except Exception as e:
                console.print(f"[red]Error saving/loading MCP server: {e}[/red]")
        else:
            sname = servers[selected_idx]
            scfg = config_servers[sname]
            cmd = scfg.get("command", "")
            args = scfg.get("args", [])
            is_active = sname in manager.sessions
            
            console.print(f"\n[bold]MCP Server Details: {sname}[/bold]")
            console.print(f"Status:  {'Active' if is_active else 'Inactive'}")
            console.print(f"Command: {cmd}")
            console.print(f"Args:    {' '.join(args)}")
            
            if is_active:
                tools = [t for t in manager.sessions[sname].tools]
                console.print(f"Provided Tools ({len(tools)}):")
                for t in tools:
                    console.print(f"  - {t.get('name')}: {t.get('description')}")
            else:
                console.print("This server is configured but not currently running.")
                err_msg = manager.load_errors.get(sname)
                if err_msg:
                    console.print(f"Error Log:\n{err_msg}")
            
            details_options = [
                "Edit Display Name / Rename",
                "Delete Server",
                "Back to List"
            ]
            
            from rich.text import Text
            title = Text.from_markup(f"\n[bold]Action for MCP server '{sname}':[/bold]").plain
            action_idx = await asyncio.to_thread(select_option, title, details_options, default_idx=0)
            
            if action_idx == 0:
                new_sname = await self.session.prompt_async(f"\nEnter new display name for '{sname}': ")
                new_sname = new_sname.strip()
                if not new_sname or new_sname == sname:
                    console.print("Rename cancelled.")
                    return
                
                console.print(f"Renaming '{sname}' to '{new_sname}'...")
                
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
                        
                        console.print(f"[green]Successfully renamed to '{new_sname}'![/green]")
                except Exception as e:
                    console.print(f"[red]Error renaming server: {e}[/red]")
                    
            elif action_idx == 1:
                confirm = await self.session.prompt_async(f"\nAre you sure you want to delete MCP server '{sname}'? [y/n]: ")
                confirm = confirm.strip().lower()
                if confirm not in ("y", "yes"):
                    console.print("Delete cancelled.")
                    return
                
                console.print(f"Deleting '{sname}'...")
                
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
                        
                        console.print(f"[green]Successfully deleted MCP server '{sname}'.[/green]")
                except Exception as e:
                    console.print(f"[red]Error deleting server: {e}[/red]")


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
        
        from rich.text import Text
        title = Text.from_markup("\n[bold]Available Tools (Select to view detailed schema):[/bold]").plain
        selected_idx = await asyncio.to_thread(select_option, title, option_labels, default_idx=0)
        
        if selected_idx >= 0 and selected_idx < len(tools_list):
            t = tools_list[selected_idx]
            console.print(Panel(
                f"[bold]Source:[/bold]      {t['source']}\n[bold]Description:[/bold] {t['description']}\n[bold]Schema:[/bold]\n{json.dumps(t['schema'], indent=2)}",
                title=f"Tool Details: {t['name']}", border_style="cyan"
            ))
        else:
            console.print("\nBrowsing cancelled.")


    async def _search_and_install_model(self, initial_query: str):
        """Search Ollama Library and optionally pull a model."""
        import urllib.request
        import urllib.parse
        import html as html_lib
        import re
        import model_summarizer

        query = initial_query
        if not query:
            console.print("\n[bold]Search Ollama Library[/bold]")
            query = await self.session.prompt_async("Enter search term (or press Enter to show top 10 uninstalled models): ")
            query = query.strip()

        url = "https://ollama.com/library"
        if query:
            url += f"?q={urllib.parse.quote(query)}"

        with Status("[bold cyan]Searching ollama.com...", spinner="dots", console=console):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Superboof/1.0"})
                resp = await asyncio.to_thread(urllib.request.urlopen, req, timeout=7)
                html_content = resp.read().decode('utf-8')
            except Exception as e:
                console.print(f"\n[red]Error fetching Ollama library:[/red] {e}")
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
            console.print(f"\nNo models found on Ollama library for '{query}'.")
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
                console.print(f"\nAll models matching '{query}' are already installed.")
            else:
                console.print(f"\nAll top models from Ollama Library are already installed.")
            return

        displayed_models = uninstalled_results[:10]

        import os
        try:
            mem_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
            mem_gb = mem_bytes / (1024 ** 3)
        except Exception:
            mem_gb = 8.0

        option_labels = []
        for name, display_name, desc in displayed_models:
            param_match = re.search(r'\b(\d+(?:\.\d+)?)\s*[Bb]\b', desc)
            if param_match:
                params = float(param_match.group(1))
                size_str = f"{params}B"
            else:
                lower_name = name.lower()
                if "plus" in lower_name or "70b" in lower_name: params = 70.0
                elif "35b" in lower_name or "command-r" in lower_name: params = 35.0
                elif "14b" in lower_name or "phi4" in lower_name: params = 14.0
                elif "9b" in lower_name or "gemma2" in lower_name: params = 9.0
                elif "8b" in lower_name or "llama3.1" in lower_name: params = 8.0
                elif "7b" in lower_name or "mistral" in lower_name or "deepseek" in lower_name: params = 7.0
                elif "3b" in lower_name or "llama3.2" in lower_name or "mini" in lower_name: params = 3.0
                elif "1b" in lower_name or "tiny" in lower_name or "smollm" in lower_name: params = 1.5
                else: params = 7.0
                size_str = "~7B"

            required_ram = params * 1.2
            if required_ram < mem_gb * 0.7: compat = "Compatible"
            elif required_ram < mem_gb * 1.0: compat = "Compatible"
            elif required_ram < mem_gb * 1.5: compat = "May run slow"
            else: compat = "High RAM required"

            option_labels.append(f"{name:<20} ({size_str}) - {compat}")
        
        option_labels.append("Search again or enter custom model name...")
        option_labels.append("Cancel Pull")

        from rich.text import Text
        title = Text.from_markup("\n[bold]Select Model to Pull:[/bold]").plain
        selected_idx = await asyncio.to_thread(select_option, title, option_labels, default_idx=0)

        if selected_idx < 0 or selected_idx == len(option_labels) - 1:
            console.print("Pull cancelled.")
            return
        elif selected_idx == len(option_labels) - 2:
            selection = await self.session.prompt_async("Enter custom model name to pull: ")
            selection = selection.strip()
            if not selection:
                console.print("Pull cancelled.")
                return
            pulled_model_name = selection
        else:
            pulled_model_name = displayed_models[selected_idx][0]

        console.print(f"\nPulling '{pulled_model_name}' from Ollama registry...")

        try:
            client = ollama.AsyncClient()
            with Status(f"[bold green]Pulling {pulled_model_name}...", spinner="dots", console=console) as status:
                async for progress in await client.pull(pulled_model_name, stream=True):
                    prog_status = progress.get('status', '')
                    if 'completed' in progress and 'total' in progress:
                        completed = progress['completed']
                        total = progress['total']
                        percent = (completed / total) * 100
                        status.update(f"[bold green]Pulling {pulled_model_name}: {percent:.1f}% ({prog_status})")
                    else:
                        status.update(f"[bold green]Pulling {pulled_model_name}: {prog_status}")
        except Exception as e:
            console.print(f"\n[red]Error pulling model:[/red] {e}")
            return

        console.print(f"\n[green]Model '{pulled_model_name}' successfully pulled![/green]")
        with Status("[cyan]Generating best-use summary...", spinner="dots", console=console):
            summary = await asyncio.to_thread(model_summarizer.get_or_create_model_summary, pulled_model_name)
        console.print(f"Best use: {summary}")

    async def process_message(self, user_input: str):
        if user_input:
            self.agent.add_user_message(user_input)
        processing = True

        while processing:
            with Status("[bold cyan]Thinking...", spinner="dots", console=console):
                try:
                    step_result = await asyncio.to_thread(self.agent.run_step)
                except Exception as e:
                    console.print(f"[red]Error:[/red] {str(e)}")
                    return

            if step_result["type"] == "message":
                console.print(Markdown(step_result['content']))
                self.agent.add_agent_message(step_result["content"])
                processing = False

            elif step_result["type"] == "tool_call":
                self.agent.add_agent_message(step_result["raw_response"])

                tool_name = step_result["tool_name"]
                args = step_result["arguments"]
                thought = step_result["thought"]

                if thought:
                    console.print(Markdown(f"_{thought}_"))

                if tool_name in ["read_file", "write_file", "list_dir"]:
                    path = args.get("path", ".")
                    if not (config.is_directory_allowed(path) or self.session_permitted):
                        console.print(f"[red]✗ Blocked:[/red] '{path}' outside workspace")
                        self.agent.add_tool_result(tool_name, {"error": "Access denied."})
                        continue

                # Plan-only strict checks
                if self.agent.plan_only:
                    if tool_name == "write_file":
                        target_file_name = os.path.basename(args.get("path", ""))
                        if target_file_name.lower() != "plan.md":
                            console.print(f"[red]✗ Blocked: write_file to '{target_file_name}' forbidden in plan-only mode (PLAN.md only)[/red]")
                            self.agent.add_tool_result(tool_name, {"error": "Plan-only mode permits write_file to PLAN.md only."})
                            continue
                    elif tool_name not in ["read_file", "list_dir", "search"]:
                        console.print(f"[red]✗ Blocked: tool '{tool_name}' forbidden in plan-only mode[/red]")
                        self.agent.add_tool_result(tool_name, {"error": f"Tool '{tool_name}' is forbidden in plan-only mode."})
                        continue

                approved = await self._ask_approval(tool_name, args)

                if approved:
                    with Status(f"[bold green]Running {tool_name}...", spinner="dots", console=console):
                        tool_res = await execute_tool(tool_name, args, self.agent.cwd)

                    if "error" in tool_res:
                        console.print(f"[red]✗ Error:[/red] {tool_res['error']}")
                    else:
                        console.print(f"[green]✓ Completed {tool_name}[/green]")
                    self.agent.add_tool_result(tool_name, tool_res)
                else:
                    self.agent.add_tool_result(tool_name, {"error": "Denied by user."})

            elif step_result["type"] == "error":
                console.print(f"[red]Error:[/red] {step_result['content']}")
                processing = False

    async def run(self):
        while self.running:
            try:
                # Run the prompt block in a thread/executor context native to prompt_toolkit async loop
                user_input = await self.session.prompt_async('❯ ')
                user_input = user_input.strip()
                if not user_input:
                    continue

                cmd_parts = user_input.split(None, 1)
                cmd_lower = cmd_parts[0].lower() if cmd_parts else ""

                if cmd_lower in ["/exit", "/quit", "/q"]:
                    console.print("Goodbye!")
                    self.running = False
                    break
                elif cmd_lower in ["/clear", "/c"]:
                    self.agent._init_system_prompt()
                    console.print("[dim]Chat history cleared.[/dim]")
                    continue
                elif cmd_lower in ["/help", "/h"]:
                    console.print("Commands: /help, /models, /tools, /newmodel, /mcp, /clear, /exit")
                    continue
                elif cmd_lower in ["/models", "/m"]:
                    models = await asyncio.to_thread(get_available_models)
                    selected_model = await select_model(models)
                    if selected_model and selected_model != self.agent.model_name:
                        self.agent = LocalAgent(model_name=selected_model, cwd=self.agent.cwd)
                        config.set_last_used_model(selected_model)
                        console.print(f"[green]Switched model to:[/green] {selected_model}")
                    continue
                elif cmd_lower in ["/tools", "/tool", "/t"]:
                    await self._show_tools()
                    continue
                elif cmd_lower in ["/mcp"]:
                    await self._show_mcp()
                    continue
                elif cmd_lower in ["/newmodel", "/new", "/n"]:
                    cmd_arg = cmd_parts[1].strip() if len(cmd_parts) > 1 else ""
                    await self._search_and_install_model(cmd_arg)
                    continue

                # Check for direct tool execution via /<tool_name>
                direct_tool_processed = False
                if cmd_lower.startswith("/"):
                    potential_name = cmd_lower[1:]

                    from mcp_client import MCPManager
                    manager = MCPManager.get_instance()
                    mcp_tools = manager.get_all_tools()

                    is_mcp_server = potential_name in manager.sessions
                    valid_tools = {"run_command", "read_file", "write_file", "list_dir"}
                    for t in mcp_tools:
                        valid_tools.add(t.get("name", "").lower())

                    tool_name = None
                    if potential_name in valid_tools:
                        tool_name = potential_name
                        if tool_name not in ("run_command", "read_file", "write_file", "list_dir"):
                            for t in mcp_tools:
                                if t.get("name", "").lower() == tool_name:
                                    tool_name = t.get("name")
                                    break
                    elif is_mcp_server:
                        server_session = manager.sessions[potential_name]
                        if server_session.tools:
                            chosen_tool = server_session.tools[0]
                            for t in server_session.tools:
                                if "search" in t.get("name", "").lower():
                                    chosen_tool = t
                                    break
                            tool_name = chosen_tool.get("name")

                    if tool_name:
                        direct_tool_processed = True
                        self.agent.add_user_message(user_input)

                        arg_string = cmd_parts[1].strip() if len(cmd_parts) > 1 else ""
                        args = parse_direct_tool_call(tool_name, arg_string)

                        console.print(f"\n[bold]Executing Tool Directly: {tool_name}[/bold]")
                        console.print(f"Arguments: {json.dumps(args)}")

                        if tool_name in ["read_file", "write_file", "list_dir"]:
                            path = args.get("path", ".")
                            if not (config.is_directory_allowed(path) or self.session_permitted):
                                console.print(f"[red]  ✗ Blocked: '{path}' outside workspace[/red]")
                                continue

                        # Plan-only mode checks for direct executions
                        if self.agent.plan_only:
                            if tool_name == "write_file":
                                target_file_name = os.path.basename(args.get("path", ""))
                                if target_file_name.lower() != "plan.md":
                                    console.print(f"[red]  ✗ Blocked: write_file to '{target_file_name}' forbidden in plan-only mode (PLAN.md only)[/red]")
                                    continue
                            elif tool_name not in ["read_file", "list_dir", "search"]:
                                console.print(f"[red]  ✗ Blocked: tool '{tool_name}' forbidden in plan-only mode[/red]")
                                continue

                        approved = await self._ask_approval(tool_name, args)
                        if approved:
                            with Status(f"[bold green]Running {tool_name}...", spinner="dots", console=console):
                                try:
                                    tool_res = await execute_tool(tool_name, args, self.agent.cwd)
                                except Exception as ex:
                                    tool_res = {"error": f"Internal execution exception: {str(ex)}"}

                            if "error" in tool_res:
                                console.print(f"[red]  ✗ Error:[/red] {tool_res['error']}")
                            else:
                                console.print(f"[green]  ✓ Completed {tool_name}[/green]")

                            self.agent.add_tool_result(tool_name, tool_res)
                        else:
                            console.print("[red]  ✗ denied[/red]")
                            self.agent.add_tool_result(tool_name, {"error": "Denied by user."})

                if not direct_tool_processed:
                    await self.process_message(user_input)
                else:
                    # After a direct tool call, trigger the agent to react to the result
                    await self.process_message("")

            except (KeyboardInterrupt, EOFError):
                console.print("\nGoodbye!")
                self.running = False
                break

async def main():
    cwd = os.getcwd()

    with Status("[dim]Querying Ollama models...", console=console):
        models = get_available_models()

    if not models:
        console.print("[red]No Ollama models found.[/red]")
        return

    selected_model = await select_model(models)
    config.set_last_used_model(selected_model)

    session_permitted = check_directory_permissions(cwd)

    from mcp_client import MCPManager
    mcp = MCPManager.get_instance()
    
    with Status(f"[dim]Loading MCP servers and warming up {selected_model}...", console=console):
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
                return f"Hello! I am Superboof, successfully loaded {selected_model}. How can I help you?"

        _, welcome_msg = await asyncio.gather(mcp.load_servers(), run_warmup())

    # Parse CLI arguments within main scope
    import argparse
    parser = argparse.ArgumentParser(description="Superboof - Local AI Agent")
    parser.add_argument("--plan", "-p", action="store_true", help="Launch in STRICT plan-only/recon mode")
    # For when run without __name__ == __main__ checks via runner scripts
    args_parsed, _ = parser.parse_known_args()

    agent = LocalAgent(model_name=selected_model, cwd=cwd, plan_only=args_parsed.plan)

    console.clear()
    console.print(r"""
     _                 __
    | |               / _|
    | |__   ___   ___|  _|
    | '_ \ / _ \ / _ \ |
    | |_) | (_) | (_) | |
    |_.__/ \___/ \___/|_|
    """)
    console.print(f"[dim]Model: {selected_model} | Workspace: {cwd}[/dim]\n")

    cli = ChatCLI(agent, session_permitted, welcome_msg)
    try:
        await cli.run()
    finally:
        with Status("[dim]Stopping MCP servers...", console=console):
            await mcp.stop_all()


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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Superboof - Local AI Agent")
    parser.add_argument("--plan", "-p", action="store_true", help="Launch in STRICT plan-only/recon mode")
    args_parsed = parser.parse_args()

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
