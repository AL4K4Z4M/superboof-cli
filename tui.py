import asyncio
import os
import sys
import json
from typing import Dict, Any, List
import ollama

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import Header, Footer, Input, Button, Static, Label, Select, Markdown, ProgressBar, RadioSet, RadioButton
from textual.screen import Screen
from textual.worker import get_current_worker, Worker
from textual import work, on
from textual.suggester import SuggestFromList

from agent import LocalAgent, get_available_models, get_loaded_models, run_command, read_file, write_file, list_dir
import config

class DirectoryPermissionScreen(Screen[str]):
    """Screen presented to the user to request permission for the current directory."""
    def __init__(self, directory_path: str):
        super().__init__()
        self.directory_path = os.path.abspath(directory_path)

    def compose(self) -> ComposeResult:
        yield Vertical(
            Label("[bold red]Directory Permission Request[/bold red]", classes="screen-title"),
            Label(
                f"Superboof is attempting to run in the following directory:\n\n"
                f"[bold cyan]{self.directory_path}[/bold cyan]\n\n"
                "As an AI Agent, Superboof will have capabilities to read, write, and execute files "
                "within this path. Please authorize access to proceed.",
                classes="screen-description"
            ),
            Horizontal(
                Button("Allow Always", variant="success", id="allow-always-btn"),
                Button("Allow This Session", variant="primary", id="allow-session-btn"),
                Button("Deny and Exit", variant="error", id="deny-btn"),
                classes="screen-actions"
            ),
            id="dialog-panel"
        )

    @on(Button.Pressed, "#allow-always-btn")
    def on_always(self) -> None:
        config.allow_directory(self.directory_path)
        self.dismiss("always")

    @on(Button.Pressed, "#allow-session-btn")
    def on_session(self) -> None:
        self.dismiss("session")

    @on(Button.Pressed, "#deny-btn")
    def on_deny(self) -> None:
        self.dismiss("deny")


class ModelSelectionScreen(Screen[str]):
    """Screen presented to select the Ollama model to load on boot."""
    def __init__(self, models: List[str], loaded_models: List[str], model_caps: Dict[str, List[str]]):
        super().__init__()
        self.models = models
        self.loaded_models = loaded_models
        self.model_caps = model_caps

    def compose(self) -> ComposeResult:
        import model_summarizer
        last_used = config.get_last_used_model()
        
        default_index = 0
        if last_used in self.models:
            default_index = self.models.index(last_used)
            
        model_options = []
        for i, m in enumerate(self.models):
            summary = model_summarizer.get_or_create_model_summary(m)
            caps = self.model_caps.get(m, [])
            
            features = []
            if "tools" in caps:
                features.append("tools")
            if "vision" in caps:
                features.append("vision")
            feat_str = f" [{'+'.join(features)}]" if features else ""
            
            tags = []
            if m in self.loaded_models:
                tags.append("loaded in memory")
            if m == last_used:
                tags.append("last used")
                
            tag_str = f" [{', '.join(tags)}]" if tags else ""
            label = f"{m}{feat_str}{tag_str} ({summary})"
            model_options.append(RadioButton(label, id=f"model-{m.replace(':', '_')}", value=(i == default_index)))
        
        yield Vertical(
            Label("[bold cyan]Select Ollama Model[/bold cyan]", classes="screen-title"),
            Label("Select which local Ollama model to load for Superboof's intelligence:"),
            RadioSet(*model_options, id="model-radios"),
            Button("Load Selected Model", variant="primary", id="load-model-btn"),
            id="dialog-panel"
        )

    def on_key(self, event) -> None:
        if event.key == "enter":
            self.on_load()

    @on(Button.Pressed, "#load-model-btn")
    def on_load(self) -> None:
        radios = self.query_one("#model-radios", RadioSet)
        selected_index = radios.pressed_index
        if selected_index >= 0 and selected_index < len(self.models):
            self.dismiss(self.models[selected_index])
        else:
            self.dismiss(self.models[0] if self.models else "qwen2.5:7b")



class ModelLoadingScreen(Screen[str]):
    """Screen displaying a loading bar while the model is warmed up in Ollama."""
    def __init__(self, model_name: str, cwd: str):
        super().__init__()
        self.model_name = model_name
        self.cwd = cwd

    def compose(self) -> ComposeResult:
        yield Vertical(
            Label(f"[bold green]Loading {self.model_name}...[/bold green]", classes="screen-title"),
            Label("Warming up model memory and generating greeting. Please wait...", classes="loading-subtext"),
            ProgressBar(total=100, show_percentage=True, id="loading-bar"),
            id="dialog-panel"
        )

    def on_mount(self) -> None:
        self.load_model_task()

    @work(exclusive=True)
    async def load_model_task(self) -> None:
        bar = self.query_one("#loading-bar", ProgressBar)
        
        # Start a background thread to call Ollama generate for warmup
        def warmup():
            try:
                system_prompt = (
                    "You are Superboof, a local AI Agent for Linux systems.\n"
                    f"Current working directory: {self.cwd}\n"
                    "Introduce yourself briefly in one friendly sentence and ask how you can help."
                )
                res = ollama.generate(
                    model=self.model_name,
                    prompt="Hello!",
                    system=system_prompt,
                    options={"temperature": 0.7, "num_predict": 100}
                )
                return res.get("response", "").strip()
            except Exception:
                return f"Hello! I am Superboof, successfully loaded model {self.model_name} and authorized for workspace {self.cwd}. How can I help you today?"

        warmup_task = asyncio.to_thread(warmup)
        
        # Animate the loading bar while waiting
        progress = 0
        while not warmup_task.done():
            await asyncio.sleep(0.1)
            progress = min(95, progress + 2)
            bar.progress = progress
            
        welcome_msg = await warmup_task
        bar.progress = 100
        await asyncio.sleep(0.5)
        self.dismiss(welcome_msg)


class MessageWidget(Static):
    """A widget to display a single message (User, Agent response, or Tool call/result)."""
    def __init__(self, sender: str, text: str, msg_type: str = "text", **kwargs):
        super().__init__(**kwargs)
        self.sender = sender
        self.text = text
        self.msg_type = msg_type

    def compose(self) -> ComposeResult:
        if self.sender == "User":
            title = "[bold cyan]User[/bold cyan]"
            classes = "msg-user"
        elif self.sender == "Superboof":
            title = "[bold green]Superboof[/bold green]"
            classes = "msg-agent"
        else:
            title = f"[bold yellow]{self.sender}[/bold yellow]"
            classes = "msg-system"

        self.add_class(classes)
        yield Label(title, classes="msg-title")
        yield Markdown(self.text, classes="msg-content")


class ToolCallRequestWidget(Static):
    """A widget shown when the Agent requests to execute a tool, requiring user confirmation."""
    def __init__(self, thought: str, tool_name: str, arguments: Dict[str, Any], callback, **kwargs):
        super().__init__(**kwargs)
        self.thought = thought
        self.tool_name = tool_name
        self.arguments = arguments
        self.callback = callback

    def compose(self) -> ComposeResult:
        self.add_class("tool-request-box")
        yield Label(f"[bold yellow]Tool Execution Request: {self.tool_name}[/bold yellow]", classes="tool-request-title")
        if self.thought:
            yield Markdown(f"**Reasoning:** {self.thought}")
        yield Markdown(f"**Arguments:**\n```json\n{self.arguments}\n```")
        
        yield Horizontal(
            Button("Approve", variant="success", id="approve-btn"),
            Button("Deny", variant="error", id="deny-btn"),
            classes="tool-actions"
        )

    @on(Button.Pressed, "#approve-btn")
    def on_approve(self) -> None:
        self.callback(True)
        self.remove()

    @on(Button.Pressed, "#deny-btn")
    def on_deny(self) -> None:
        self.callback(False)
        self.remove()


class SuperboofApp(App):
    CSS_PATH = "superboof.tcss"
    TITLE = "Superboof Local AI Agent"
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("c", "clear_history", "Clear Chat"),
        ("escape", "cancel_agent", "Cancel Agent")
    ]

    def __init__(self):
        super().__init__()
        self.agent = None
        self.loop_active = False
        self.session_permitted = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Horizontal(
            # Sidebar
            Vertical(
                Label("[bold underline]Status[/bold underline]", classes="sidebar-header"),
                Static("Idle", id="status-label", classes="status-idle"),
                Button("Cancel", variant="error", id="cancel-btn"),
                Label("[bold underline]Active Model[/bold underline]", classes="sidebar-header"),
                Static("None", id="active-model-lbl"),
                Label("[bold underline]System Info[/bold underline]", classes="sidebar-header"),
                Static(f"OS: Linux\nOllama: Connected\nWorkspace: {os.getcwd()}", id="sys-info"),
                id="sidebar"
            ),
            # Chat Area
            Vertical(
                ScrollableContainer(id="chat-history"),
                Horizontal(
                    Input(
                        placeholder="Type your instruction or message here...",
                        id="chat-input",
                        suggester=SuggestFromList(['/help', '/clear', '/exit', '/quit', '/tools', '/tool', '/c', '/q', '/h', '/t'])
                    ),
                    Button("Send", variant="primary", id="send-btn"),
                    id="input-container"
                ),
                id="main-chat"
            )
        )
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#cancel-btn", Button).display = False
        self.check_permissions_and_boot()

    @work(exclusive=True)
    async def check_permissions_and_boot(self) -> None:
        cwd = os.getcwd()
        
        # 1. Directory Permission check
        if not config.is_directory_allowed(cwd):
            permission_action = await self.push_screen_wait(DirectoryPermissionScreen(cwd))
            if permission_action == "deny" or permission_action is None:
                self.exit(message="Permission denied. Exiting Superboof.")
                return
            elif permission_action == "session":
                self.session_permitted = True
        else:
            self.session_permitted = True

        # 1.8 Load MCP servers
        from mcp_client import MCPManager
        await MCPManager.get_instance().load_servers()

        # 2. Query available Ollama models
        models = get_available_models()
        if not models:
            self.exit(message="No Ollama models found. Start Ollama and download a model (e.g., ollama pull qwen2.5:7b).")
            return

        # 3. Model selection screen
        loaded_models = get_loaded_models()
        
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
        
        selected_model = await self.push_screen_wait(ModelSelectionScreen(models, loaded_models, model_caps))
        if not selected_model:
            self.exit(message="No model selected. Exiting.")
            return

        # 4. Model Loading / Warmup Screen
        welcome_msg = await self.push_screen_wait(ModelLoadingScreen(selected_model, cwd))

        # 5. Initialize active agent and UI
        config.set_last_used_model(selected_model)
        self.agent = LocalAgent(model_name=selected_model)
        self.agent.add_agent_message(welcome_msg)
        self.query_one("#active-model-lbl", Static).update(selected_model)
        self.query_one("#chat-input").focus()
        
        self.add_chat_msg("Superboof", welcome_msg)

    def add_chat_msg(self, sender: str, content: str, msg_type: str = "text") -> None:
        chat_history = self.query_one("#chat-history")
        new_widget = MessageWidget(sender, content, msg_type)
        chat_history.mount(new_widget)
        chat_history.scroll_end(animate=False)

    @on(Input.Submitted, "#chat-input")
    @on(Button.Pressed, "#send-btn")
    def handle_submit(self) -> None:
        if not self.agent:
            return # Block interaction until loaded
            
        input_widget = self.query_one("#chat-input", Input)
        user_text = input_widget.value.strip()
        if not user_text:
            return
        
        input_widget.value = ""
        
        # Handle Slash commands in TUI
        if user_text.startswith("/"):
            cmd = user_text.lower().strip()
            if cmd in ["/exit", "/quit", "/q"]:
                self.exit()
                return
            elif cmd in ["/clear", "/c"]:
                self.action_clear_history()
                return
            elif cmd in ["/help", "/h"]:
                self.add_chat_msg("System", "Available commands:\n  `/clear` (`/c`) - Clear history\n  `/exit` (`/q`) - Exit Superboof\n  `/tools` (`/t`) - List available tools\n  `/help` (`/h`) - Show this help")
                return
            elif cmd in ["/tools", "/tool", "/t"]:
                from mcp_client import MCPManager
                tools = [
                    "- `run_command` (core): Execute a shell command",
                    "- `read_file` (core): View file contents",
                    "- `write_file` (core): Create/write file contents",
                    "- `list_dir` (core): List files in directory"
                ]
                mcp_tools = MCPManager.get_instance().get_all_tools()
                for t in mcp_tools:
                    tools.append(f"- `{t.get('name')}` (mcp:{t.get('server')}): {t.get('description')}")
                tools_str = "\n".join(tools)
                self.add_chat_msg("System", f"**Available Tools:**\n{tools_str}")
                return

        self.add_chat_msg("User", user_text)
        self.agent.add_user_message(user_text)
        
        # Start the Agent processing loop if not already running
        if not self.loop_active:
            self.loop_active = True
            self.run_agent_loop()

    def update_status(self, text: str, status_class: str) -> None:
        label = self.query_one("#status-label", Static)
        label.update(text)
        label.remove_class("status-idle", "status-thinking", "status-waiting", "status-running")
        label.add_class(status_class)
        
        # Programmatically toggle the Cancel button based on state
        cancel_btn = self.query_one("#cancel-btn", Button)
        if status_class == "status-idle":
            cancel_btn.display = False
        else:
            cancel_btn.display = True

    @on(Button.Pressed, "#cancel-btn")
    def handle_cancel_btn(self) -> None:
        self.action_cancel_agent()

    def action_cancel_agent(self) -> None:
        if self.loop_active:
            self.loop_active = False
            self.workers.cancel_group(self, "agent_loop")
            self.update_status("Idle", "status-idle")
            self.add_chat_msg("System", "Agent execution cancelled by user.")

    @work(exclusive=True, group="agent_loop")
    async def run_agent_loop(self) -> None:
        self.update_status("Thinking...", "status-thinking")
        
        while self.loop_active:
            step_result = await asyncio.to_thread(self.agent.run_step)
            
            if step_result["type"] == "message":
                self.add_chat_msg("Superboof", step_result["content"])
                self.agent.add_agent_message(step_result["content"])
                self.loop_active = False
                self.update_status("Idle", "status-idle")
                
            elif step_result["type"] == "tool_call":
                self.agent.add_agent_message(step_result["raw_response"])
                tool_name = step_result["tool_name"]
                args = step_result["arguments"]
                thought = step_result["thought"]
                
                # Verify that tool arguments targeting directories stay inside allowed paths
                if not self.is_tool_access_allowed(tool_name, args):
                    self.add_chat_msg("System", f"Blocked tool `{tool_name}`: Target path is outside permitted workspaces.")
                    self.agent.add_tool_result(tool_name, {"error": "Access denied. Target path is outside permitted workspaces."})
                    self.update_status("Thinking...", "status-thinking")
                    continue
                
                self.update_status("Waiting for Approval", "status-waiting")
                
                loop = asyncio.get_running_loop()
                approval_future = loop.create_future()
                
                def on_tool_decision(approved: bool):
                    if not approval_future.done():
                        approval_future.set_result(approved)
                
                chat_history = self.query_one("#chat-history")
                req_widget = ToolCallRequestWidget(thought, tool_name, args, on_tool_decision)
                chat_history.mount(req_widget)
                chat_history.scroll_end(animate=False)
                
                approved = await approval_future
                
                if approved:
                    self.update_status(f"Running {tool_name}...", "status-running")
                    self.add_chat_msg("System", f"Executing `{tool_name}` with arguments: `{args}`")
                    
                    tool_res = await self.execute_tool_locally(tool_name, args)
                    
                    res_str = json.dumps(tool_res, indent=2)
                    self.add_chat_msg("System", f"**Result from `{tool_name}`:**\n```json\n{res_str}\n```")
                    
                    self.agent.add_tool_result(tool_name, tool_res)
                    self.update_status("Thinking...", "status-thinking")
                else:
                    self.add_chat_msg("System", f"User denied execution of `{tool_name}`.")
                    self.agent.add_tool_result(tool_name, {"error": "Execution denied by user."})
                    self.update_status("Thinking...", "status-thinking")
            
            elif step_result["type"] == "error":
                self.add_chat_msg("System", f"[bold red]Error:[/bold red] {step_result['content']}")
                self.loop_active = False
                self.update_status("Idle", "status-idle")

    def is_tool_access_allowed(self, tool_name: str, args: Dict[str, Any]) -> bool:
        # Check files/directories targeted by tools to ensure they remain inside allowed workspaces
        if tool_name in ["read_file", "write_file", "list_dir"]:
            path = args.get("path", ".")
            return config.is_directory_allowed(path) or self.session_permitted
        return True

    async def execute_tool_locally(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if tool_name == "run_command":
            return await asyncio.to_thread(run_command, arguments.get("command", ""))
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

    def action_clear_history(self) -> None:
        if self.agent:
            self.query_one("#chat-history").query(MessageWidget).remove()
            self.query_one("#chat-history").query(ToolCallRequestWidget).remove()
            self.agent._init_system_prompt()
            self.add_chat_msg("System", "Chat history and agent memory cleared.")

if __name__ == "__main__":
    app = SuperboofApp()
    try:
        app.run()
    finally:
        from mcp_client import MCPManager
        import asyncio
        try:
            asyncio.run(MCPManager.get_instance().stop_all())
        except Exception:
            pass
