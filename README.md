# Superboof — Local AI Agent CLI

Superboof is a local AI agent for Linux that runs entirely on your machine via [Ollama](https://ollama.com). No API keys, no cloud — just your hardware and your models.

## Features

- **100% Local** — Runs privately via Ollama. No data leaves your machine.
- **Interactive Tool Approval** — Every command and file operation requires your explicit permission before execution.
- **Task Summaries** — After each request, see a clear summary of what was done, whether it succeeded, and how long it took.
- **Keyboard-Driven** — Arrow-key menus for model selection, permissions, and tool approvals. Full prompt history with ↑/↓.
- **Security Boundaries** — Directory-level access control. File tools are blocked outside permitted workspaces.
- **Conversation Memory** — Maintains context across exchanges with automatic trimming to stay within model limits.

## Tools

| Tool | Description |
|------|-------------|
| `run_command` | Execute shell commands with output capture (60s timeout) |
| `read_file` | Read file contents (up to 20k chars) |
| `write_file` | Create or modify files |
| `list_dir` | List directory contents |

## Getting Started

1. Install and start [Ollama](https://ollama.com) with at least one model:
   ```bash
   ollama pull qwen2.5:7b
   ```

2. Run:
   ```bash
   ./run.sh
   ```

   Or install globally:
   ```bash
   ln -sf "$(pwd)/run.sh" ~/.local/bin/boof
   ```

## CLI Commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/clear` | Clear chat history and agent memory |
| `/exit` | Exit (also `/quit`, `/q`) |

## Keyboard

| Key | Action |
|-----|--------|
| `↑` / `↓` | Navigate history or menu options |
| `Enter` | Submit or confirm |
| `Ctrl+C` | Cancel current agent operation |
| `Ctrl+D` | Exit |
