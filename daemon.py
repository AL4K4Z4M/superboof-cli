import os
import json
import time
import schedule
import subprocess
import asyncio
from plyer import notification
from agent import LocalAgent

SCHEDULE_FILE = os.path.expanduser("~/.config/superboof/schedule.json")
DAEMON_PID_FILE = os.path.expanduser("~/.config/superboof/daemon.pid")

def load_tasks():
    if not os.path.exists(SCHEDULE_FILE):
        return []
    try:
        with open(SCHEDULE_FILE, 'r') as f:
            return json.load(f)
    except Exception:
        return []

def notify(title, message):
    try:
        notification.notify(
            title=title,
            message=message,
            app_name="Superboof",
            timeout=5
        )
    except Exception as e:
        print(f"Failed to send notification: {e}")

def run_task(task):
    prompt = task.get("prompt")
    if not prompt:
        return

    notify("Superboof Task Started", f"Executing scheduled task:\n{prompt}")

    try:
        # Load last used model or fallback
        import config
        model_name = config.get_last_used_model() or "qwen2.5:7b"

        agent = LocalAgent(model_name=model_name)
        # Disable tool approval for daemon
        config._cached_config = config.load_config()
        if "always_allowed_commands" not in config._cached_config:
            config._cached_config["always_allowed_commands"] = []

        async def run_agent():
            agent.add_user_message(prompt)
            from mcp_client import MCPManager
            manager = MCPManager.get_instance()

            # Use standalone tool execution implementations to avoid cli.py TTY dependencies
            from agent import run_command, read_file, write_file, list_dir
            async def exec_tool(name, args, cwd):
                if name == "run_command":
                    # Allow commands in daemon since the user explicitly scheduled and approved them
                    return await asyncio.to_thread(run_command, args.get("command", ""), cwd=cwd)
                elif name == "read_file":
                    return await asyncio.to_thread(read_file, args.get("path", ""))
                elif name == "write_file":
                    return await asyncio.to_thread(write_file, args.get("path", ""), args.get("content", ""))
                elif name == "list_dir":
                    return await asyncio.to_thread(list_dir, args.get("path", "."))
                else:
                    return await manager.call_tool(name, args)

            generator = agent.stream_response()
            response_text = ""
            async for chunk in generator:
                response_text += chunk

            tool_calls = agent.extract_tool_calls(response_text)
            for tc in tool_calls:
                tool_name = tc.get("name")
                arguments = tc.get("arguments", {})
                result = await exec_tool(tool_name, arguments, cwd=agent.cwd)
                agent.add_tool_result(tool_name, result)

            if tool_calls:
                generator2 = agent.stream_response()
                response_text2 = ""
                async for chunk in generator2:
                    response_text2 += chunk
                response_text = response_text2

            return response_text

        result_text = asyncio.run(run_agent())

        # Keep notification short
        short_result = result_text[:100] + "..." if len(result_text) > 100 else result_text
        notify("Superboof Task Completed", short_result)

    except Exception as e:
        notify("Superboof Task Failed", str(e))

def setup_schedules():
    schedule.clear()
    tasks = load_tasks()
    for task in tasks:
        freq = task.get("frequency")
        at_time = task.get("time")

        job = None
        if freq == "daily":
            job = schedule.every().day
            if at_time:
                job = job.at(at_time)
        elif freq == "hourly":
            job = schedule.every().hour
        elif freq == "minutely":
            job = schedule.every().minute

        if job:
            job.do(run_task, task)

def main():
    with open(DAEMON_PID_FILE, 'w') as f:
        f.write(str(os.getpid()))

    last_mtime = 0

    try:
        while True:
            # Check if schedule file updated
            if os.path.exists(SCHEDULE_FILE):
                mtime = os.path.getmtime(SCHEDULE_FILE)
                if mtime > last_mtime:
                    setup_schedules()
                    last_mtime = mtime

            schedule.run_pending()
            time.sleep(10)
    finally:
        if os.path.exists(DAEMON_PID_FILE):
            os.remove(DAEMON_PID_FILE)

if __name__ == "__main__":
    main()
