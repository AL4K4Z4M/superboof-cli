import asyncio
import json
import os
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger("mcp_client")

class MCPServerSession:
    def __init__(self, name: str, command: str, args: List[str], env: Optional[Dict[str, str]] = None):
        self.name = name
        self.command = command
        self.args = args
        self.env = env
        self.process: Optional[asyncio.subprocess.Process] = None
        self.tools: List[Dict[str, Any]] = []
        self.request_id = 0
        self._pending_requests: Dict[int, asyncio.Future] = {}
        self._read_task: Optional[asyncio.Task] = None
        self.last_error: Optional[str] = None
        self.stderr_log: List[str] = []
        self._stderr_task: Optional[asyncio.Task] = None

    async def start(self) -> bool:
        self.last_error = None
        self.stderr_log = []
        logger.info(f"Starting MCP server session '{self.name}' (cmd: {self.command}, args: {self.args})")
        try:
            process_env = os.environ.copy()
            if self.env:
                process_env.update(self.env)

            try:
                self.process = await asyncio.create_subprocess_exec(
                    self.command,
                    *self.args,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=process_env
                )
            except Exception as e:
                self.last_error = f"subprocess_exec failed: {str(e)}"
                logger.error(f"Failed to exec subprocess for '{self.name}': {e}")
                raise e

            self._read_task = asyncio.create_task(self._read_loop())
            self._stderr_task = asyncio.create_task(self._stderr_loop())
            
            try:
                # Initialize connection
                await self.send_request("initialize", {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "Superboof", "version": "1.0"}
                }, timeout=30.0)
                
                # Send initialized notification
                await self.send_notification("notifications/initialized")
                
                # List tools
                tools_res = await self.send_request("tools/list", {}, timeout=10.0)
                self.tools = tools_res.get("tools", [])
                logger.info(f"MCP server '{self.name}' successfully initialized with tools: {[t.get('name') for t in self.tools]}")
                return True
            except Exception as e:
                self.last_error = f"MCP initialization protocol failed: {str(e)}"
                logger.error(f"MCP server '{self.name}' initialization exception: {e}")
                if self.stderr_log:
                    self.last_error += f"\nStderr output:\n" + "\n".join(self.stderr_log[-10:])
                    logger.error(f"MCP server '{self.name}' stderr log:\n" + "\n".join(self.stderr_log))
                await self.stop()
                return False
        except Exception:
            # Fallback to shell execution if direct exec fails
            try:
                cmd_str = f"{self.command} " + " ".join(f'"{a}"' for a in self.args)
                self.process = await asyncio.create_subprocess_shell(
                    cmd_str,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=process_env
                )
                self._read_task = asyncio.create_task(self._read_loop())
                self._stderr_task = asyncio.create_task(self._stderr_loop())
                
                try:
                    await self.send_request("initialize", {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {"name": "Superboof", "version": "1.0"}
                    }, timeout=30.0)
                    await self.send_notification("notifications/initialized")
                    tools_res = await self.send_request("tools/list", {}, timeout=10.0)
                    self.tools = tools_res.get("tools", [])
                    return True
                except Exception as e:
                    self.last_error = f"MCP initialization protocol (shell fallback) failed: {str(e)}"
                    if self.stderr_log:
                        self.last_error += f"\nStderr output:\n" + "\n".join(self.stderr_log[-10:])
                    await self.stop()
                    return False
            except Exception as e:
                if not self.last_error:
                    self.last_error = f"Fallback shell execution failed: {str(e)}"
                return False

    async def _stderr_loop(self):
        while self.process and self.process.stderr:
            try:
                line = await self.process.stderr.readline()
                if not line:
                    break
                decoded = line.decode('utf-8', errors='replace').strip()
                if decoded:
                    self.stderr_log.append(decoded)
                    if len(self.stderr_log) > 50:
                        self.stderr_log.pop(0)
            except asyncio.CancelledError:
                break
            except Exception:
                pass

    async def _read_loop(self):
        while self.process and self.process.stdout:
            try:
                line = await self.process.stdout.readline()
                if not line:
                    break
                message = json.loads(line.decode('utf-8'))
                if "id" in message:
                    msg_id = message["id"]
                    if msg_id in self._pending_requests:
                        future = self._pending_requests.pop(msg_id)
                        if "error" in message:
                            future.set_exception(Exception(message["error"].get("message", "Unknown error")))
                        else:
                            future.set_result(message.get("result", {}))
            except asyncio.CancelledError:
                break
            except Exception:
                pass

    async def send_request(self, method: str, params: Dict[str, Any], timeout: float = 5.0) -> Dict[str, Any]:
        if not self.process or not self.process.stdin:
            raise Exception(f"Server {self.name} is not running")
            
        self.request_id += 1
        rid = self.request_id
        
        req = {
            "jsonrpc": "2.0",
            "id": rid,
            "method": method,
            "params": params
        }
        
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        self._pending_requests[rid] = fut
        
        data = json.dumps(req) + "\n"
        self.process.stdin.write(data.encode('utf-8'))
        await self.process.stdin.drain()
        
        return await asyncio.wait_for(fut, timeout=timeout)

    async def send_notification(self, method: str, params: Optional[Dict[str, Any]] = None):
        if not self.process or not self.process.stdin:
            return
        msg = {
            "jsonrpc": "2.0",
            "method": method
        }
        if params is not None:
            msg["params"] = params
        data = json.dumps(msg) + "\n"
        self.process.stdin.write(data.encode('utf-8'))
        await self.process.stdin.drain()

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        return await self.send_request("tools/call", {
            "name": name,
            "arguments": arguments
        }, timeout=30.0)

    async def stop(self):
        if self._read_task:
            self._read_task.cancel()
        if self._stderr_task:
            self._stderr_task.cancel()
        if self.process:
            try:
                self.process.terminate()
                try:
                    await asyncio.wait_for(self.process.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    self.process.kill()
                    await self.process.wait()
            except Exception:
                pass
            self.process = None


class MCPManager:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.sessions: Dict[str, MCPServerSession] = {}
        self.config_path = os.path.expanduser("~/.config/superboof/mcp_config.json")
        self.errored_servers = set()
        self.load_errors = {}

    async def load_servers(self):
        if not os.path.exists(self.config_path):
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, "w") as f:
                json.dump({"mcpServers": {}}, f, indent=2)
            return

        try:
            with open(self.config_path, "r") as f:
                cfg = json.load(f)
            
            self.errored_servers.clear()
            self.load_errors.clear()
            servers_cfg = cfg.get("mcpServers", {})
            for sname, scfg in servers_cfg.items():
                cmd = scfg.get("command")
                args = scfg.get("args", [])
                env = scfg.get("env")
                
                if sname in self.sessions:
                    old_session = self.sessions[sname]
                    if old_session.command == cmd and old_session.args == args and old_session.env == env:
                        continue
                    else:
                        await old_session.stop()
                        del self.sessions[sname]
                        
                if cmd:
                    session = MCPServerSession(sname, cmd, args, env)
                    success = await session.start()
                    if success:
                        self.sessions[sname] = session
                    else:
                        self.errored_servers.add(sname)
                        self.load_errors[sname] = session.last_error or "Initialization failed"
                else:
                    self.errored_servers.add(sname)
                    self.load_errors[sname] = "Missing command in configuration"
        except Exception as e:
            pass

    def get_all_tools(self) -> List[Dict[str, Any]]:
        all_tools = []
        for sname, session in self.sessions.items():
            for t in session.tools:
                tool_info = t.copy()
                tool_info["server"] = sname
                all_tools.append(tool_info)
        return all_tools

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        for sname, session in self.sessions.items():
            for t in session.tools:
                if t.get("name") == tool_name:
                    return await session.call_tool(tool_name, arguments)
        raise Exception(f"MCP Tool {tool_name} not found")

    async def stop_all(self):
        for session in list(self.sessions.values()):
            await session.stop()
        self.sessions.clear()
