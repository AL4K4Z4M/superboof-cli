import os
import json
from typing import List, Optional

CONFIG_PATH = os.path.expanduser("~/.config/superboof/config.json")

# In-memory cache to avoid re-reading disk on every check
_cached_config: Optional[dict] = None
_cached_mtime: float = 0.0

def load_config() -> dict:
    global _cached_config, _cached_mtime

    # Check if file exists
    if not os.path.exists(CONFIG_PATH):
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        default_config = {"allowed_directories": []}
        save_config(default_config)
        _cached_config = default_config
        _cached_mtime = os.path.getmtime(CONFIG_PATH)
        return default_config

    # Use cached version if file hasn't changed
    try:
        mtime = os.path.getmtime(CONFIG_PATH)
        if _cached_config is not None and mtime == _cached_mtime:
            return _cached_config
    except OSError:
        pass

    try:
        with open(CONFIG_PATH, "r") as f:
            _cached_config = json.load(f)
            _cached_mtime = os.path.getmtime(CONFIG_PATH)
            return _cached_config
    except Exception:
        return {"allowed_directories": []}

def save_config(config: dict):
    global _cached_config, _cached_mtime
    try:
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        with open(CONFIG_PATH, "w") as f:
            json.dump(config, f, indent=2)
        _cached_config = config
        _cached_mtime = os.path.getmtime(CONFIG_PATH)
    except Exception:
        pass

def is_directory_allowed(path: str) -> bool:
    abs_path = os.path.abspath(path)
    config = load_config()
    allowed_dirs = config.get("allowed_directories", [])

    for allowed in allowed_dirs:
        allowed_abs = os.path.abspath(allowed)
        if abs_path == allowed_abs or abs_path.startswith(allowed_abs + os.sep):
            return True
    return False

def allow_directory(path: str):
    abs_path = os.path.abspath(path)
    config = load_config()
    allowed_dirs = config.setdefault("allowed_directories", [])
    if abs_path not in allowed_dirs:
        allowed_dirs.append(abs_path)
        save_config(config)

def revoke_directory(path: str):
    """Remove a directory from the allowed list."""
    abs_path = os.path.abspath(path)
    config = load_config()
    allowed_dirs = config.get("allowed_directories", [])
    if abs_path in allowed_dirs:
        allowed_dirs.remove(abs_path)
        save_config(config)

def get_allowed_directories() -> List[str]:
    """Return the list of currently allowed directories."""
    config = load_config()
    return config.get("allowed_directories", [])

def get_last_used_model() -> Optional[str]:
    """Get the last used model from config."""
    config = load_config()
    return config.get("last_used_model")

def set_last_used_model(model_name: str):
    """Save the last used model in config."""
    config = load_config()
    config["last_used_model"] = model_name
    save_config(config)

def get_always_allowed_tools() -> List[str]:
    """Get the list of always allowed tool names."""
    config = load_config()
    return config.get("always_allowed_tools", [])

def allow_tool_persistently(tool_name: str):
    """Add a tool to the always allowed list persistently."""
    config = load_config()
    always_allowed = config.setdefault("always_allowed_tools", [])
    if tool_name not in always_allowed:
        always_allowed.append(tool_name)
        save_config(config)

def get_always_allowed_commands() -> List[str]:
    """Get the list of always allowed shell commands."""
    config = load_config()
    return config.get("always_allowed_commands", [])

def allow_command_persistently(command: str):
    """Add a command to the always allowed list persistently."""
    config = load_config()
    always_allowed = config.setdefault("always_allowed_commands", [])
    if command not in always_allowed:
        always_allowed.append(command)
        save_config(config)
