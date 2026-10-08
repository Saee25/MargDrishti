import yaml
from pathlib import Path

class Config(dict):
    """
    Configuration object allowing attribute access to dictionary keys.
    """
    def __init__(self, *args, **kwargs):
        super(Config, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v
        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        val = self.get(attr)
        if isinstance(val, dict):
            return Config(val)
        return val

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
        super(Config, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Config, self).__delitem__(key)
        del self.__dict__[key]

def deep_merge(base, update):
    """
    Recursively merge two dictionaries.
    """
    merged = base.copy()
    for k, v in update.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged

def set_nested_value(d, keys, value):
    """
    Set a value in a nested dictionary using a list of keys.
    """
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value

def get_project_root() -> Path:
    """
    Returns the project root directory based on the location of this file.
    """
    return Path(__file__).resolve().parent.parent

def load_config(experiment_path: str = None, overrides: list = None) -> Config:
    """
    Loads base.yaml, deep-merges an optional experiment YAML, applies overrides,
    resolves paths to absolute paths relative to project root, and returns a Config object.
    """
    project_root = get_project_root()
    base_path = project_root / "configs" / "base.yaml"
    
    with open(base_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
        
    if experiment_path:
        exp_path = Path(experiment_path)
        if not exp_path.is_absolute():
            exp_path = project_root / exp_path
        if exp_path.exists():
            with open(exp_path, "r", encoding="utf-8") as f:
                exp_dict = yaml.safe_load(f)
                config_dict = deep_merge(config_dict, exp_dict)
                
    if overrides:
        for override in overrides:
            if "=" in override:
                key_path, value_str = override.split("=", 1)
                keys = key_path.split(".")
                
                # Try to cast value
                if value_str.lower() == "true":
                    value = True
                elif value_str.lower() == "false":
                    value = False
                else:
                    try:
                        value = int(value_str)
                    except ValueError:
                        try:
                            value = float(value_str)
                        except ValueError:
                            value = value_str
                            
                set_nested_value(config_dict, keys, value)

    # Resolve paths relative to project root
    if "paths" in config_dict:
        for k, v in config_dict["paths"].items():
            config_dict["paths"][k] = str((project_root / v).resolve())
            
    return Config(config_dict)

def dump_config(config: Config, save_path: str | Path):
    """
    Dumps the Config object back to a YAML file.
    """
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert Config back to dict
    def config_to_dict(c):
        if isinstance(c, Config) or isinstance(c, dict):
            return {k: config_to_dict(v) for k, v in c.items()}
        return c
        
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config_to_dict(config), f, default_flow_style=False, sort_keys=False)
