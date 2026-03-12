import json
import yaml
import os
import threading

import torch.distributed as dist
from tqdm import tqdm


_file_locks: dict = {}
_lock_lock = threading.Lock()


def _get_file_lock(filepath: str) -> threading.Lock:
    """Get a per-file threading lock (creates one if it doesn't exist)."""
    with _lock_lock:
        if filepath not in _file_locks:
            _file_locks[filepath] = threading.Lock()
        return _file_locks[filepath]


def load_json_file(filepath: str):
    """Load JSON from filepath with file-level locking."""
    with _get_file_lock(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)


def write_json_file(data, filepath: str) -> None:
    """Write data as JSON to filepath with file-level locking."""
    with _get_file_lock(filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


def process_jsonl(file_path: str) -> list:
    """Load a JSONL file and return a list of dicts."""
    data = []
    with _get_file_lock(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    return data


def write_jsonl(data: list, file_path: str) -> None:
    """Write a list of dicts as JSONL."""
    with _get_file_lock(file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")


def merge_jsonl(input_file_dir: str, output_filepath: str) -> None:
    """Merge all JSONL files in a directory into one output JSONL file."""
    filepaths = [os.path.join(input_file_dir, fn) for fn in os.listdir(input_file_dir)]
    merged_data = []
    for filepath in filepaths:
        with _get_file_lock(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        merged_data.append(json.loads(line))
    with _get_file_lock(output_filepath):
        with open(output_filepath, "w", encoding="utf-8") as out:
            for item in merged_data:
                out.write(json.dumps(item, ensure_ascii=False) + "\n")


def append_jsonl(data, filename: str) -> None:
    """Append a single dict as a JSONL line."""
    with _get_file_lock(filename):
        with open(filename, "a", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
            f.write("\n")


def load_txt_file_as_list(filepath: str) -> list:
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]


def load_txt_file_as_str(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def write_txt_file(data: list, filepath: str) -> None:
    with open(filepath, "a", encoding="utf-8") as f:
        for item in data:
            f.write(item + "\n")


def print_rank0(msg: str) -> None:
    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() == 0:
            print(msg)
    else:
        print(msg)


def str2list(input_str) -> list:
    if isinstance(input_str, str):
        raw_list = input_str.strip().replace("\n", "").split(",")
        return [item.strip() for item in raw_list]
    elif isinstance(input_str, list):
        return input_str
    else:
        raise TypeError("input_str should be str or list")


def get_two_words(word1: str, word2: str) -> str:
    if word1 < word2:
        return f"{word1},{word2}"
    return f"{word2},{word1}"


def load_yaml_file(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml_file(data, filepath: str) -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        yaml.dump(data, f, indent=4)


def tqdm_rank0(total: int, desc: str):
    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() == 0:
            return tqdm(total=total, desc=desc)
        return None
    return tqdm(total=total, desc=desc)


def is_main_process() -> bool:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True
