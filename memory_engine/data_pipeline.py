import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class PipelineStats:
    total_dialogues_seen: int = 0
    dialogues_kept: int = 0
    turns_kept: int = 0
    turns_dropped_too_short: int = 0
    turns_dropped_noisy: int = 0


def _normalize_speaker(raw_speaker: str) -> str:
    raw = (raw_speaker or "").strip().lower()
    if raw in {"human", "user"}:
        return "human"
    if raw in {"assistant", "gpt", "chatgpt", "bot"}:
        return "assistant"
    return "system"


def _is_too_short(text: str, min_chars: int) -> bool:
    return len(text.strip()) < min_chars


def _is_noisy(text: str) -> bool:
    if not text.strip():
        return True
    if "\x00" in text:
        return True
    replacement_char_ratio = text.count("\ufffd") / max(len(text), 1)
    if replacement_char_ratio > 0.01:
        return True
    control_chars = sum(1 for ch in text if ord(ch) < 32 and ch not in "\n\t\r")
    if control_chars > 0:
        return True
    repeated_symbol_noise = bool(re.search(r"([^\w\s])\1{15,}", text))
    return repeated_symbol_noise


def load_sharegpt(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected top-level list in ShareGPT file.")
    return data


def build_sample_turns(
    raw_dialogues: list[dict[str, Any]],
    sample_size: int = 300,
    min_turn_chars: int = 8,
    min_turns_per_dialogue: int = 2,
    shuffle: bool = False,
    seed: int | None = None,
    prefer_memory_turns: bool = False,
) -> tuple[list[dict[str, Any]], PipelineStats]:
    stats = PipelineStats()
    normalized_turns: list[dict[str, Any]] = []
    now_iso = datetime.now(timezone.utc).isoformat()

    for dialogue in raw_dialogues[:sample_size]:
        stats.total_dialogues_seen += 1
        dialogue_id = dialogue.get("id", f"dialogue_{stats.total_dialogues_seen}")
        turns = dialogue.get("conversations", [])
        if not isinstance(turns, list):
            continue

        local_turns: list[dict[str, Any]] = []
        for turn_idx, turn in enumerate(turns):
            text = str(turn.get("value", "")).strip()
            if _is_too_short(text, min_turn_chars):
                stats.turns_dropped_too_short += 1
                continue
            if _is_noisy(text):
                stats.turns_dropped_noisy += 1
                continue

            local_turns.append(
                {
                    "dialogue_id": dialogue_id,
                    "turn_id": f"{dialogue_id}::turn_{turn_idx}",
                    "turn_index": turn_idx,
                    "speaker": _normalize_speaker(str(turn.get("from", ""))),
                    "text": text,
                    "timestamp_utc": now_iso,
                }
            )

        if len(local_turns) < min_turns_per_dialogue:
            continue

        stats.dialogues_kept += 1
        stats.turns_kept += len(local_turns)
        normalized_turns.extend(local_turns)

    return normalized_turns, stats


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")
