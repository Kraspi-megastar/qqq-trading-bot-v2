"""
runtime_settings.py — настройки, изменяемые «на лету» через сообщения боту.

Отличие от config.py: config.py читается из .env при старте и неизменен.
Здесь — параметры, которые пользователь меняет командами в чате, и которые
должны переживать перезапуск бота. Хранятся в JSON в cache_dir.

БЕЗОПАСНОСТЬ: менять эти настройки может только владелец (проверка user_id
делается в handlers, не здесь).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

logger = logging.getLogger(__name__)

_FILENAME = "runtime_settings.json"


@dataclass
class RuntimeSettings:
    # Стоп-лосс: закрыть позицию, если она в минусе на >= этот % (0 = выключен)
    stop_loss_pct: float = 0.0
    # После стопа не переоткрывать то же направление N минут (0 = переоткрывать сразу)
    stop_cooldown_min: int = 15
    # Размер позиции: % от свободных денег
    position_pct: float = 5.0
    # Максимум контрактов на сделку
    max_contracts: int = 1
    # Фильтр входов по тренду 1h: не открывать против тренда старшего ТФ
    htf_filter_on: bool = False
    htf_slope_threshold: float = 0.6
    # Бумажная (виртуальная) стратегия #2 параллельно — только сигналы, без торговли
    paper_s2_on: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def settings_path(cache_dir) -> Path:
    return Path(cache_dir) / _FILENAME


def load_settings(cache_dir, defaults: RuntimeSettings | None = None) -> RuntimeSettings:
    """Грузит настройки из файла. Если файла нет — возвращает дефолты."""
    p = settings_path(cache_dir)
    base = defaults or RuntimeSettings()
    if not p.exists():
        return base
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return RuntimeSettings(
            stop_loss_pct=float(data.get("stop_loss_pct", base.stop_loss_pct)),
            stop_cooldown_min=int(data.get("stop_cooldown_min", base.stop_cooldown_min)),
            position_pct=float(data.get("position_pct", base.position_pct)),
            max_contracts=int(data.get("max_contracts", base.max_contracts)),
            htf_filter_on=bool(data.get("htf_filter_on", base.htf_filter_on)),
            htf_slope_threshold=float(data.get("htf_slope_threshold", base.htf_slope_threshold)),
            paper_s2_on=bool(data.get("paper_s2_on", base.paper_s2_on)),
        )
    except Exception as e:
        logger.warning("runtime_settings load error: %s", repr(e))
        return base


def save_settings(cache_dir, settings: RuntimeSettings) -> bool:
    """Атомарно сохраняет настройки в файл."""
    p = settings_path(cache_dir)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(settings.to_dict(), ensure_ascii=False, indent=2),
                       encoding="utf-8")
        tmp.replace(p)
        return True
    except Exception as e:
        logger.warning("runtime_settings save error: %s", repr(e))
        return False


def format_settings(settings: RuntimeSettings) -> str:
    """Человекочитаемая сводка настроек."""
    stop = f"{settings.stop_loss_pct:.0f}%" if settings.stop_loss_pct > 0 else "выключен"
    cd = f"{settings.stop_cooldown_min} мин" if settings.stop_cooldown_min > 0 else "нет"
    htf = f"вкл (порог {settings.htf_slope_threshold:.1f})" if settings.htf_filter_on else "выкл"
    paper = "вкл" if settings.paper_s2_on else "выкл"
    return (
        "⚙️ <b>Текущие настройки</b>\n"
        f"Стоп-лосс: {stop}\n"
        f"Пауза после стопа: {cd}\n"
        f"Размер позиции: {settings.position_pct:.0f}% от свободных\n"
        f"Макс. контрактов: {settings.max_contracts}\n"
        f"Фильтр тренда 1h: {htf}\n"
        f"Бумажная #2: {paper}"
    )
