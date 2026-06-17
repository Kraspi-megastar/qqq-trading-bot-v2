from __future__ import annotations
from typing import Any, Awaitable, Callable, Dict

from aiogram import BaseMiddleware
from aiogram.types import TelegramObject


class AppMiddleware(BaseMiddleware):
    def __init__(self, app: Any):
        super().__init__()
        self._app = app

    async def __call__(
        self,
        handler: Callable[[TelegramObject, Dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: Dict[str, Any],
    ) -> Any:
        data["app"] = self._app
        return await handler(event, data)
