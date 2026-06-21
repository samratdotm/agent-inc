"""Cheap health probe for the Tinker training service. Prints OK if responsive, else DOWN.

Hits the same service (rl.beta.hud.ai/.../train/*) the forward-backward call uses, so a
clean response is a reasonable 'the pool has a pulse' signal for the supervisor to gate on.
Self-limits with an internal timeout (macOS has no `timeout` binary).
"""

import asyncio

from hud import TrainingClient


async def main() -> None:
    try:
        await asyncio.wait_for(TrainingClient("agent-inc-rl-v4").available_losses(), timeout=60)
        print("OK")
    except Exception:  # noqa: BLE001 - timeout or any failure means not healthy enough to start
        print("DOWN")


if __name__ == "__main__":
    asyncio.run(main())
