#!./.venv/Scripts/python.exe

import asyncio
from hyperviz.vizualizer import Vizualizer


if __name__ == '__main__':
    viz = Vizualizer()
    viz.launch()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(viz.cycle_event_loop())
