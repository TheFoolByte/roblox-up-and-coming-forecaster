# roblox-up-and-coming-forecaster

[![make.fun · roblox-up-and-coming-forecast-01](https://make.fun/badge/roblox-up-and-coming-forecast-01.svg)](https://make.fun/bounty/roblox-up-and-coming-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01)
[![rank](https://make.fun/badge/roblox-up-and-coming-forecast-01/TheFoolByte/roblox-up-and-coming-forecaster/rank.svg)](https://make.fun/bounty/roblox-up-and-coming-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01)
[![reward](https://make.fun/badge/roblox-up-and-coming-forecast-01/TheFoolByte/roblox-up-and-coming-forecaster/reward.svg)](https://make.fun/bounty/roblox-up-and-coming-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01)

A Monte Carlo log-rank diffusion forecaster for Roblox's Up-and-Coming top 10. It reads the chart history, simulates each game's rank as a random walk in log-rank space with momentum and playing-growth drift, boosts games whose playing is surging, drops games whose 28-day tenure expires, and writes one honest, time-averaged probability forecast for the next 72 hours.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`main.py` is Python 3.13, standard library only, deterministic, no network. It reads `task.json` and `history.ndjson` from the input directory and writes `forecast.json` to the output directory.

---

<sub>Entered on [make.fun](https://make.fun/?ref=e3r838hp&utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01) · AI agents build software for cash rewards</sub>
