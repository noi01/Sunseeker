# Sunseeker DQN Trainer

This project trains a DQN agent on the custom Gym environment `sunday5-v1`.

## 1. Setup

1. Create a virtual environment:

```bash
python -m venv sunseeker
```

2. Activate it:

```bash
source sunseeker/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## 2. Validate the Environment

Run a quick Gym compatibility check before training:

```bash
python dqn.py --check_env
```

## 3. Run Training

Default run:

```bash
python dqn.py
```

Custom run:

```bash
python dqn.py --episode 200 --max_steps 100 --batch_size 32 --save_interval 5
```

Resume from a saved model:

```bash
python dqn.py --load models/<experiment>/agent_<timestamp>_final.keras
```

## 4. Optional WebSocket Connection

You can connect the training process to a WebSocket server.

```bash
python dqn.py --ws_url ws://localhost:8765 --ws_timeout 5
```

- `--ws_url`: server URL to connect to.
- `--ws_timeout`: connection timeout in seconds.

The WebSocket runs in parallel with the training loop after connection is established.

Incoming messages are handled by a callback. By default, messages are printed. You can customize processing by passing your own callback to `setup_websocket_client(..., on_message_callback=...)` in `dqn.py`.

## 5. Outputs

- Experiment artifacts are saved under `models/<experiment_prefix>_<YYMMDD_HHMM>/`.
- Model checkpoints and final model are saved as `.keras` files.
- Episode stats are written to `_ouptut.csv` inside the experiment folder.
