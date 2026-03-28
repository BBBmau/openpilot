#!/usr/bin/env python3
"""
Pipecat worker: LiveKit room (WebRTC) ↔ Gemini Live API (WebSocket).

Subscribes to the comma body's microphone + camera from the LiveKit room and streams
multimodal input to Gemini; publishes assistant audio back into the room for the body
to play through soundd (see ``gemini_livekit_body_client.py``).

Environment (required):

- ``LIVEKIT_URL`` — e.g. ``wss://your-project.livekit.cloud``
- ``LIVEKIT_API_KEY``, ``LIVEKIT_API_SECRET``
- ``LIVEKIT_ROOM_NAME`` — same room the body client joins
- ``GOOGLE_API_KEY`` — Gemini API key

Optional CLI overrides: ``-u/--url``, ``-r/--room``

Run from the openpilot repo (or any env with ``pipecat-ai[livekit,google]``)::

  export LIVEKIT_URL=... LIVEKIT_API_KEY=... LIVEKIT_API_SECRET=... LIVEKIT_ROOM_NAME=body-demo GOOGLE_API_KEY=...
  PYTHONPATH=. uv run --extra gemini_live python tools/body/gemini_live_worker/bot.py
"""
from __future__ import annotations

import asyncio
import os
import sys

try:
  from dotenv import load_dotenv
except ImportError:
  load_dotenv = None

from loguru import logger

from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.runner.livekit import configure
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService
from pipecat.transports.livekit.transport import LiveKitParams, LiveKitTransport

if load_dotenv is not None:
  load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="INFO")


async def main() -> None:
  url, token, room_name = await configure()

  if not os.getenv("GOOGLE_API_KEY"):
    raise SystemExit("GOOGLE_API_KEY is required for Gemini Live.")

  transport = LiveKitTransport(
    url=url,
    token=token,
    room_name=room_name,
    params=LiveKitParams(
      audio_in_enabled=True,
      audio_out_enabled=True,
      video_in_enabled=True,
    ),
  )

  llm = GeminiLiveLLMService(
    api_key=os.getenv("GOOGLE_API_KEY"),
    settings=GeminiLiveLLMService.Settings(
      voice=os.getenv("GEMINI_LIVE_VOICE", "Aoede"),
      system_instruction=os.getenv(
        "GEMINI_LIVE_INSTRUCTIONS",
        "You are a concise voice assistant for someone using a comma body device in a car. "
        + "You may receive camera video from the vehicle. Reply in plain spoken language; no markdown or emojis.",
      ),
    ),
  )

  context = LLMContext()
  user_aggregator, assistant_aggregator = LLMContextAggregatorPair(context)

  pipeline = Pipeline(
    [
      transport.input(),
      user_aggregator,
      llm,
      transport.output(),
      assistant_aggregator,
    ],
  )

  task = PipelineTask(
    pipeline,
    params=PipelineParams(
      enable_metrics=True,
      enable_usage_metrics=True,
    ),
  )

  @transport.event_handler("on_first_participant_joined")
  async def on_first_participant_joined(_transport, _participant_id):
    await asyncio.sleep(0.5)
    await task.queue_frames([LLMRunFrame()])

  runner = PipelineRunner()
  await runner.run(task)


if __name__ == "__main__":
  asyncio.run(main())
