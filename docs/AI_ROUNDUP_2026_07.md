# AI Roundup — July 2026

Quick summaries of each item, with source URLs.

## World models / interactive video generation

**ABot-World**
Real-time interactive world simulator running on a single desktop GPU. Takes user actions as input, generates continuous "infinite" world environments at 720p/16fps via action-conditioned rollout.
https://amap-cvlab.github.io/ABot-World/

**PixWorld**
Single pixel-space diffusion model unifying 3D scene generation and reconstruction. Takes text/image/multi-view input, decodes directly to explorable 3D Gaussian scenes; distilled 4-step version generates a scene in ~0.6s (~1000x faster than typical diffusion world generators).
https://sensengao.github.io/PixWorld/

**Mira**
5B-parameter multiplayer world model that simulates Rocket League 2v2 matches in real time (20fps, single GPU) purely from player key-presses and past frames — no physics or rendering engine involved. Built by General Intuition + Kyutai Labs with Epic Games.
https://mira-wm.com/

**LingBot-World 2.0**
Open-source world model from Robbyant (Ant Group). Generates hour-long, real-time, 720p/60fps interactive worlds with no quality decay, plus a "native agent" mechanism so worlds keep evolving on their own, not just react to input.
https://technology.robbyant.com/lingbot-world-v2

**Alaya World**
Interactive autoregressive world model generating extended video (60+ seconds, 720p/24fps) with real-time control — users set an initial scene and steer it via camera moves and mid-scene text prompts while it holds spatial/temporal consistency.
https://alaya-lab.github.io/AlayaWorld/

## Pose / motion / robotics

**ProxyPose**
Not a body-mesh recovery method — it's a general 6-DoF pose tracker. From monocular video + one user-clicked pixel, it renders a "proxy video" of a cube undergoing the same motion as that point, then extracts pose geometrically. Works on rigid objects, cameras, and reportedly non-rigid surfaces (e.g. faces) without retraining. Competes with point trackers like CoTracker, not with SMPL-based HMR.
https://ruihangzhang97.github.io/proxypose/

**Humanoid Surgery**
Research evaluating whether humanoid robots can perform minimally-invasive surgery. Built a teleoperation system letting a humanoid execute laparoscopic tasks under a remote surgeon's control; validated via benchtop tests, user studies, and live porcine cholecystectomy.
https://humanoid-surgeon.github.io/

## Image generation

**SeFi-Image**
Efficient text-to-image foundation model (1B/2B/5B variants) using "semantic-first diffusion" — denoises semantic structure slightly ahead of texture. Trained cheaply (~125K A800 GPU-hrs for the 5B model), competitive with Qwen-Image/Z-Image. Open weights, non-commercial license.
https://jmliu206.github.io/sefi-web/

**Seedream 5.0 Pro**
ByteDance's multimodal image model: strong image-text alignment, structural coherence, and text rendering; adds spatial "grounding" for pixel-level regional editing, intelligent layer separation into editable design assets, and 10+ language support (incl. RTL).
https://seed.bytedance.com/en/seedream5_0_pro

**Muse Image / Muse Video**
Meta Superintelligence Labs' first image + video models. Muse Image acts as an agent (invokes search/coding tools, self-refines, scales test-time compute) rather than a direct prompt→image mapping; includes an invisible "Content Seal" watermark. Muse Video (same pretraining base) adds native audio and ranks #3 in human-preference Elo for text-to-video.
https://ai.meta.com/blog/introducing-muse-image-muse-video-msl/

**Reve 2.1**
Updated version of Reve's 4K image model — better layout planning, improved multilingual text rendering, and precise, editable image elements you can iteratively re-render.
https://blog.reve.com/posts/launching-reve-2.1/

## LLMs / agents

**GPT-Live**
OpenAI's new full-duplex voice model replacing default ChatGPT Voice — can listen and speak simultaneously, gives conversational back-channel cues ("mhmm"), and delegates harder queries to GPT-5.5 in the background. No video/screen-share support at launch.
https://openai.com/index/introducing-gpt-live/

**Grok 4.5**
xAI's frontier coding/agentic model, built on the 1.5T-parameter V9 base and co-trained with Cursor. Elon Musk calls it "Opus-class"; runs at ~80 tokens/sec, priced $2/$6 per million input/output tokens.
https://x.ai/news/grok-4-5

**Muse Spark 1.1**
Meta's multimodal reasoning model for agentic tasks (tool/computer use, coding), 1M-token context. First paid model on the new Meta Model API ($1.25/$4.25 per million input/output tokens).
https://ai.meta.com/blog/introducing-muse-spark-meta-model-api/

**Hy3**
Tencent's open 295B-parameter MoE LLM (21B active params, 256K context) for reasoning, agentic workflows, and long-context tasks — claimed to rival flagship models several times its active size. Open weights on Hugging Face. (Distinct from Tencent's separate "HY 3D" text/image-to-3D generation tool.)
https://hy.tencent.com/research/hy3

**Wan-Streamer v0.2**
Alibaba's real-time, low-latency (~200ms) native-streaming audio-visual conversation model. v0.2 raises resolution to 640x368 at the same latency, keeping posture/gaze/hands/scene legible during live interaction.
https://wan-streamer.com/v0.2/

## Other

**"GPT 5.6 is a BEAST"**
A third-party YouTube review video, not a research paper or product page — covers OpenAI's GPT-5.6 model (released alongside GPT-Live in the same July 2026 announcement wave).
