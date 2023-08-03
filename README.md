# Generative Agents Llama

This repository is motivated by [generative-agents](https://github.com/mkturkcan/generative-agents) and the paper titled "Generative Agents: Interactive Simulacra of Human Behavior" [link to the paper](https://arxiv.org/pdf/2304.03442.pdf).

Recently, powerful Language Models like [llama-2-70B](https://ai.meta.com/llama/) have become publicly available, enabling simulations using the LLM. However, the simulations require an extremely large amount of time.

Motivated by this, the code has been revised. Specifically, the memory scoring and the location selection function have been changed.

## How to Use
Modify `model_name_or_path` in `agents/agent.py`, for example, uncomment and set `model_name_or_path = "Llama-2-70B-chat-GPTQ"`.
Then, run `python main.py`.

## Differences from [generative-agents](https://github.com/mkturkcan/generative-agents)
- Memory Scoring: Instead of generating the answer that contains the rating, the logits for the score (1~5) are evaluated, and the score with the largest logit is selected.
- Location Selection: Instead of scoring each location, the logits for the location are evaluated, and the location with the largest logits is selected.

Note that direct logits estimation is unavailable when using the OpenAI API (yet). However, logits are more efficient. Specifically, logits estimation is approximately 4~5 times faster than generation (A6000 48G, llama-70B). Logits always output the score or location, and in the case of location estimation, the required time does not increase linearly with the number of locations.
