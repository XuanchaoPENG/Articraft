# Model providers

Articraft supports OpenAI, Anthropic, Gemini, OpenRouter, and Codex CLI. OpenAI is the
default.

## Configure a provider

Set the API key for the provider that you want to use:

| Provider | API key | Default model | Reference images |
| --- | --- | --- | --- |
| OpenAI | `OPENAI_API_KEY` | `gpt-5.6` | Yes |
| Anthropic | `ANTHROPIC_API_KEY` | `claude-sonnet-5` | Yes |
| Gemini | `GEMINI_API_KEY` | `gemini-3.6-flash` | Yes |
| OpenRouter | `OPENROUTER_API_KEY` | `nvidia/nemotron-3-ultra-550b-a55b:free` | No |
| Codex CLI | Codex login | Codex configured default | Yes |

You can put the key in `.env` or set it for one command. Do not commit API keys.

Select a provider with `--provider`:

```shell
ANTHROPIC_API_KEY=your_key_here uv run articraft \
  --provider anthropic "a folding chair"
```

## Select a model

Use `--model` to replace the default model:

```shell
GEMINI_API_KEY=your_key_here uv run articraft \
  --provider gemini --model gemini-3.6-flash "a folding chair"
```

Articraft passes an unknown model name to the selected provider. The provider returns an
error if it does not accept the name.

The live interface cannot estimate cost or context use for an unknown model. The run can
still continue if the provider accepts the model.

## Use OpenRouter

OpenRouter accepts text prompts only. Do not pass `--image` with this provider.

Set `OPENROUTER_HTTP_REFERER` and `OPENROUTER_APP_TITLE` if you want OpenRouter attribution.
These values are optional.

OpenRouter reports token use and request cost when its API returns them. Articraft does not
estimate the context percentage for arbitrary OpenRouter models.

## Use the Python API

Pass the same provider and model names to `generate()` or `generate_async()`:

```python
result = articraft.generate(
    "a folding chair",
    provider="anthropic",
    model="claude-sonnet-5",
)
```

The function checks the required API key before it starts the run.

## Use Codex CLI

Install and log in to Codex CLI, then select its provider:

```shell
uv run articraft generate --provider codex-cli "a folding chair"
```

Articraft remains in control of the agent loop, tools, compile feedback, and run record.
Codex CLI returns one structured assistant turn at a time and does not edit the workspace
directly. Use `--model` to override the model configured by Codex. Total tokens are recorded
when the CLI reports them; Codex CLI does not expose dollar cost accounting.
