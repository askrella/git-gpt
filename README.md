# Git-GPT

Ask questions against any git repository.

Local. Private. No API key required.

# Requirements

- Python 3.10+

# Supported file types

Check out `LOADER_MAPPING` in `ingest.py`.

# Installation

Download the gpt4all model and put it into the `models/` folder. Adjust the path in the .env file.

- https://gpt4all.io/models/ggml-gpt4all-j-v1.3-groovy.bin (~3,7 GB)
- https://gpt4all.io/models/ggml-gpt4all-l13b-snoozy.bin (~7,6 GB) (Doesnt work yet)

```bash
# Installation
pip install -r requirements.txt

# Run Git-GPT
python app.py
```

# GPT
You can also use OpenAI's GPT-3 API. Just set the `OPENAI_API_KEY` environment variable in the `.env` file.

The default model is `gpt4all`. If the api key is set, it will use OpenAI GPT-3.

# Credits

Built with [LangChain](https://github.com/hwchase17/langchain), [GPT4All](https://github.com/nomic-ai/gpt4all), [LlamaCpp](https://github.com/ggerganov/llama.cpp), [Chroma](https://www.trychroma.com/) and [SentenceTransformers](https://www.sbert.net/).
