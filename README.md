# JusticeBridge

On-device, multilingual, privacy-first legal navigator for the Snapdragon
Multiverse hackathon. The project lives in **[`justicebridge/`](justicebridge/)** —
see [justicebridge/README.md](justicebridge/README.md) for the full architecture,
per-agent docs, backend options, and eval results.

```bash
pip install -r justicebridge/requirements.txt
python -m justicebridge.build_corpus
python -m justicebridge.build_index
python -m justicebridge.run_cli "the contractor hasn't paid my wages for two months"
streamlit run justicebridge/app.py
```

Turns a citizen's **spoken or scanned** legal problem into plain-language,
**statute-grounded** guidance + a colour-coded urgency signal, and always hands
off to a real **free** lawyer (DLSA). Legal *information*, never advice.
