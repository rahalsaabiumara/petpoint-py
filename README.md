```bash
├── Dockerfile
├── README.md
├── app
│   ├── __init__.py
│   ├── data
│   │   ├── data2.json
│   │   ├── df_utterances.pkl
│   │   ├── stopword_list_tala.txt
│   │   ├── tfidf_matrix.pickle
│   │   └── vectorizer.pickle
│   ├── encoders
│   │   ├── label_encoder.pickle
│   │   ├── ner_label_encoder.pickle
│   │   ├── tokenizer.pickle
│   │   └── transition_params.pickle
│   ├── models
│   │   ├── model_intent.keras
│   │   └── model_ner_crf.keras
│   ├── routes.py
│   └── utils.py
├── config.py
├── docker-compose.yml
├── requirements.txt
└── run.py
```
