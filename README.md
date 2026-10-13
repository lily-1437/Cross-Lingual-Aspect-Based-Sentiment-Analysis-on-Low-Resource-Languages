# Cross-Lingual Aspect-Based Sentiment Analysis on Low-Resource Languages

**XLM-RoBERTa vs. mT5 on the M-ABSA Dataset**

Aspect-Based Sentiment Analysis (ABSA) finds out how people feel about *specific parts* of something instead of judging the overall tone of a review. For example, a restaurant review may praise the food but criticize the service, and ABSA captures both.

This project compares two multilingual transformer models, **XLM-RoBERTa** (discriminative, token classification) and **mT5** (generative, sequence-to-sequence), on the multilingual, multi-domain **M-ABSA** dataset. It covers three research questions:

1. **Architecture:** Which model works better for ABSA triplet extraction, XLM-RoBERTa or mT5?
2. **Zero-shot transfer:** How well does a model trained only on English transfer to other (low-resource) languages?
3. **Domain effects:** Does training on more varied domains improve cross-lingual transfer, and does domain similarity predict cross-domain performance?

## ❗ Problem It Solves

- Most ABSA systems and benchmarks are built and tested **only in English**, so how they behave in other languages is poorly understood.
- Low-resource languages often have **no labeled ABSA data**. If English-trained models transfer well, ABSA becomes far easier to deploy worldwide.
- Zero-shot transfer is **uneven across languages**, and the reasons (script, training data size, language family, morphology) are not clearly separated in existing research.
- Existing work studies architecture comparison, cross-lingual transfer, or domain effects **separately**. This project examines all three together.

## 🎯 Tasks

| Task | Description |
|---|---|
| **TASD** (Target-Aspect-Sentiment Detection) | Harder: predicts aspect + category + sentiment triplets |
| **UABSA** (Unified ABSA) | Simpler: predicts aspect + sentiment pairs, without category |

## 🔑 Key Findings

- **mT5 beats XLM-RoBERTa** when both are fully trained (Restaurant, English):

| Task | XLM-RoBERTa F1 | mT5 F1 |
|---|---|---|
| TASD | 0.5159 | 0.6169 |
| UABSA | 0.5354 | 0.6809 |

- **Ensembling helps slightly.** Confidence-weighted voting of both models reaches F1 = **0.6645** on TASD.
- **Zero-shot transfer varies by language**, from **0.356 (Croatian)** down to **0.162 (Vietnamese)**. Indo-European languages do better on average, but language family alone doesn't explain the results (Arabic also scored highly).
- **More training domains don't always help.** A paired bootstrap test showed a significant effect in only 3 of 7 languages, and in 2 of those, more domain variety actually *reduced* performance.
- **Domain similarity matters.** A restaurant-trained model transferred meaningfully to Hotel reviews (F1 = 0.2866) but scored 0 on Laptop and Phone, even though Hotel shares no category names with Restaurant.

## 🛠️ Tech Stack & Methods

| Component | Details |
|---|---|
| Dataset | [M-ABSA](https://huggingface.co/datasets/Multilingual-NLP/M-ABSA) (21 languages, 7 domains, about 310K rows) |
| Models | `xlm-roberta-base` (token classification), `mT5-base` (text generation) |
| Labeling | Aspect-category-sentiment triples mapped to BIO tag sequences |
| Evaluation | Entity-level precision, recall, F1 using the `seqeval` library |
| Significance testing | Paired bootstrap (1,000 resamples) |
| Ensemble | Confidence-weighted voting, tuned by grid search on the dev set |
| Data analysis | Cramér's V association analysis between dataset variables |
| Language | Python |

## ⚙️ Experimental Setup

- **Training:** 15 epochs, single seed (42) due to computational limits
- **Architecture comparison:** Restaurant domain, English only
- **Zero-shot evaluation:** Train on English Restaurant, test on other languages with no extra training
- **Domain variety:** Restaurant-only vs. Restaurant + Hotel + Food training, tested on seven languages
- **Domain similarity:** Restaurant checkpoint tested on Hotel, Laptop, and Phone (English)

## 🚀 Getting Started

```bash
# Clone the repository
git clone YOUR_REPO_LINK_HERE
cd YOUR_REPO_NAME

# Install dependencies
pip install -r requirements.txt

# Train / evaluate (adjust to your actual scripts)
python train.py --model xlmr --task tasd
python train.py --model mt5 --task tasd
```

## ⚠️ Limitations

- Results use a **single seed**, so there are no variance estimates.
- The mT5 result is about 4.65 F1 points below the M-ABSA source paper baseline (0.6634), likely because of the single-run setup.
- The ensemble's margin over the published baseline is very small (0.0011), so treat it as indicative rather than a controlled comparison.
- Only a small set of languages was sampled.

## 🔮 Future Work

- Multi-seed training to estimate variance
- Broader language sampling with validated linguistic distance measures (URIEL, lang2vec)
- Embedding-based domain similarity measures to explain cross-domain asymmetries

## 👩‍💻 Authors

- Kaphian Hoque
- Arpita Nath
- Faranaz Safari

Department of Computer Science, Asian University for Women, Chattogram, Bangladesh

## 📄 Reference

Hoque, K., Nath, A., & Safari, F. *Cross-Lingual Aspect-Based Sentiment Analysis on Low-Resource Languages: XLM-RoBERTa vs. mT5 on M-ABSA.*

Dataset: Wu et al., "M-ABSA: A multilingual dataset for aspect-based sentiment analysis," EMNLP 2025.
