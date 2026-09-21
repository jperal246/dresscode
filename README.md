# dresscode
Computer vision closet assistant. Segments and tags garments from ordinary photos, learns which pieces work together, and recommends outfits based on your style, your calendar, and the weather.

Computer vision semester project, Fall 2026.

---

## Pitch

> How many of you have put on an outfit, realized the top doesn't work with the pants, swapped the top, then swapped the pants, then started over, and somehow ended up further from a good outfit than when you started?
>
> You're not alone. That's exactly why I built this.
>
> It's a personal AI stylist that puts outfits together for you, based on your actual closet, your style, your schedule, and the weather that day. And because it can see everything you own, it pulls out the pieces you haven't touched in months or never quite figured out how to style.


---

## What this is

A three-stage vision pipeline that turns photographs of a real wardrobe into ranked outfit recommendations:

1. **Garment understanding.** Instance segmentation plus fine-grained attribute prediction on unconstrained photos of clothing. Trained on Fashionpedia.
2. **Outfit compatibility.** Type-aware embeddings that score whether a set of garments works together. Trained and evaluated on Polyvore Outfits.
3. **Context re-ranking.** Candidate outfits re-ranked by a personal style embedding (CLIP over inspiration images), then filtered by weather-derived warmth requirements and calendar-derived formality.

An optional gesture module lets you browse suggestions hands-free using a temporal classifier trained on MediaPipe hand landmarks.

## Datasets

| Dataset | Link | Used for |
|---|---|---|
| Fashionpedia | https://fashionpedia.github.io/home/ | Segmentation + attributes |
| Polyvore Outfits | https://github.com/mvasil/fashion-compatibility | Outfit compatibility |
| Personal closet set | `data/closet/` (in this repo) | Domain-transfer evaluation only |

---

## Repository structure

```
wardrobe-vision/
│
├── README.md
├── LICENSE
├── requirements.txt
├── environment.yml
├── .gitignore                      # excludes data/raw, checkpoints, .env
├── Makefile                        # make data | make train-seg | make eval
│
├── docs/
│   ├── proposal.pdf                # the graded proposal
│   ├── decisions.md                # running log: what you tried, what broke, why
│   └── figures/                    # plots and qualitative examples for the writeup
│
├── configs/                        # every experiment is a config file, never a CLI flag
│   ├── seg_maskrcnn_r50.yaml
│   ├── seg_yolov8m.yaml
│   ├── attr_multilabel.yaml
│   ├── compat_typeaware.yaml
│   ├── compat_baseline_shared.yaml
│   └── gesture_gru.yaml
│
├── data/
│   ├── raw/                        # gitignored, populated by scripts
│   │   ├── fashionpedia/
│   │   └── polyvore/
│   ├── processed/                  # gitignored, COCO-format conversions
│   ├── closet/                     # COMMITTED: your ~80 labeled photos + annotations
│   │   ├── images/
│   │   └── annotations.json
│   ├── inspiration/                # style-reference images (gitignored if personal)
│   └── gestures/                   # COMMITTED: recorded landmark sequences (.npy)
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── download.py             # fetch + verify checksums for both datasets
│   │   ├── fashionpedia.py         # dataset class, COCO conversion
│   │   ├── polyvore.py             # outfit/FITB/compatibility loaders
│   │   ├── closet.py               # your own photos
│   │   └── augment.py              # background compositing, lighting, deformation
│   │
│   ├── models/
│   │   ├── segmentation.py         # Mask R-CNN / YOLO-seg wrapper
│   │   ├── attributes.py           # multi-label attribute head
│   │   ├── compatibility.py        # type-aware embedding network
│   │   ├── style.py                # CLIP encoding + centroid clustering
│   │   └── gesture.py              # GRU / 1D-CNN over landmark sequences
│   │
│   ├── features/
│   │   ├── color.py                # k-means in CIELAB over the mask
│   │   └── warmth.py               # attributes -> warmth score mapping
│   │
│   ├── training/
│   │   ├── train_segmentation.py
│   │   ├── train_attributes.py
│   │   ├── train_compatibility.py  # triplet loss + hard negative mining
│   │   └── train_gesture.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py              # AUC, FITB, mAP, mIoU, macro-F1
│   │   ├── baselines.py            # CLIP zero-shot, COCO-only, HOG+color, per-frame
│   │   ├── domain_gap.py           # benchmark vs closet comparison
│   │   └── report.py               # writes results/metrics.json
│   │
│   ├── pipeline/
│   │   ├── digitize.py             # photo -> structured garment record
│   │   ├── generate.py             # wardrobe -> candidate outfits
│   │   └── rerank.py               # style + weather + calendar re-ranking
│   │
│   └── integrations/               # keep thin, keep last
│       ├── weather.py              # Open-Meteo, no API key needed
│       ├── calendar_sync.py        # Google Calendar API / .ics parsing
│       └── notion_log.py           # wear history logging
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_segmentation_results.ipynb
│   ├── 03_compatibility_results.ipynb
│   ├── 04_domain_gap_analysis.ipynb
│   └── 05_failure_cases.ipynb      # the one graders remember
│
├── app/
│   ├── main.py                     # FastAPI or Streamlit demo
│   ├── mirror.py                   # gesture-driven browsing mode
│   └── static/
│
├── results/
│   ├── metrics.json                # every number in the report, reproducibly
│   └── figures/
│
├── checkpoints/                    # gitignored, link weights in a release instead
│
└── tests/
    ├── test_data_loaders.py
    ├── test_metrics.py
    └── test_color.py
```

## Setup

```bash
git clone https://github.com/jperal246/dresscode.git
cd dresscode
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.data.download --dataset fashionpedia
python -m src.data.download --dataset polyvore
```

## Results

Populated as the semester progresses. All numbers generated by `python -m src.evaluation.report`.

| Metric | Baseline | Ours | Target |
|---|---|---|---|
| Compatibility AUC (nondisjoint) | | | 0.85 |
| Compatibility AUC (disjoint) | | | report |
| Fill-in-the-blank accuracy | | | 52% |
| Segmentation mAP@0.5 | | | 0.45 |
| Attribute macro-F1 | | | report |
| Domain gap (benchmark to closet) | | | minimize |
| Gesture top-1 accuracy | | | 90% |

## References

- Jia et al. *Fashionpedia: Ontology, Segmentation, and an Attribute Localization Dataset.* ECCV 2020.
- Vasileva et al. *Learning Type-Aware Embeddings for Fashion Compatibility.* ECCV 2018.
