# Grammar Scoring Engine for Spoken Audio

SHL Hiring Assessment 2026. The model takes a 45-60 s speech recording and predicts a continuous grammar score (MOS Likert scale, range 0-5).

## Results

| Metric | Value |
|---|---|
| **Training RMSE** (model refit on all 769 training clips) | **0.4360** |
| 5-fold cross-validated (out-of-fold) RMSE | 0.6394 |
| Baseline RMSE (always predict the training mean) | 1.2382 |
| Pearson correlation (out-of-fold) | 0.861 |
| Kaggle leaderboard score | _<add your score here>_ |

The cross-validated RMSE is the more honest estimate of performance on unseen audio. The training RMSE is lower because the models are evaluated on data they were fitted to.

Per-model results:

| Model | Training RMSE | CV RMSE | Blend weight |
|---|---|---|---|
| Gradient boosting on hand-crafted features | 0.6724 | 1.0516 | 0.079 |
| Ridge on audio + text embeddings | 0.4540 | 0.6430 | 0.723 |
| SVR on all features combined | 0.4138 | 0.6697 | 0.198 |

## Approach

```
wav -> 16 kHz mono, trim silence, peak-normalise
    -> Whisper (small) ASR with word timestamps
    -> features: text grammar + fluency/timing + embeddings
    -> 3 regressors (5-fold CV) -> NNLS blend -> clip to [0, 5] -> submission.csv
```

1. **Preprocessing:** resample to 16 kHz mono, trim leading/trailing silence, peak-normalise, cap at 60 s.
2. **Transcription:** `openai/whisper-small` with word-level timestamps.
3. **Text grammar features:** LanguageTool error rates (overall and by category), `distilgpt2` perplexity, spaCy sentence length, POS diversity, subordinate-clause rate and verb-tense variety, type-token ratio.
4. **Fluency features:** speech rate, articulation rate, pause count, mean and maximum pause length, filler-word rate, word-repetition rate.
5. **Embeddings:** wav2vec2-base (mean + std pooled hidden states) for the audio, and `all-MiniLM-L6-v2` for the transcript.
6. **Models:** HistGradientBoosting on the hand-crafted features, RidgeCV on the embeddings, and an SVR on everything. Out-of-fold predictions are blended with non-negative least squares, and the final output is clipped to 0-5.

## Findings

- The embedding model carries most of the blend (about 72% weight). The hand-crafted grammar features alone are weak (CV RMSE 1.05).
- A likely reason is that Whisper tends to clean up grammatical mistakes in its transcripts, which weakens the error-count features. This is a limitation of the approach.
- The most informative hand-crafted features by permutation importance were type-token ratio, GPT-2 perplexity and maximum pause length.
- Residuals are centred on zero. Error is largest for the few clips with a true score of 1.
- Some training clips have a score of 0 and are predicted well.

## Repository contents

| File | Description |
|---|---|
| `grammar_scoring_engine.ipynb` | Full pipeline with code, report, plots and outputs, including the training RMSE |
| `submission.csv` | Predictions for the 216 test clips (`filename,label`) |
| `README.md` | This file |

## How to run

1. Open the notebook on Kaggle (or Colab) with a **GPU** and **Internet** enabled, and attach the competition data.
2. Install the extra packages:
   ```
   pip install language_tool_python
   python -m spacy download en_core_web_sm
   ```
   Other packages used: `torch`, `transformers`, `sentence-transformers`, `librosa`, `scikit-learn`, `scipy`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `tqdm`.
3. Set `DATA_DIR`, `TRAIN_DIR` and `TEST_DIR` in the setup cell to the dataset location (on Kaggle: `/kaggle/input/competitions/shl-hiring-assessment-2026/Dataset_Final`).
4. Run all cells. Transcription takes around 3 hours on a T4 GPU. Intermediate results are cached in `./cache`, so reruns are fast.

## Notes and limitations

- `sample_submission.csv` has 204 rows and mostly different file names from the 216 clips in `test.csv`. Predictions were produced for the 216 clips in `test.csv`, which match the files in the test folder exactly.
- If LanguageTool or spaCy is unavailable, the notebook sets those features to 0 and still runs, with lower accuracy.
- Possible improvements: a larger Whisper model for more literal transcripts, fine-tuning a transformer on the transcripts, ordinal-aware losses, and multi-seed ensembling.
