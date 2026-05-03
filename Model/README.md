# Model Modulu

Bu klasor, ALAN 3B bobrek ROI hacimleri uzerinde ikili anomali siniflandirmasi yapan model mimarilerini, egitim motorunu ve model secimi/degerlendirme yardimci komutlarini icerir. Komutlar proje kok dizininden `python -m Model.<modul>` biciminde calistirilir.

## Dosya Yapisi

```text
Model/
├── __init__.py
├── factory.py            # ModelConfig.architecture alanina gore model kurar
├── engine.py             # Egitim, CV, validasyon, test, kalibrasyon ve kayit akisi
├── train.py              # Tek egitim veya k-fold CV icin CLI girisi
├── search.py             # Optuna tabanli hiperparametre aramasi ve best_run yeniden egitimi
├── oof_predictions.py    # CV fold checkpoint'lerinden kalibre OOF tahminleri uretir
├── ensemble.py           # Birden fazla CV trial icin OOF kilitli ensemble degerlendirmesi
├── threshold_scan.py     # Kayitli validasyon/test olasiliklari uzerinde esik taramasi
├── resnet3d.py           # ResNet3D-18 ve ResNet3D-34 siniflandiricilari
├── unet3d.py             # Segmentasyon yerine logit ureten 3B U-Net siniflandirici
├── pointnet.py           # 3B maskeyi point cloud'a ceviren PointNet siniflandirici
└── README.md             # Model klasoru icin bu dokumantasyon
```

Model katmani su proje modullerine baglidir:

- `Preprocessing.dataset`: `ALAN/info.csv`, `ALAN/metadata.csv` ve `ALAN/alan/*.npy` kayitlarini okur; crop, pad, resize, NaN stratejisi ve tabular ozellikleri hazirlar.
- `Preprocessing.transforms`: yalnizca train split icin 3B flip, affine ve morfolojik augmentasyon uygular.
- `Utils.config`: `DataConfig`, `AugmentationConfig`, `ModelConfig`, `TrainConfig` ve `SearchConfig` dataclass'larini tutar.
- `Utils.metrics`: ROC-AUC, PR-AUC, F1, F-beta, MCC, balanced accuracy, bootstrap CI ve model skoru hesaplar.
- `Utils.calibration`: temperature scaling, isotonic calibration ve bootstrap esik secimi yardimcilarini saglar.
- `evaluate_final.py`: egitilmis `best_run` klasorunden son test raporlarini ve figurlerini uretir.

## Desteklenen Mimariler

Tum mimariler `factory.py` uzerinden ayni arayuzle kurulur. Girdi sekli `B x 1 x D x H x W`, cikti sekli `B` boyutlu logit vektorudur ve kayip fonksiyonu varsayilan olarak `BCEWithLogitsLoss` tabanlidir.

| `--architecture` | Dosya | Ana ayarlar |
|---|---|---|
| `resnet3d` | `resnet3d.py` | `--depth 18/34`, `--base-channels`, `--dropout`, `--norm-type batch/group` |
| `unet3d` | `unet3d.py` | `--unet-depth`, `--unet-base-channels`, `--unet-channel-multiplier`, `--unet-bottleneck-channels` |
| `pointnet` | `pointnet.py` | `--pointnet-num-points`, `--pointnet-point-features 3/4`, `--pointnet-mlp-channels`, `--pointnet-global-dim`, `--pointnet-use-input-transform` |

Tabular ek ozellikler varsayilan olarak aciktir. Bu ozellikler train split istatistikleriyle normalize edilir ve model basligina eklenir. Kapatmak icin `--disable-tabular-features` kullanilir.

## Tek Egitim

Ornek ResNet3D egitimi:

```bash
python -m Model.train \
  --architecture resnet3d \
  --epochs 30 \
  --batch-size 8 \
  --learning-rate 2e-4 \
  --target-shape 64 64 64 \
  --output-dir outputs/resnet3d_baseline
```

Ornek U-Net3D egitimi:

```bash
python -m Model.train \
  --architecture unet3d \
  --unet-depth 4 \
  --unet-base-channels 16 \
  --epochs 30 \
  --batch-size 8 \
  --output-dir outputs/unet3d_baseline
```

Ornek PointNet egitimi:

```bash
python -m Model.train \
  --architecture pointnet \
  --pointnet-num-points 2048 \
  --pointnet-point-features 4 \
  --epochs 30 \
  --batch-size 8 \
  --output-dir outputs/pointnet_baseline
```

`--cv-folds 2` veya daha buyuk verildiginde `train.py` tek train/dev/test akisi yerine k-fold cross-validation calistirir.

## Egitim Akisi

`engine.py` egitim sirasinda su adimlari yonetir:

1. Veri kayitlarini splitlere ayirir ve DataLoader'lari kurar.
2. Sadece train split icin augmentasyon uygular.
3. Sinif dengesizligi icin `--pos-weight-strategy` veya `--use-weighted-sampler` kullanir.
4. `adam`, `adamw` veya `sgd` optimizer'i ve istege bagli cosine scheduler/warmup ile egitir.
5. `--primary-metric` skoruna gore en iyi checkpoint'i `best_model.pt` olarak kaydeder.
6. Validasyon tahminleri uzerinden temperature ve/veya isotonic kalibrasyon uygular.
7. `youden`, `f1`, `fbeta` veya `fixed` yontemiyle karar esigini secer.
8. Test seti icin sabit ve ayarlanmis esik metriklerini, tahminleri ve bootstrap guven araliklarini kaydeder.

Baslica egitim secenekleri:

| Alan | CLI |
|---|---|
| Veri sekli | `--target-shape`, `--bbox-margin`, `--disable-bbox-crop`, `--disable-pad-to-cube` |
| NaN davranisi | `--nan-strategy`, `--nan-fill-value` |
| Cache | `--cache-mode none/memory/disk`, `--cache-dir` |
| Optimizasyon | `--optimizer`, `--learning-rate`, `--weight-decay`, `--scheduler`, `--warmup-epochs` |
| Dengesizlik | `--pos-weight-strategy`, `--use-weighted-sampler`, `--loss-type bce/focal`, `--focal-gamma` |
| Secim/metrik | `--primary-metric`, `--threshold-selection`, `--threshold-fbeta`, `--threshold-min-specificity`, `--threshold-min-precision` |
| Kalibrasyon | `--calibration-method temperature/isotonic/temperature+isotonic`, `--disable-calibration` |
| Donanim | `--device auto/cuda/mps/cpu`, `--disable-amp`, `--num-workers` |

## Uretilen Ciktilar

Tek egitim ciktilari `--output-dir` altina yazilir:

```text
outputs/<run>/
├── best_model.pt
├── checkpoint_meta.json
├── config.json
├── history.json
├── best_val_metrics.json
├── calibration.json
├── test_metrics.json
├── test_metrics_fixed_threshold.json
├── test_predictions.json
└── test_confidence_intervals.json
```

CV modunda her fold kendi `fold_XX/` klasorune `best_model.pt` ve `fold_result.json` yazar; kok klasorde `cv_summary.json` olusur.

## Optuna Aramasi

Hiperparametre aramasi icin:

```bash
python -m Model.search \
  --architecture resnet3d \
  --output-dir outputs/optuna_resnet3d \
  --study-name alan_resnet3d \
  --n-trials 20 \
  --n-folds 3 \
  --epochs 30 \
  --target-edge-choices 48 64 \
  --cache-mode disk
```

`search.py` her trial icin `trial_XXX/` klasoru olusturur, `leaderboard.json` dosyasini gunceller, en iyi trial parametreleriyle `best_run/` yeniden egitimi yapar ve kok dizine `study_summary.json` ile kolay erisim icin `best_model.pt` yazar. Arama daha once baslatildiysa ayni `--output-dir` ve `--study-name` ile SQLite tabanli `optuna_study.db` uzerinden devam eder.

Mimariye gore arama uzayi degisir:

- `resnet3d`: `depth`, `base_channels`, normalizasyon, dropout ve egitim hiperparametreleri.
- `unet3d`: encoder derinligi, taban kanal sayisi, kanal carpani, bottleneck secimi ve ortak egitim hiperparametreleri.
- `pointnet`: nokta sayisi, MLP genisligi, global boyut, siniflandirici basligi, input transform ve ortak egitim hiperparametreleri.

## OOF Tahminleri ve Ensemble

CV fold checkpoint'lerinden out-of-fold tahmin uretmek icin:

```bash
python -m Model.oof_predictions \
  --study-dir outputs/optuna_resnet3d \
  --trial-number 7 \
  --device auto
```

Bu komut varsayilan olarak `outputs/optuna_resnet3d/trial_007/oof_predictions.json` yazar. Ayrica kolay erisim icin study kokunde `oof_predictions.json` olusturabilir.

Birden fazla CV trial'i OOF uzerinden kilitli esikle ensemble etmek icin:

```bash
python -m Model.ensemble \
  --study-dirs outputs/optuna_resnet3d outputs/optuna_unet3d \
  --trial-numbers 7 4 \
  --output-dir outputs/ensemble_oof \
  --probability-mode arithmetic \
  --threshold-name f1_threshold
```

Ensemble ciktilari:

```text
outputs/ensemble_oof/
├── ensemble_predictions.json
├── oof_thresholds.json
├── final_test_metrics.json
├── test_confidence_intervals.json
└── interpretation.txt
```

`--threshold-name clinical_threshold` klinik odakli alternatif esik secimini kullanir. Ensemble esigi OOF tahminlerinden secilir; test seti esik veya model secimi icin yeniden kullanilmaz.

## Esik Taramasi

Kayitli `calibration.json` ve varsa `test_predictions.json` uzerinde hizli esik incelemesi:

```bash
python -m Model.threshold_scan --run-dir outputs/resnet3d_baseline
```

Varsayilan olarak kalibre olasiliklari kullanir. Ham olasiliklarla karsilastirma icin `--use-uncalibrated` eklenebilir.

## Son Test Raporu

Model klasoru disindaki `evaluate_final.py`, `best_model.pt` ve `config.json` bulunan bir run klasorunu okuyup son test metriklerini, siniflandirma raporlarini ve figurleri uretir:

```bash
python evaluate_final.py \
  --run-dir outputs/optuna_resnet3d/best_run \
  --use-saved-predictions
```

Varsayilan cikis yolu `results/final_evaluation/<architecture>_<run>/` bicimindedir. `--threshold` ile kayitli tuned esik yerine gecici bir karar esigi verilebilir.
