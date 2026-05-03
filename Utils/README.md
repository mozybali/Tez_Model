# Utils Modulu

Bu klasor, ALAN 3B bobrek anomali siniflandirma projesinde ortak kullanilan konfig, metrik, kalibrasyon, gorsellestirme ve tekrarlanabilirlik yardimcilarini icerir. Model egitimi, Optuna aramasi, OOF tahminleri, ensemble degerlendirmesi, final test raporu ve smoke testleri bu modulleri ortak sozlesme olarak kullanir.

## Dosya Yapisi

```text
Utils/
+-- __init__.py          # Paket isaretleyici; disari API re-export etmez
+-- config.py            # Data, augmentasyon, model, egitim ve arama dataclass'lari
+-- metrics.py           # Ikili siniflandirma metrikleri, esik optimizasyonu ve bootstrap CI
+-- calibration.py       # Temperature/isotonic kalibrasyon ve bootstrap esik secimi
+-- plot_metrics.py      # history.json ve calibration.json uzerinden metrik grafikleri
+-- reproducibility.py   # Rastgelelik ve deterministik PyTorch ayarlari
`-- README.md            # Utils klasoru icin bu dokumantasyon
```

## Projedeki Yeri

Utils katmani su akislar tarafindan kullanilir:

| Kullanan | Kullandigi modul(ler) |
|---|---|
| `Preprocessing.dataset` | `Utils.config.DataConfig`, `Utils.config.AugmentationConfig` |
| `Model.factory` | `Utils.config.ModelConfig` |
| `Model.train` | `DataConfig`, `AugmentationConfig`, `ModelConfig`, `TrainConfig` |
| `Model.engine` | Config dataclass'lari, metrikler, kalibrasyon ve `seed_everything` |
| `Model.search` | `SearchConfig`, diger config dataclass'lari ve `to_serializable` |
| `Model.oof_predictions` | Kalibrasyon, metrikler ve config serilestirme |
| `Model.ensemble` | Kalibrasyon yardimcilari ve final metrikleri |
| `evaluate_final.py` | Config, kalibrasyon ve siniflandirma raporu metrikleri |
| `Tests/test_smoke.py` | Metrik sozlesmesi, NaN ve mini egitim smoke kontrolleri |

## `config.py`

Projedeki ana ayar gruplari dataclass olarak burada tutulur.

| Dataclass | Gorev |
|---|---|
| `DataConfig` | Veri yollarini, split adlarini, hedef hacim boyutunu, bbox crop, cube padding, sag ROI kanoniklestirme, NaN stratejisi ve cache ayarlarini tutar. |
| `AugmentationConfig` | Train-time flip, affine ve morfolojik augmentasyon olasiliklarini ve araliklarini tanimlar. |
| `ModelConfig` | `resnet3d`, `unet3d` ve `pointnet` mimarileri icin ortak ve mimariye ozel parametreleri tutar. |
| `TrainConfig` | Egitim klasoru, epoch, batch size, optimizer, scheduler, early stopping, sinif dengesizligi, kalibrasyon, esik secimi, CV ve TTA ayarlarini tutar. |
| `SearchConfig` | Optuna study adi, cikti dizini, trial sayisi, timeout ve sampler seed alanlarini tutar. |

`DataConfig.__post_init__`, `nan_strategy` ve `cache_mode` degerlerini dogrular. Desteklenen NaN stratejileri:

```text
none, drop_record, fill_zero, fill_mean, fill_median, fill_constant
```

Desteklenen cache modlari:

```text
none, memory, disk
```

`DataConfig.resolved()` yol ve sayisal alanlari normalize eder; `target_shape` tek integer verilirse kupe genisletilir. `to_serializable(...)`, dataclass, `Path`, dict, tuple ve list degerlerini JSON'a yazilabilir hale getirir.

## `metrics.py`

Ikili siniflandirma metrikleri proje genelinde tek merkezden uretilir.

| Fonksiyon | Gorev |
|---|---|
| `compute_binary_classification_metrics(...)` | Threshold uygulayip accuracy, balanced accuracy, F1, precision, recall, ROC-AUC, PR-AUC, confusion matrix, specificity, NPV, FPR/FNR, MCC, Cohen kappa, macro ve weighted ortalamalari hesaplar. |
| `compute_per_class_report(...)` | Negatif/pozitif siniflar icin precision, recall, F1 ve support raporu dondurur. |
| `optimize_threshold(...)` | Validasyon olasiliklari uzerinde `youden`, `f1` veya `fbeta` yontemiyle karar esigi secer. |
| `bootstrap_confidence_intervals(...)` | ROC-AUC, PR-AUC, F1, balanced accuracy, precision ve recall icin bootstrap guven araliklari hesaplar. |
| `select_model_score(...)` | `primary_metric` degerini okur; yoksa `pr_auc`, `balanced_accuracy`, `f1`, son olarak negatif loss fallback'ini kullanir. |

`compute_binary_classification_metrics`, tek sinifli `y_true` durumunda ROC-AUC ve PR-AUC icin crash etmek yerine `nan` dondurur. Confusion-matrix turevi oranlarda sifira bolme durumlari kontrollu varsayilanlara duser.

Esik optimizasyonunda `min_specificity` ve `min_precision` guard'lari, F1/F-beta seciminin cok dusuk esige kayip asiri pozitif tahmin yapmasini sinirlamak icin kullanilir.

## `calibration.py`

Model logitleri ve olasiliklari icin post-hoc kalibrasyon yardimcilari burada yer alir.

| API | Gorev |
|---|---|
| `TemperatureResult` | Temperature scaling sonucunu; sicaklik, NLL ve ECE oncesi/sonrasi degerleriyle tutar. |
| `ThresholdBootstrapResult` | Bootstrap esik seciminin medyanini, yuzdelik araligini ve gecerli ornek sayisini tutar. |
| `IsotonicResult` | Isotonic kalibrasyon esiklerini ve ECE oncesi/sonrasi degerlerini tutar. |
| `logits_from_probs(...)` | Olasiliklari clipping ile logit uzayina geri cevirir. |
| `fit_temperature(...)` | Validasyon logitleri uzerinde tek skaler sicaklik parametresi fit eder. |
| `apply_temperature(...)` | Yeni logitlere fit edilen sicakligi uygulayip olasilik dondurur. |
| `reliability_bins(...)` | Reliability diagram icin esit genislikli bin istatistikleri uretir. |
| `expected_calibration_error(...)` | ECE hesaplar. |
| `select_threshold_bootstrap(...)` | Bootstrap resample'lari uzerinden daha stabil karar esigi secer. |
| `fit_isotonic(...)` | Sklearn `IsotonicRegression` ile monoton olasilik kalibrasyonu fit eder. |
| `apply_isotonic(...)` | Fit edilmis isotonic mapping'i yeni olasiliklara uygular. |

Temperature scaling varsayilan olarak sicakligi `[0.25, 10.0]` araliginda tutar. Isotonic fit, bos veri veya tek sinifli label durumunda mevcut olasiliklari koruyan guvenli bir sonuc dondurur.

## `plot_metrics.py`

Egitim gecmisinden tek bir ozet figur uretir. Komut proje kokunden calistirilir:

```bash
python -m Utils.plot_metrics outputs/resnet3d_baseline/history.json
```

Opsiyonel cikti dizini ve ekranda gosterme:

```bash
python -m Utils.plot_metrics outputs/resnet3d_baseline/history.json \
  --out-dir outputs/resnet3d_baseline \
  --show
```

Varsayilan cikti:

```text
outputs/<run>/metrics_plot.png
```

`history.json` yaninda `calibration.json` varsa PR curve ve reliability diagram panelleri de doldurulur. Uretilen figur su panelleri icerir:

| Panel | Icerik |
|---|---|
| Loss | Train ve validation loss egirileri |
| ROC-AUC & PR-AUC | Train/val ayriminda AUC metrikleri |
| Accuracy & Balanced Accuracy | Accuracy ve balanced accuracy egirileri |
| Precision, Recall & F1 | Temel karar metrikleri |
| Confusion Matrix | Son epoch icin train ve val confusion matrix |
| PR Curve | Validation uzerinde calibrated/uncalibrated PR karsilastirmasi |
| Reliability Diagram | Kalibrasyon oncesi/sonrasi guvenilirlik |
| Ozet Tablo | Son epoch metriklerinin tablo ozeti |

`Model.engine`, egitim sonunda `generate_plots(...)` fonksiyonunu kullanarak mevcut run klasorune metrik figurunu yazabilir.

## `reproducibility.py`

`seed_everything(seed)` fonksiyonu Python `random`, NumPy, PyTorch CPU/CUDA seed'lerini ve `PYTHONHASHSEED` ortam degiskenini ayarlar. Ayrica `torch.backends.cudnn.deterministic=True` ve `torch.backends.cudnn.benchmark=False` kullanarak egitim tekrarlarini daha deterministik hale getirir.

Bu fonksiyon `Model.engine` icinde egitim basinda cagrilir.

## Kullanim Notlari

- Utils modulleri veri dosyalarini dogrudan yuklemez; veri okuma ve on isleme `Preprocessing/` altindadir.
- `__init__.py` bos tutulmustur; cagiran kodlar fonksiyon ve dataclass'lari dogrudan ilgili modulden import eder.
- Config dataclass'lari CLI argumanlariyla `Model.train` ve `Model.search` tarafinda doldurulur, sonra checkpoint ve JSON ciktilarina `to_serializable` ile yazilir.
- Metrik anahtarlarinin geriye uyumlulugu onemlidir; `evaluate_final.py`, `Model.engine`, `Model.ensemble` ve testler ayni JSON sozlesmesini bekler.
- Kalibrasyon ve esik secimi validasyon/OOF tahminleri uzerinden yapilir; test seti model veya esik secimi icin kullanilmaz.
