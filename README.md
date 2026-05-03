# ALAN 3B Böbrek Anomali Sınıflandırması

Bu proje, ALAN veri setindeki böbrek ROI hacimlerinde anomali varlığını ikili sınıflandırma problemi olarak ele alır. Girdi olarak `.npy` formatındaki 3B böbrek maskeleri kullanılır; eğitim hattı deterministik ön işleme, train-time augmentasyon, tabular ek özellikler, sınıf dengesizliği yönetimi, post-hoc kalibrasyon, eşik seçimi, Optuna araması, k-fold CV, OOF tahminleri ve OOF-kilitli ensemble değerlendirmesini destekler.

Komutlar proje kök dizininden çalıştırılmalıdır.

## Proje Yapısı

```text
TezModel/
├── ALAN/
│   ├── info.csv              # ROI_id, subset ve ROI_anomaly alanları
│   ├── metadata.csv          # ROI istatistikleri ve bbox bilgisi
│   ├── summary.json          # Veri seti özeti
│   └── alan/                 # 3B .npy hacim dosyaları
├── Preprocessing/
│   ├── analyze_dataset.py    # metadata.csv ve summary.json üretimi
│   ├── dataset.py            # Kayıt okuma, split ayırma, ön işleme ve cache
│   ├── transforms.py         # 3B flip, affine ve morfolojik augmentasyonlar
│   └── README.md             # Ön işleme ayrıntıları
├── Model/
│   ├── resnet3d.py           # ResNet3D-18 / ResNet3D-34 sınıflandırıcıları
│   ├── unet3d.py             # Logit üreten 3B U-Net sınıflandırıcı
│   ├── pointnet.py           # 3B maskeden point cloud türeten PointNet
│   ├── factory.py            # architecture alanına göre model kuran fabrika
│   ├── engine.py             # Eğitim, CV, test, kalibrasyon ve kayıt akışı
│   ├── train.py              # Tek eğitim veya k-fold CV CLI girişi
│   ├── search.py             # Optuna hiperparametre araması
│   ├── oof_predictions.py    # CV fold checkpoint'lerinden OOF tahminleri
│   ├── ensemble.py           # OOF-kilitli kalibre ensemble değerlendirmesi
│   ├── threshold_scan.py     # Kayıtlı olasılıklar üzerinde eşik taraması
│   └── README.md             # Model katmanı ayrıntıları
├── Utils/
│   ├── config.py             # Data, augmentasyon, model, eğitim ve arama ayarları
│   ├── metrics.py            # ROC-AUC, PR-AUC, F1, F-beta, MCC vb.
│   ├── calibration.py        # Temperature / isotonic kalibrasyon ve bootstrap eşik seçimi
│   ├── plot_metrics.py       # Görselleştirme yardımcıları
│   ├── reproducibility.py    # Rastgelelik kontrolü
│   └── README.md             # Yardımcı modüller ayrıntıları
├── Tests/
│   ├── test_dataset.py       # Gerçek veri, split ve cache kontrolleri
│   ├── test_model.py         # Model/factory çıktı şekli ve guard testleri
│   ├── test_search.py        # Optuna konfig rekonstrüksiyonu testleri
│   ├── test_smoke.py         # Sentetik veriyle hafif entegrasyon testleri
│   └── README.md             # Test kapsamı ayrıntıları
├── evaluate_final.py         # Kayıtlı run için final test raporu ve figürler
├── requirements.txt
└── README.md
```

## Veri Seti Özeti

Mevcut `ALAN/summary.json` dosyasına göre dağılım:

| Özellik | Değer |
|---|---:|
| ROI sayısı | 1584 |
| Hasta sayısı | 792 |
| Sol / sağ ROI | 792 / 792 |
| Normal ROI | 1234 |
| Anomali ROI | 350 |
| Train split | 1188 ROI |
| Dev split | 98 ROI |
| Test split | 298 ROI |
| NaN içeren örnek | 0 |

Splitler `ZS-train`, `ZS-dev` ve `ZS-test` olarak beklenir. Metadata üretimi sırasında aynı hastaya ait ROI kayıtlarının farklı splitlere düşmediği kontrol edilir.

## Kurulum

Unix benzeri kabuklar için:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell için:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Temel bağımlılıklar `requirements.txt` içinde tutulur: NumPy, scikit-learn, Optuna, matplotlib, pytest ve PyTorch. Dosyada CUDA 12.8 için `torch==2.11.0+cu128` pini bulunur; CUDA kullanmayan platformlarda PyTorch kurulumu ortamınıza göre ayrıca uyarlanabilir. PointNet uygulaması saf PyTorch kullanır; PyTorch Geometric, torch-scatter veya torch-cluster gerekmez.

## Hızlı Başlangıç

Metadata dosyalarını üretmek veya güncellemek:

```bash
python -m Preprocessing.analyze_dataset
```

Varsayılan ResNet3D akışına yakın tek eğitim:

```bash
python -m Model.train \
  --architecture resnet3d \
  --epochs 30 \
  --batch-size 8 \
  --learning-rate 2e-4 \
  --target-shape 64 64 64 \
  --output-dir outputs/resnet3d_baseline
```

U-Net3D veya PointNet denemek için:

```bash
python -m Model.train \
  --architecture unet3d \
  --epochs 30 \
  --batch-size 8 \
  --output-dir outputs/unet3d_baseline

python -m Model.train \
  --architecture pointnet \
  --pointnet-num-points 2048 \
  --pointnet-point-features 4 \
  --epochs 30 \
  --batch-size 8 \
  --output-dir outputs/pointnet_baseline
```

`--cv-folds 2` veya daha büyük verildiğinde `Model.train` tek train/dev/test akışı yerine k-fold cross-validation çalıştırır.

## Ön İşleme

`Preprocessing.dataset` her ROI için şu deterministik sırayı uygular:

1. `.npy` hacmi `float32` olarak yüklenir.
2. Seçilen NaN stratejisi uygulanır.
3. Foreground bbox ve `bbox_margin` ile crop yapılır.
4. İstenirse sağ böbrekler `right_flip_axis` üzerinde kanonikleştirilir.
5. Hacim küp forma pad edilir.
6. Trilinear interpolation ile `target_shape` boyutuna yeniden örneklenir.
7. Değerler `[0, 1]` aralığına kırpılır.

Varsayılan hedef boyut `64 x 64 x 64`, bbox margin `8`, cube padding açık, sağ böbrek kanonikleştirme kapalıdır. Desteklenen NaN stratejileri `none`, `drop_record`, `fill_zero`, `fill_mean`, `fill_median` ve `fill_constant` değerleridir.

Uzun eğitimlerde veya Optuna aramalarında deterministik ön işleme cache'i açılabilir:

```bash
python -m Model.train \
  --cache-mode disk \
  --cache-dir outputs/preprocessed_cache \
  --num-workers 4
```

Train split için augmentasyonlar `RandomFlip3D`, `RandomAffine3D` ve `RandomMorphology3D` ile uygulanır. Validation ve test splitlerinde augmentasyon kullanılmaz. Kapatmak için `--disable-augmentations` verilebilir.

Daha ayrıntılı veri formatı, metadata kolonları ve cache davranışı için [Preprocessing/README.md](Preprocessing/README.md) dosyasına bakın.

## Modeller

Tüm mimariler `Model.factory` üzerinden aynı arayüzle kurulur. Ana giriş şekli `B x 1 x D x H x W`, çıktı şekli `B` boyutlu logit vektörüdür ve kayıp varsayılan olarak `BCEWithLogitsLoss` tabanlıdır.

| `--architecture` | Açıklama | Öne çıkan ayarlar |
|---|---|---|
| `resnet3d` | ResNet3D-18 veya ResNet3D-34 sınıflandırıcı | `--depth`, `--base-channels`, `--dropout`, `--norm-type` |
| `unet3d` | Segmentasyon maskesi değil tek logit üreten 3B U-Net sınıflandırıcı | `--unet-depth`, `--unet-base-channels`, `--unet-channel-multiplier` |
| `pointnet` | Foreground voxel koordinatlarından sabit boyutlu point cloud ile sınıflandırma | `--pointnet-num-points`, `--pointnet-point-features`, `--pointnet-mlp-channels` |

Tabular ek özellikler varsayılan olarak açıktır. `log_voxel_count_z` ve `side_is_left`, yalnızca train split istatistikleriyle normalize edilerek model başlığına eklenir. Kapatmak için `--disable-tabular-features` kullanılabilir.

## Eğitim Akışı

`Model.engine` eğitim sırasında şu işleri yönetir:

1. Kayıtları yükler, splitlere ayırır ve DataLoader'ları kurar.
2. Sadece train split için augmentasyon uygular.
3. Sınıf dengesizliği için pozitif sınıf ağırlığı veya weighted sampler kullanabilir.
4. `adam`, `adamw` veya `sgd` optimizer'ı ve isteğe bağlı cosine scheduler/warmup ile eğitir.
5. `--primary-metric` skoruna göre en iyi checkpoint'i `best_model.pt` olarak kaydeder.
6. Validation tahminleri üzerinden temperature ve/veya isotonic kalibrasyon uygular.
7. `youden`, `f1`, `fbeta` veya `fixed` yöntemiyle karar eşiğini seçer.
8. Test seti için tuned ve fixed eşik metriklerini, tahminleri ve bootstrap güven aralıklarını kaydeder.

Sık kullanılan eğitim seçenekleri:

| Alan | CLI |
|---|---|
| Veri şekli | `--target-shape`, `--bbox-margin`, `--disable-bbox-crop`, `--disable-pad-to-cube` |
| NaN davranışı | `--nan-strategy`, `--nan-fill-value` |
| Cache | `--cache-mode none/memory/disk`, `--cache-dir` |
| Optimizasyon | `--optimizer`, `--learning-rate`, `--weight-decay`, `--scheduler`, `--warmup-epochs` |
| Dengesizlik | `--pos-weight-strategy`, `--use-weighted-sampler`, `--loss-type bce/focal`, `--focal-gamma` |
| Seçim/metrik | `--primary-metric`, `--threshold-selection`, `--threshold-fbeta`, `--threshold-min-specificity`, `--threshold-min-precision` |
| Kalibrasyon | `--calibration-method temperature/isotonic/temperature+isotonic`, `--disable-calibration` |
| Donanım | `--device auto/cuda/mps/cpu`, `--disable-amp`, `--num-workers` |

Tek eğitim çıktıları `--output-dir` altına yazılır:

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

CV modunda her fold kendi `fold_XX/` klasörüne `best_model.pt` ve `fold_result.json` yazar; kök klasörde `cv_summary.json` oluşur.

## Optuna Araması

Hiperparametre araması için:

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

`--n-folds 1` varsayılan eski train/dev objektifini kullanır; daha büyük değerlerde HPO skoru CV fold metriklerinden hesaplanır. Final `best_run/` yeniden eğitimi tek ZS train/dev/test akışını kullanır.

Arama çıktıları:

| Dosya/Klasör | Açıklama |
|---|---|
| `optuna_study.db` | Aynı `--output-dir` ve `--study-name` ile devam edilebilen SQLite study |
| `trial_000/`, `trial_001/`, ... | Her trial için eğitim veya CV çıktıları |
| `trial_summary.json` | Trial parametreleri ve skor özeti |
| `leaderboard.json` | Trial sıralaması |
| `best_run/` | En iyi trial parametreleriyle final yeniden eğitim |
| `best_model.pt` | En iyi final modelin arama kökündeki kopyası |
| `study_summary.json` | Arama ve final değerlendirme özeti |
| `final_evaluation/` | Başarılı olursa otomatik final test raporu |

Mimariye göre arama uzayı değişir: ResNet3D için derinlik/kanal sayısı, U-Net3D için encoder ayarları, PointNet için nokta sayısı/MLP/global boyut/input transform ve tüm mimariler için ortak eğitim, veri, augmentasyon, kalibrasyon ve eşik ayarları örneklenir.

## OOF Tahminleri ve Ensemble

CV fold checkpoint'lerinden out-of-fold tahmin üretmek için:

```bash
python -m Model.oof_predictions \
  --study-dir outputs/optuna_resnet3d \
  --trial-number 7 \
  --device auto
```

Bu komut `outputs/optuna_resnet3d/trial_007/oof_predictions.json` üretir. OOF tahminleri fold bazlı temperature kalibrasyonu, pooled isotonic kalibrasyon metadatası ve OOF üzerinden seçilmiş `f1_threshold` / `clinical_threshold` bilgilerini içerir.

Birden fazla CV trial'ı OOF üzerinde kilitlenen eşikle ensemble etmek için:

```bash
python -m Model.ensemble \
  --study-dirs outputs/optuna_resnet3d outputs/optuna_unet3d \
  --trial-numbers 7 4 \
  --output-dir outputs/ensemble_oof \
  --probability-mode arithmetic \
  --threshold-name f1_threshold
```

Ensemble test setinde eşik veya üye seçimi yapmaz. Eşik OOF tahminlerinden seçilir, test spliti bu kilitli eşikle bir kez raporlanır. Çıktılar:

```text
outputs/ensemble_oof/
├── ensemble_predictions.json
├── oof_thresholds.json
├── final_test_metrics.json
├── test_confidence_intervals.json
└── interpretation.txt
```

`--threshold-name clinical_threshold`, F-beta tabanlı ve minimum specificity korumalı klinik odaklı alternatifi kullanır. Olasılık birleştirme için `--probability-mode arithmetic` veya `logit` seçilebilir.

## Eşik Taraması

Kayıtlı validation/test olasılıkları üzerinde hızlı eşik incelemesi:

```bash
python -m Model.threshold_scan --run-dir outputs/resnet3d_baseline
```

Varsayılan olarak kalibre olasılıkları kullanılır. Ham olasılıklarla karşılaştırmak için `--use-uncalibrated` eklenebilir.

## Final Değerlendirme

`evaluate_final.py`, `best_model.pt` ve `config.json` bulunan bir run klasörünü okuyarak son test metriklerini, sınıflandırma raporlarını, tanılayıcı figürleri ve kısa yorumu üretir. Mimari `config.json` içinden okunduğu için ResNet3D, U-Net3D ve PointNet run'ları aynı CLI ile değerlendirilebilir.

```bash
python evaluate_final.py \
  --run-dir outputs/optuna_resnet3d/best_run \
  --use-saved-predictions
```

Varsayılan çıktı yolu `results/final_evaluation/<architecture>_<run>/` biçimindedir. `--threshold` ile kayıtlı tuned eşik yerine geçici bir karar eşiği verilebilir.

## Metrikler

Metrikler `Utils.metrics.compute_binary_classification_metrics` üzerinden merkezi olarak üretilir. Eğitim, ensemble ve final değerlendirme aynı yardımcıları kullanır.

Raporlanan ana metrikler: `accuracy`, `balanced_accuracy`, `precision`, `recall`, `f1`, `sensitivity`, `specificity`, `npv`, `fpr`, `fnr`, `fdr`, `for`, `mcc`, `cohen_kappa`, macro/weighted ortalamalar, `roc_auc`, `pr_auc`, confusion matrix bileşenleri ve destek sayılarıdır. Tek sınıflı `y_true` durumunda ROC-AUC/PR-AUC güvenli şekilde `NaN` döner.

Tek eğitimde varsayılan checkpoint seçimi `pr_auc` ile yapılır. Optuna aramasında varsayılan objektif `roc_auc` değeridir; iki akışta da metrik hesaplanamazsa yardımcı fonksiyon uygun fallback zincirine düşer.

Konfig, metrik, kalibrasyon, grafik ve tekrarlanabilirlik yardımcılarının ayrıntıları için [Utils/README.md](Utils/README.md) dosyasına bakın.

## Testler

Tüm testleri çalıştırmak:

```bash
python -m pytest Tests
```

`unittest` ile eşdeğer keşif:

```bash
python -m unittest discover -s Tests -p "test_*.py"
```

Gerçek ALAN dosyalarına bağlı testler `Tests/test_dataset.py` içindedir. Saf sentetik ve daha hızlı kontroller için önce şu komut kullanılabilir:

```bash
python -m pytest Tests/test_model.py Tests/test_search.py Tests/test_smoke.py
```

Test kapsamının ayrıntıları için [Tests/README.md](Tests/README.md) dosyasına bakın.

## Notlar

- `ALAN/info.csv` ve `ALAN/alan/*.npy` varsayılan veri konumlarıdır.
- `metadata.csv` eksikse veya beklenen NaN kolonlarını içermiyorsa otomatik olarak yeniden üretilebilir.
- `.npy` hacimleri değişirse bbox ve özet istatistikleri için metadata yeniden üretilmelidir.
- Sınıf dağılımı dengesiz olduğu için accuracy tek başına yeterli değildir; PR-AUC, ROC-AUC, balanced accuracy, sensitivity/specificity ve eşik davranışı birlikte yorumlanmalıdır.
- Bu kod araştırma ve deney amaçlıdır; klinik karar verme amacıyla doğrulanmış bir sistem olarak kullanılmamalıdır.
