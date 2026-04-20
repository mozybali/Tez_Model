# ALAN 3B Böbrek Anomali Sınıflandırması

Bu proje, ALAN veri setindeki böbrek ROI hacimlerinde anomali olup olmadığını 3B ResNet modeliyle ikili sınıflandırma olarak ele alır. Girdi olarak `.npy` formatındaki 3B binary maske hacimleri kullanılır; eğitim sürecinde ön işleme, isteğe bağlı augmentasyon, tabular ek özellikler, erken durdurma ve Optuna tabanlı hiperparametre araması desteklenir.

## Proje Yapısı

```text
TezModel/
├── ALAN/
│   ├── info.csv              # ROI_id, subset ve ROI_anomaly alanları
│   ├── metadata.csv          # Üretilmiş ROI istatistikleri
│   ├── summary.json          # Veri seti özeti
│   └── alan/                 # 3B .npy hacim dosyaları
├── Preprocessing/
│   ├── analyze_dataset.py    # metadata.csv ve summary.json üretimi
│   ├── dataset.py            # Kayıt okuma, split ayırma ve ön işleme
│   └── transforms.py         # 3B flip, affine ve morfolojik augmentasyonlar
├── Model/
│   ├── resnet3d.py           # PyTorch ile ResNet3D-18 / ResNet3D-34
│   ├── unet3d.py             # 3B U-Net tabanlı sınıflandırıcı (ikili anomali etiketi)
│   ├── pointnet.py           # 3B maskelerden türetilen point cloud üzerinde çalışan PointNet
│   ├── factory.py            # architecture alanına göre model kuran ortak fabrika
│   ├── engine.py             # Eğitim, doğrulama, test, kalibrasyon ve TTA akışı
│   ├── train.py              # Tek eğitim çalıştırması için CLI
│   ├── search.py             # Optuna ile hiperparametre araması
│   └── ensemble.py           # En iyi K Optuna trial'ı üzerinde soft-voting ensemble
├── Utils/
│   ├── config.py             # Data, augmentasyon, model, eğitim ve arama ayarları
│   ├── metrics.py            # ROC-AUC, PR-AUC, F1, F-beta, balanced accuracy vb.
│   ├── calibration.py        # Temperature / isotonic kalibrasyon ve bootstrap eşik seçimi
│   ├── plot_metrics.py       # Metrik görselleştirme yardımcıları
│   └── reproducibility.py    # Rastgelelik kontrolü
├── Tests/
│   ├── test_dataset.py       # Veri seti ve şekil kontrolleri
│   ├── test_model.py         # Model çıktı şekli kontrolleri
│   └── test_smoke.py         # Sentetik veriyle uçtan uca duman testi
├── requirements.txt
└── README.md
```

## Veri Seti Özeti

Mevcut `ALAN/summary.json` dosyasına göre veri seti aşağıdaki dağılıma sahiptir:

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

Splitler `ZS-train`, `ZS-dev` ve `ZS-test` olarak tanımlıdır. Metadata üretimi sırasında aynı hastaya ait sol ve sağ ROI kayıtlarının farklı splitlere düşmediği kontrol edilir.

## Kurulum

Komutları proje kök dizininden çalıştırın.

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

Temel gereksinimler:

- Python 3.10 veya üzeri
- PyTorch 2.0 veya üzeri (CUDA wheel kurulumu `requirements.txt` içinde korunur)
- NumPy 1.24 veya üzeri
- scikit-learn 1.0 veya üzeri
- Optuna 3.0 veya üzeri
- matplotlib, pytest

PointNet mimarisi saf PyTorch ile uygulanmıştır; PyTorch Geometric, torch-scatter veya torch-cluster gibi nokta bulutuna özgü kütüphaneler gerekmez. Bu nedenle `requirements.txt` değişmeden kalır.

## Hızlı Başlangıç

Önce metadata dosyalarını üretin veya güncelleyin:

```bash
python -m Preprocessing.analyze_dataset
```

Ardından varsayılanlara yakın bir eğitim çalıştırın:

```bash
python -m Model.train \
  --epochs 20 \
  --batch-size 8 \
  --learning-rate 1e-3 \
  --depth 18 \
  --base-channels 32 \
  --dropout 0.2 \
  --optimizer adamw \
  --target-shape 64 64 64 \
  --bbox-margin 8 \
  --decision-threshold 0.5 \
  --threshold-selection youden \
  --calibration-method temperature \
  --output-dir outputs/baseline
```

Recall'u öne çıkarmak isterseniz `--threshold-selection fbeta --threshold-fbeta 1.5`, kalibrasyonun yetersiz kaldığı durumlarda `--calibration-method temperature+isotonic`, değerlendirmede flip tabanlı test-time augmentation için `--tta` kullanılabilir.

Eğitim tamamlandığında model ve metrikler `outputs/baseline/` altına yazılır.

## Ön İşleme

Her ROI için uygulanan ana adımlar şunlardır:

1. `.npy` hacmi `float32` olarak yüklenir.
2. Seçilen NaN stratejisi uygulanır.
3. Foreground voxel sınırlarına göre tight bounding box crop yapılır.
4. İstenirse sağ böbrekler sol tarafa kanonikleştirilir.
5. Hacim küp forma pad edilir.
6. Trilinear interpolation ile hedef boyuta yeniden örneklenir.
7. Tensor değerleri `[0, 1]` aralığına kırpılır.

Desteklenen NaN stratejileri:

| Strateji | Açıklama |
|---|---|
| `none` | NaN değerlerine dokunmaz. Varsayılan davranıştır. |
| `drop_record` | NaN içeren kayıtları splitlerden çıkarır. |
| `fill_zero` | NaN voxel değerlerini `0` ile doldurur. |
| `fill_mean` | NaN voxel değerlerini hacmin ortalamasıyla doldurur. |
| `fill_median` | NaN voxel değerlerini hacmin medyanıyla doldurur. |
| `fill_constant` | NaN voxel değerlerini `--nan-fill-value` ile doldurur. |

Varsayılan hedef boyut `64 x 64 x 64` olarak ayarlanmıştır. Sağ böbreği kanonikleştirme (`--canonicalize-right`) varsayılan olarak kapalıdır.

## Model

Proje üç farklı 3B omurga mimarisini destekler ve `--architecture` argümanı ile seçilir:

- `resnet3d` (varsayılan): Saf PyTorch ile yazılmış ResNet3D. `--depth 18` ve `--depth 34` desteklenir.
- `unet3d`: Sınıflandırma için uyarlanmış hafif bir 3B U-Net. Encoder-decoder ve skip bağlantılarından sonra global pooling ve küçük bir sınıflandırıcı başlığı kullanır; çıktı segmentasyon maskesi değil, logit değeridir.
- `pointnet`: İkili 3B ROI maskesinden foreground voxellerin koordinatlarını çıkartarak sabit boyutlu bir point cloud oluşturur, koordinatları yaklaşık `[-1, 1]` aralığına normalize eder ve shared MLP + global max-pooling ile sınıflandırma logiti üretir. Eğitimde replacement ile rastgele örnekleme, değerlendirmede deterministik örnekleme kullanılır; boş maskeler güvenli bir şekilde sıfır noktalara düşürülür. Uygulama saf PyTorch'tur; PyTorch Geometric, torch-scatter veya torch-cluster gibi ek bağımlılıklar gerektirmez.

Ana giriş her üç mimaride de `1 x D x H x W` şekilli 3B ROI maskesidir ve çıktı `BCEWithLogitsLoss` ile uyumlu `(B,)` şekilli logit vektörüdür.

U-Net seçildiğinde ek CLI argümanları kullanılabilir:

| Parametre | Varsayılan | Açıklama |
|---|---:|---|
| `--unet-depth` | `4` | Encoder seviyesi sayısı |
| `--unet-base-channels` | `16` | İlk seviyedeki kanal sayısı |
| `--unet-channel-multiplier` | `2` | Her seviye arasındaki kanal çarpanı |
| `--unet-bottleneck-channels` | `None` | Dar boğaz kanal sayısını açıkça belirlemek için opsiyonel |

Örnek U-Net eğitimi:

```bash
python -m Model.train \
  --architecture unet3d \
  --unet-depth 4 \
  --unet-base-channels 16 \
  --unet-channel-multiplier 2 \
  --epochs 20 \
  --batch-size 8 \
  --target-shape 64 64 64 \
  --output-dir outputs/unet3d_baseline
```

PointNet seçildiğinde ek CLI argümanları kullanılabilir:

| Parametre | Varsayılan | Açıklama |
|---|---:|---|
| `--pointnet-num-points` | `1024` | Her örnek için sabit örneklenen foreground voxel sayısı |
| `--pointnet-point-features` | `3` | `3` yalnızca xyz, `4` ek bir occupancy kanalı ekler |
| `--pointnet-mlp-channels` | `64 128 256` | Global pooling öncesi shared-MLP kanal zinciri |
| `--pointnet-global-dim` | `512` | Max-pool sonrası global öznitelik boyutu |
| `--pointnet-head-hidden-dim` | `128` | Sınıflandırıcı MLP'nin gizli boyutu (`0` → tek doğrusal katman) |
| `--pointnet-binary-threshold` | `0.5` | Maske değerini foreground voxel olarak sayan eşik |
| `--pointnet-use-input-transform` | kapalı | Birim matrise initialize edilmiş küçük bir T-Net'i xyz üzerinde uygular |

Örnek PointNet eğitimi:

```bash
python -m Model.train \
  --architecture pointnet \
  --pointnet-num-points 1024 \
  --pointnet-mlp-channels 64 128 256 \
  --pointnet-global-dim 512 \
  --pointnet-head-hidden-dim 128 \
  --epochs 20 \
  --batch-size 8 \
  --target-shape 64 64 64 \
  --output-dir outputs/pointnet_baseline
```

Varsayılan olarak modele iki ek tabular özellik de verilir:

| Özellik | Açıklama |
|---|---|
| `log_voxel_count_z` | `log1p(voxel_count)` değerinin train split istatistikleriyle normalize edilmiş hali |
| `side_is_left` | Sol ROI için `1`, sağ ROI için `0` |

Bu özellikler yalnızca train split üzerinden hesaplanan istatistiklerle normalize edilir; doğrulama ve test bilgisinin eğitime sızması engellenir. Ek özellikleri kapatmak için `--disable-tabular-features` kullanılabilir.

## Eğitim Akışı

`Model.train` aşağıdaki işleri tek çalıştırmada yürütür:

- veri kayıtlarını yükler ve splitlere ayırır,
- sadece train split için augmentasyon uygular,
- sınıf dengesizliği için `BCEWithLogitsLoss` içinde pozitif sınıf ağırlığı kullanır,
- seçilen optimizer ve scheduler ile eğitir,
- doğrulama skoruna göre en iyi checkpoint'i kaydeder,
- validasyon üzerinde post-hoc olasılık kalibrasyonu (temperature ve/veya isotonic) uygular,
- bootstrap-ortalamalı karar eşiğini (Youden / F1 / F-beta) validasyondan seçer,
- isteğe bağlı flip tabanlı test-time augmentation ile en iyi checkpoint üzerinde test setini değerlendirir.

Cihaz seçimi `--device auto` ile otomatik yapılır. Uygunsa CUDA, ardından Apple MPS, aksi halde CPU kullanılır.

Başlıca eğitim parametreleri:

| Parametre | Varsayılan | Açıklama |
|---|---:|---|
| `--epochs` | `20` | Maksimum epoch sayısı |
| `--batch-size` | `8` | Mini-batch boyutu |
| `--learning-rate` | `1e-3` | Öğrenme oranı |
| `--optimizer` | `adamw` | `adam`, `adamw` veya `sgd` |
| `--scheduler` | `cosine` | `cosine` veya `none` |
| `--patience` | `6` | Erken durdurma sabrı |
| `--gradient-clip-norm` | `1.0` | Gradient clipping sınırı |
| `--decision-threshold` | `0.5` | `fixed` modunda kullanılan sabit eşik |
| `--threshold-selection` | `youden` | `youden`, `f1`, `fbeta` veya `fixed` |
| `--threshold-fbeta` | `1.0` | `fbeta` modunda beta; >1 recall'u öne çıkarır |
| `--calibration-method` | `temperature` | `temperature`, `isotonic` veya `temperature+isotonic` |
| `--tta` | kapalı | Değerlendirmede flip tabanlı test-time augmentation |
| `--disable-calibration` | kapalı | Tüm post-hoc kalibrasyonu devre dışı bırakır |

Eğitim çıktıları:

| Dosya | Açıklama |
|---|---|
| `config.json` | Çalıştırmada kullanılan tam konfigürasyon |
| `history.json` | Epoch bazlı train ve validation metrikleri |
| `best_val_metrics.json` | En iyi validation epoch metrikleri |
| `test_metrics.json` | En iyi checkpoint ile test metrikleri |
| `calibration.json` | Temperature / isotonic kalibrasyon, reliability bin'leri, seçilen eşik ve kalibre probabiliteler |
| `best_model.pt` | En iyi model checkpoint'i |

## Augmentasyonlar

Augmentasyonlar yalnızca train split üzerinde uygulanır.

| Augmentasyon | Açıklama |
|---|---|
| `RandomFlip3D` | Seçilen eksenlerde olasılıksal flip |
| `RandomAffine3D` | Küçük rotasyon, öteleme ve ölçekleme |
| `RandomMorphology3D` | Düşük olasılıklı dilatasyon veya erozyon |

Veri binary maske olduğu için brightness, contrast veya noise gibi yoğunluk augmentasyonları kullanılmaz. Tüm augmentasyonları kapatmak için `--disable-augmentations` verilebilir.

## Hiperparametre Araması

Optuna tabanlı arama için:

```bash
python -m Model.search \
  --n-trials 30 \
  --epochs 12 \
  --device auto \
  --output-dir outputs/optuna \
  --final-epochs 20
```

`--epochs`, her trial için maksimum epoch bütçesidir; arama sırasında bu bütçenin daha kısa alt değerleri de denenebilir. `--final-epochs` verilirse en iyi trial parametreleriyle `best_run/` altında bu epoch sayısıyla yeniden eğitim yapılır. Verilmezse en iyi trial'ın epoch bütçesi kullanılır.

Arama uzayı mimari seçimini (`resnet3d` / `unet3d` / `pointnet`), learning rate, weight decay, optimizer, scheduler, batch size, epoch bütçesi, early stopping patience, gradient clipping, eşik seçimi stratejisi (`youden` / `f1` / `fbeta`) ve F-beta değeri, kalibrasyon yöntemi (`temperature` / `isotonic` / `temperature+isotonic`), flip tabanlı TTA, AMP, weighted sampler, ResNet derinliği ve kanal sayısı, U-Net derinliği / base channel / kanal çarpanı, PointNet için `pointnet_num_points` ∈ {512, 1024, 2048, 4096}, global dim, MLP varyantı (`small` / `medium` / `large`), sınıflandırıcı head boyutu, nokta özellik sayısı ve opsiyonel input transform, dropout, tabular özellik kullanımı, hedef hacim boyutu, bounding box crop, cube padding, sağ böbrek kanonikleştirme, NaN stratejisi ve augmentasyon parametrelerini kapsar. Mimariye özgü hiperparametreler yalnızca ilgili mimari seçildiğinde örneklenir; eski (mimari alanı içermeyen) trial parametreleri geriye dönük olarak ResNet3D varsayımıyla yeniden kurulur, PointNet-özgü alanları eksik trial'lar ise PointNet için güvenli varsayılanlara düşer.

Arama uzayında PointNet'i de içeren örnek komut:

```bash
python -m Model.search \
  --n-trials 30 \
  --epochs 12 \
  --device auto \
  --output-dir outputs/optuna_with_pointnet
```

Optuna çıktıları:

| Dosya/Klasör | Açıklama |
|---|---|
| `trial_000/`, `trial_001/`, ... | Her trial için eğitim çıktıları |
| `trial_summary.json` | İlgili trial parametreleri ve metrikleri |
| `leaderboard.json` | Trial sıralaması |
| `best_run/` | En iyi parametrelerle yeniden eğitim |
| `best_model.pt` | En iyi modelin kök arama klasöründeki kopyası |
| `study_summary.json` | Genel arama özeti |

## Ensemble

Arama tamamlandıktan sonra en iyi K trial'ın checkpoint'leri soft-voting ile birleştirilebilir:

```bash
python -m Model.ensemble \
  --study-dir outputs/optuna \
  --top-k 5
```

`Model.ensemble`, `leaderboard.json` üzerinden en iyi K trial'ı seçer, her trial'ın `best_model.pt` ve `checkpoint_meta.json` dosyalarından konfigürasyonu yeniden kurar (ResNet3D, U-Net3D ve PointNet trial'ları karışık olsa bile her bir trial doğru mimaride yeniden oluşturulur), test splitinde kalibre probabiliteleri ortalar ve bootstrap güven aralıklarıyla birlikte ensemble metriklerini raporlar.

## Kalibrasyon ve Eşik Seçimi

`Utils/calibration.py` şu araçları sağlar:

- `fit_temperature` / `apply_temperature`: validasyon logitleri üzerinde tek skalerli temperature scaling.
- `fit_isotonic` / `apply_isotonic`: temperature sonrası olasılıklar üzerinde monoton parçalı-lineer isotonic regresyon (yoğunluk dağılımı bimodal veya çarpık olduğunda faydalıdır).
- `reliability_bins` ve `expected_calibration_error`: reliability diyagramı ve ECE hesabı.
- `select_threshold_bootstrap`: bootstrap-ortalamalı Youden, F1 veya F-beta ile eşik seçimi — tek splitten kaynaklanan varyansı azaltır.

Seçilen yöntem ve sonuçlar her çalıştırmanın `calibration.json` dosyasına yazılır; test metrikleri kalibre edilmiş probabilitelerle ve seçilen eşikle raporlanır.

## Testler

Pytest ile:

```bash
python -m pytest Tests/ -v
```

Standart unittest keşfi ile:

```bash
python -m unittest discover -s Tests -v
```

Testler veri seti dönüşümlerini, model çıktı şekillerini ve sentetik veriyle uçtan uca eğitim akışını kontrol eder.

## Metrikler

Eğitim ve değerlendirme sırasında aşağıdaki metrikler hesaplanır:

- loss
- accuracy
- balanced accuracy
- precision
- recall
- F1 ve F-beta (eşik seçimi için)
- ROC-AUC
- PR-AUC
- expected calibration error (ECE) — kalibrasyon öncesi/sonrası
- confusion matrix bileşenleri (`tn`, `fp`, `fn`, `tp`)

Model seçimi varsayılan olarak `roc_auc` üzerinden yapılır. Seçilen metrik hesaplanamazsa sırasıyla `pr_auc`, `balanced_accuracy`, `f1` ve negatif loss değerine düşülür.

## Notlar

- Komutlar proje kök dizininden çalıştırılmalıdır.
- `ALAN/info.csv` ve `ALAN/alan/*.npy` dosyaları veri yükleme için beklenen varsayılan konumlardır.
- `metadata.csv` eksikse veya beklenen NaN kolonlarını içermiyorsa otomatik olarak yeniden üretilebilir.
- Sınıf dağılımı dengesiz olduğu için accuracy tek başına yeterli değildir; özellikle PR-AUC, ROC-AUC ve balanced accuracy birlikte yorumlanmalıdır.
- Bu kod araştırma ve deney amaçlıdır; klinik karar verme amacıyla doğrulanmış bir sistem olarak kullanılmamalıdır.
