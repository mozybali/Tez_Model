# Tests Modulu

Bu klasor, ALAN 3B bobrek anomali siniflandirma projesinin veri hazirlama, model mimarileri, Optuna konfig rekonstruksiyonu, metrikler ve hafif uc-tan-uca egitim davranislarini dogrulayan testleri icerir. Testler `unittest` siniflariyla yazilmistir; proje icinde hem `python -m unittest` hem de `pytest` ile calistirilabilir.

## Dosya Yapisi

```text
Tests/
+-- __init__.py
+-- test_dataset.py       # Gercek ALAN metadata/veri splitleri ve dataset cache testleri
+-- test_model.py         # ResNet3D, U-Net3D, PointNet ve Model.factory sekil/guard testleri
+-- test_search.py        # Optuna search yardimcilari ve trial parametrelerinden config kurma testleri
+-- test_smoke.py         # Sentetik veriyle on isleme, metrik, import ve mini egitim duman testleri
`-- README.md             # Tests klasoru icin bu dokumantasyon
```

## Projedeki Yeri

Testler su proje katmanlarini kontrol eder:

| Katman | Ilgili testler |
|---|---|
| `Preprocessing.dataset` | `test_dataset.py`, `test_smoke.py` |
| `Model.resnet3d` | `test_model.py`, `test_smoke.py` |
| `Model.unet3d` | `test_model.py` |
| `Model.pointnet` | `test_model.py` |
| `Model.factory` | `test_model.py` |
| `Model.search` | `test_search.py` |
| `Model.engine` tabular yardimcilari | `test_smoke.py` |
| `Utils.metrics` | `test_smoke.py` |
| `evaluate_final.py` import sozlesmesi | `test_smoke.py` |

## Calistirma

Tum testleri proje kokunden calistirmak icin:

```bash
python -m pytest Tests
```

`unittest` ile ayni kapsam:

```bash
python -m unittest discover -s Tests -p "test_*.py"
```

Tek dosya calistirma ornekleri:

```bash
python -m pytest Tests/test_model.py
python -m pytest Tests/test_search.py
python -m pytest Tests/test_smoke.py
python -m pytest Tests/test_dataset.py
```

`test_dataset.py` gercek `ALAN/` klasorune baglidir. Sadece sentetik veriye dayanan hizli kontroller icin once `test_model.py`, `test_search.py` ve `test_smoke.py` calistirilabilir.

## Veri Bagimliligi

`test_dataset.py`, proje kokunde su dosyalarin bulunmasini bekler:

```text
ALAN/
+-- info.csv
+-- metadata.csv
+-- summary.json
`-- alan/
    `-- <ROI_id>.npy
```

Bu test dosyasi mevcut ALAN dagilimini sabit varsayimlarla kontrol eder:

| Kontrol | Beklenen deger |
|---|---:|
| Toplam ROI kaydi | `1584` |
| Train split | `1188` |
| Val/dev split | `98` |
| Test split | `298` |

Metadata eksikse veya eski formatta ise `Preprocessing.dataset.load_records`, `Preprocessing.analyze_dataset.ensure_metadata` uzerinden `ALAN/metadata.csv` ve `ALAN/summary.json` dosyalarini yeniden uretebilir. Bu yuzden dataset testleri gercek veri dosyalarini okuyabilir ve metadata dosyalarini guncelleyebilir.

## Test Dosyalari

### `test_dataset.py`

Gercek ALAN kayitlari ve `AlanKidneyDataset` davranisini kontrol eder.

Kapsam:

- `load_records` ile metadata okuma.
- `split_records` ile `ZS-train`, `ZS-dev`, `ZS-test` ayrimi.
- Hasta seviyesinde split sizintisi olmamasi.
- Dataset orneginin `1 x 64 x 64 x 64` hacim tensoru ve ikili label dondurmesi.
- `cache_mode="disk"` ile on islenmis tensorun `.npy` olarak yazilmasi.
- `cache_mode="memory"` icin dataset'in clone dondurmesi; disaridan degistirilen tensor cache'i bozmamali.
- Disk cache yaziminda `os.replace` yarisi olursa mevcut hedef dosyanin korunmasi ve gecici dosyanin temizlenmesi.

Bu dosya hem gercek veri setine hem de gecici sentetik hacimlere dokunur.

### `test_model.py`

Model mimarilerinin cikti sekillerini, hatali parametre guard'larini ve `Model.factory` secimini test eder.

Kapsam:

- `build_resnet3d` icin ResNet3D-18 ve ResNet3D-34 logit sekli.
- ResNet3D icin tabular ozellikli ve tek ornekli forward.
- Gecersiz ResNet depth degeri icin `ValueError`.
- `build_unet3d_classifier` icin tabularli/tabularsiz forward.
- U-Net icin kucuk input smoke testi.
- Gecersiz U-Net `depth`, `base_channels`, `channel_multiplier`, `dropout` guard'lari.
- `build_pointnet_classifier` icin tabularli/tabularsiz forward.
- Bos maske durumunda PointNet'in finite logit uretmesi.
- Foreground voxel sayisi `num_points` degerinden az oldugunda deterministik oversampling.
- `point_features=4`, `use_input_transform=True` ve eval modunda deterministik sampling.
- `build_model` fabrika fonksiyonunun `resnet3d`, `unet3d`, `pointnet` kurmasi ve bilinmeyen mimariyi reddetmesi.

Bu testler sentetik tensor kullanir; ALAN verisine ihtiyac duymaz.

### `test_search.py`

`Model.search` icindeki Optuna yardimci fonksiyonlarini ve eski/yeni trial parametrelerinden dataclass konfiglerinin geri kurulmasini test eder.

Kapsam:

- `_epoch_choices` ve `_patience_choices` seceneklerinin sinir/monotonluk davranisi.
- `_flip_axes_from_choice` secenek cozumu.
- `_resolve_flip_axes` ile `canonicalize_right=True` iken canonical flip ekseninin augmentasyon fliplerinden cikarilmasi.
- `_configs_from_params` ile `DataConfig`, `AugmentationConfig`, `ModelConfig`, `TrainConfig` geri kurulumu.
- Augmentasyon acik/kapali parametre setlerinin `KeyError` uretmemesi.
- `epochs_override`, `warmup_epochs`, `pos_weight_strategy` ve varsayilan fallback davranislari.
- Eski trial parametrelerinde `architecture` yoksa `resnet3d` varsayimina dusulmesi.
- `unet3d` ve `pointnet` mimari parametrelerinin round-trip edilmesi.
- Eksik PointNet parametrelerinde `ModelConfig` taban varsayilanlarinin korunmasi.

Bu testler veri dosyasi okumaz; saf konfig ve yardimci fonksiyon testleridir.

### `test_smoke.py`

Sentetik veriyle daha genis entegrasyon yuzeyini hafif sekilde yoklar.

Kapsam:

- `crop_to_bbox`, `pad_to_cube`, `resize_volume` on isleme yardimcilari.
- `compute_binary_classification_metrics`, `compute_per_class_report`, `select_model_score`.
- Confusion-matrix turevleri: sensitivity, specificity, NPV, FPR, FNR, FDR, FOR, MCC, Cohen kappa, macro/weighted ortalamalar.
- Tek sinifli `y_true` durumunda ROC-AUC/PR-AUC hesaplarinin crash etmemesi ve `nan` donmesi.
- `evaluate_final.run_final_evaluation` import yuzeyi ve varsayilan imzasi.
- Gecici sentetik `.npy` hacimlerle mini ResNet3D forward + BCE loss smoke testi.
- Tabular ozellik olusturma: `log_voxel_count_z` ve `side_is_left`.
- `apply_nan_strategy` icin `none`, `fill_zero`, `fill_mean`, `fill_median`, `fill_constant` ve tum-NaN fallback davranisi.
- NaN iceren sentetik dataset icin `nan_strategy="fill_zero"`.
- `drop_record` mantiginin `has_nan` alanina gore filtrelenebilmesi.

Bu dosya gercek ALAN verisine ihtiyac duymaz; gecici klasorlerde sentetik hacimler olusturur.

## Hangi Test Ne Zaman Calistirilmeli?

Hizli model ve konfig kontrolu:

```bash
python -m pytest Tests/test_model.py Tests/test_search.py
```

Preprocessing, metrik ve hafif entegrasyon kontrolu:

```bash
python -m pytest Tests/test_smoke.py
```

Gercek veri seti ve metadata kontrolu:

```bash
python -m pytest Tests/test_dataset.py
```

Tum regresyon seti:

```bash
python -m pytest Tests
```

## Test Yazarken Dikkat Edilecekler

- Gercek ALAN dosyalarina bagli yeni testler `test_dataset.py` icinde tutulmali veya README'de veri bagimliligi acikca belirtilmelidir.
- Saf birim testleri mumkunse sentetik tensor/hacim kullanmali; bu tarz testler `test_model.py`, `test_search.py` veya `test_smoke.py` icin daha uygundur.
- Uzun egitim, Optuna aramasi veya buyuk veri isleyen testlerden kacinilmalidir. Mevcut smoke testleri sadece forward/loss seviyesinde kalir.
- Yeni model mimarisi eklendiginde `Model.factory` icin en az bir cikti sekli testi eklenmelidir.
- Metrik JSON sozlesmesini degistiren her duzenleme, `test_smoke.py` icindeki geriye uyumluluk anahtar kontrolleriyle birlikte guncellenmelidir.
- Cache davranisi degistirilirse `test_dataset.py` icindeki disk ve memory cache testleri de guncellenmelidir.

## Gereksinimler

Testler proje bagimliliklarini kullanir:

- `pytest` veya Python `unittest`
- `numpy`
- `torch`
- `scikit-learn`
- `matplotlib` (`evaluate_final.py` import kontrolu icin)

Bagimlilikler proje kokundeki `requirements.txt` uzerinden kurulabilir:

```bash
pip install -r requirements.txt
```
