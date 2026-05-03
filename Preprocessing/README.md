# Preprocessing Modulu

Bu klasor, ALAN veri setindeki 3B bobrek ROI maskelerini model egitimine hazirlayan veri analizi, metadata uretimi, deterministik on isleme ve train-time augmentasyon kodlarini icerir. Model egitimi sirasinda bu modul dogrudan `Model.engine` tarafindan kullanilir; komut satirindan calistirilabilen ana giris ise `analyze_dataset.py` dosyasidir.

## Dosya Yapisi

```text
Preprocessing/
+-- __init__.py
+-- analyze_dataset.py    # ALAN/info.csv ve .npy hacimlerden metadata.csv + summary.json uretir
+-- dataset.py            # AlanRecord, split ayirma, on isleme, cache ve PyTorch Dataset
+-- transforms.py         # Train split icin 3B flip, affine ve morfolojik augmentasyonlar
`-- README.md             # Preprocessing klasoru icin bu dokumantasyon
```

## Projedeki Yeri

Preprocessing akisi su dosya ve modullere baglidir:

- Girdi kayitlari: `ALAN/info.csv`
- Girdi hacimleri: `ALAN/alan/<ROI_id>.npy`
- Uretilen metadata: `ALAN/metadata.csv`
- Uretilen ozet: `ALAN/summary.json`
- Ayarlar: `Utils.config.DataConfig` ve `Utils.config.AugmentationConfig`
- Egitim baglantisi: `Model.engine.build_dataloaders`
- Testler: `Tests/test_dataset.py` ve `Tests/test_smoke.py`

Model tarafi `load_records`, `split_records`, `AlanKidneyDataset` ve `build_train_augmentations` fonksiyonlarini kullanarak train/val/test DataLoader'larini kurar. Tabular ek ozellikler `Model.engine` icinde bu moduldaki `voxel_count` ve `side` alanlarindan uretilir.

## Beklenen Veri Formati

`ALAN/info.csv` dosyasi en az su kolonlari icermelidir:

| Kolon | Aciklama |
|---|---|
| `ROI_id` | Hacim dosyasinin temel adi. Ornek: `PATIENT001_L`; karsilik gelen dosya `ALAN/alan/PATIENT001_L.npy` olur. |
| `subset` | Split etiketi. Varsayilanlar `ZS-train`, `ZS-dev`, `ZS-test`. |
| `ROI_anomaly` | `TRUE` ise anomali etiketi `1`, aksi halde `0` kabul edilir. |

Hacimler `.npy` formatinda 3 boyutlu numpy array olmalidir. Egitim boru hatti bu hacimleri `float32` tensore cevirir ve model girisi olarak `1 x D x H x W` sekline getirir.

`ROI_id` sonundaki `_L` veya `_R` parcasi `side` bilgisi olarak kullanilir. Hasta kimligi `ROI_id.rsplit("_", 1)[0]` ile turetilir; metadata uretiminde ayni hastaya ait ROI'lerin farkli splitlere dagilmadigi kontrol edilir.

## Metadata Uretimi

Metadata ve veri seti ozetini uretmek icin proje kokunden:

```bash
python -m Preprocessing.analyze_dataset \
  --info-csv ALAN/info.csv \
  --volumes-dir ALAN/alan \
  --metadata-csv ALAN/metadata.csv \
  --summary-json ALAN/summary.json
```

Bu komut tum `ROI_id` kayitlari icin ilgili `.npy` dosyasini yukler, foreground voxel'lerden bounding box hesaplar, temel istatistikleri cikarir ve hasta seviyesinde split sizintisi kontrolu yapar.

`metadata.csv` kolonlari:

| Kolon | Aciklama |
|---|---|
| `ROI_id` | ROI kimligi |
| `subset` | Orijinal split etiketi |
| `ROI_anomaly` | Orijinal anomali etiketi |
| `label_int` | `ROI_anomaly == TRUE` icin `1`, aksi halde `0` |
| `side` | ROI taraf bilgisi (`L` veya `R`) |
| `volume_path` | Hacim dosya adi veya yolu |
| `voxel_count` | Sifirdan farkli voxel sayisi |
| `bbox_min_d/h/w` | Foreground bounding box minimum koordinatlari |
| `bbox_max_d/h/w` | Foreground bounding box maksimum koordinatlari |
| `center_d/h/w` | Foreground voxel koordinatlarinin ortalamasi; bos maskede `0` |
| `shape_d/h/w` | Orijinal hacim boyutu |
| `nan_count` | Hacimdeki NaN voxel sayisi |
| `nan_ratio` | `nan_count / volume.size` |
| `has_nan` | NaN varsa `1`, yoksa `0` |

`summary.json` alanlari:

| Alan | Aciklama |
|---|---|
| `samples` | Toplam ROI sayisi |
| `patients` | Hasta sayisi |
| `subset_counts` | Split bazli ROI dagilimi |
| `label_counts` | Etiket dagilimi |
| `split_label_counts` | Split + etiket dagilimi |
| `side_counts` | Sol/sag ROI dagilimi |
| `bbox_mean`, `bbox_p95` | Bounding box boyutu ortalamasi ve 95. yuzdeligi |
| `voxel_count_mean/std/min/max` | Foreground voxel sayisi istatistikleri |
| `nan_samples`, `nan_total_voxels`, `nan_split_counts` | NaN ozeti |

`dataset.load_records` icindeki `ensure_metadata` cagrisi, `metadata.csv` yoksa veya eski metadata `nan_count`, `nan_ratio`, `has_nan` kolonlarini icermiyorsa metadata dosyasini otomatik yeniden uretir.

## Dataset Akisi

`dataset.py` icindeki ana API:

| API | Gorev |
|---|---|
| `AlanRecord` | Tek ROI kaydini tutan dataclass |
| `load_records(...)` | `metadata.csv` okuyup `list[AlanRecord]` dondurur |
| `split_records(...)` | Kayitlari `train`, `val`, `test` listelerine ayirir |
| `AlanKidneyDataset` | PyTorch `Dataset`; hacmi on isleyip batch icin sozluk dondurur |
| `infer_positive_class_weight(...)` | Pozitif sinif agirligi icin `negatives / positives` hesabi |
| `crop_to_bbox`, `pad_to_cube`, `resize_volume` | Deterministik on isleme yardimcilari |
| `apply_nan_strategy` | NaN doldurma stratejilerini uygular |

`AlanKidneyDataset.__getitem__` su alanlari dondurur:

| Alan | Tip | Aciklama |
|---|---|---|
| `id` | `str` | ROI kimligi |
| `volume` | `torch.Tensor` | `1 x target_d x target_h x target_w` hacim tensoru |
| `label` | `torch.float32` tensor | Ikili etiket (`0.0` veya `1.0`) |
| `side` | `str` | `L` veya `R` |
| `subset` | `str` | Orijinal subset etiketi |
| `voxel_count` | `torch.float32` tensor | Metadata'dan gelen foreground voxel sayisi |

## Deterministik On Isleme Sirasi

Her hacim icin `AlanKidneyDataset._preprocess_uncached` su sirayi izler:

1. `.npy` hacmi yuklenir ve `float32` olarak ele alinir.
2. `nan_strategy` uygulanir.
3. `use_bbox_crop=True` ise metadata'daki bbox ve `bbox_margin` ile crop yapilir.
4. `canonicalize_right=True` ve kayit `side == "R"` ise hacim `right_flip_axis` ekseninde cevrilir.
5. `pad_to_cube_input=True` ise hacim en buyuk kenara gore kupe pad edilir.
6. Hacim `target_shape` boyutuna trilinear interpolation ile yeniden orneklenir.
7. Degerler `[0, 1]` araligina kirpilir.

Varsayilan on isleme ayarlari `Utils.config.DataConfig` icindedir:

| Ayar | Varsayilan | Aciklama |
|---|---:|---|
| `target_shape` | `(64, 64, 64)` | Model giris hacmi |
| `use_bbox_crop` | `True` | Foreground bbox crop kullanimi |
| `bbox_margin` | `8` | Bbox etrafina eklenecek voxel marji |
| `pad_to_cube` | `True` | Resize oncesi kupe pad |
| `canonicalize_right` | `False` | Sag ROI'leri ortak yonelime cevirmek icin flip |
| `right_flip_axis` | `0` | Sag ROI flip ekseni |
| `nan_strategy` | `none` | NaN davranisi |
| `cache_mode` | `none` | On isleme cache modu |

## NaN Stratejileri

`apply_nan_strategy` su stratejileri destekler:

| Strateji | Davranis |
|---|---|
| `none` | Hacme dokunmaz. |
| `fill_zero` | NaN degerleri `0.0` yapar. |
| `fill_constant` | NaN degerleri `nan_fill_value` ile doldurur. |
| `fill_mean` | NaN degerleri hacimdeki gecerli voxel ortalamasiyla doldurur. |
| `fill_median` | NaN degerleri hacimdeki gecerli voxel medyaniyla doldurur. |
| `drop_record` | `Model.engine.build_dataloaders` icinde NaN iceren kayitlari splitlerden dusurur; dataset icinde hacme uygulanmaz. |

Egitim CLI tarafindan ornek:

```bash
python -m Model.train \
  --nan-strategy fill_zero \
  --output-dir outputs/fill_zero_run
```

## Cache Modlari

Cache yalnizca deterministik on isleme sonucunu saklar; train augmentasyonlari cache'e yazilmaz.

| Mod | Davranis | Ne zaman kullanilir |
|---|---|---|
| `none` | Her erisimde `.npy` yukler ve on isler. | Kucuk denemeler ve debug |
| `memory` | Tensorleri process belleginde tutar. `__getitem__` clone dondurur. | Kucuk veri veya `num_workers=0` |
| `disk` | On islenmis tensorleri `cache_dir/*.npy` olarak yazar. | Uzun egitimler, Optuna, `num_workers > 0` |

Disk cache anahtari; ROI kimligi, hacim yolu/stat bilgisi, hedef boyut, bbox/canonicalization/NaN ayarlari ve on isleme kodunun fingerprint'i ile uretilir. On isleme kodu veya veri dosyasi degistiginde yeni cache anahtari olusur.

Ornek:

```bash
python -m Model.train \
  --cache-mode disk \
  --cache-dir outputs/preprocessed_cache \
  --num-workers 4
```

## Augmentasyonlar

`transforms.py`, train split icin tensor uzerinde calisan augmentasyonlari tanimlar. Validation ve test datasetlerinde `transform=None` kullanilir.

| Sinif/Fonksiyon | Aciklama |
|---|---|
| `Compose3D` | Birden fazla transformu sirayla uygular |
| `RandomFlip3D` | Verilen spatial eksenlerde olasilikli flip yapar |
| `RandomAffine3D` | 3B rotasyon, olcekleme ve translasyon uygular |
| `RandomMorphology3D` | Olasilikli dilation veya erosion benzeri 3B max-pool islemi uygular |
| `build_train_augmentations(config)` | `AugmentationConfig` ayarlarindan train transform zinciri kurar |

Transformlar `1 x D x H x W` tensor bekler. `RandomFlip3D.axes` degerleri spatial eksenleri temsil eder: `0 -> D`, `1 -> H`, `2 -> W`.

Varsayilan augmentasyon ayarlari:

| Ayar | Varsayilan |
|---|---:|
| `enabled` | `True` |
| `flip_probability` | `0.5` |
| `flip_axes` | `(1, 2)` |
| `affine_probability` | `0.6` |
| `rotation_degrees` | `10.0` |
| `translation_fraction` | `0.05` |
| `scale_min` / `scale_max` | `0.9` / `1.1` |
| `morphology_probability` | `0.1` |

Egitimde augmentasyonlari kapatmak icin:

```bash
python -m Model.train --disable-augmentations
```

## Model Pipeline ile Baglanti

`Model.engine.build_dataloaders` su adimlari uygular:

1. `load_records` ile metadata okur veya gerekirse uretir.
2. `split_records` ile `ZS-train`, `ZS-dev`, `ZS-test` splitlerini kurar.
3. `nan_strategy == "drop_record"` ise NaN iceren kayitlari her splitten dusurur.
4. `build_train_augmentations` ile sadece train transformunu kurar.
5. Train/val/test icin ortak `AlanKidneyDataset` ayarlariyla datasetleri olusturur.
6. `use_weighted_sampler=True` ise train split icin inverse-frequency sampler kullanir.
7. PyTorch `DataLoader` nesnelerini dondurur.

Model tarafinda tabular ozellikler de bu batch sozlugunden turetilir:

| Ozellik | Kaynak | Not |
|---|---|---|
| `log_voxel_count_z` | `voxel_count` | Yalnizca train split ortalama/std ile normalize edilir |
| `side_is_left` | `side` | `L` icin `1.0`, digerleri icin `0.0` |

Bu nedenle `metadata.csv` icindeki `voxel_count` ve `side` alanlari sadece dokumantasyon degil, model girdisinin bir parcasi olabilir.

## Hizli Kontrol Komutlari

Metadata dosyalarini yeniden uretmek:

```bash
python -m Preprocessing.analyze_dataset
```

Dataset testlerini calistirmak:

```bash
python -m pytest Tests/test_dataset.py
```

Sentetik veri uzerinde temel on isleme ve egitim duman testini calistirmak:

```bash
python -m pytest Tests/test_smoke.py
```

## Dikkat Edilecek Noktalar

- `metadata.csv` bbox bilgisi hacimlerin mevcut haline gore uretilir. `.npy` dosyalari degisirse metadata yeniden uretilmelidir.
- `drop_record` stratejisi dataset sinifinda degil, `Model.engine.build_dataloaders` icinde uygulanir.
- Disk cache, on islenmis tensorleri saklar; cache klasoru buyuyebilir.
- Augmentasyonlar rastgeledir ve sadece train split icin kullanilir.
- Hasta seviyesinde split kontrolu metadata uretiminde yapilir; ayni hastanin sol ve sag ROI kayitlari farkli splitlerdeyse islem hata verir.
