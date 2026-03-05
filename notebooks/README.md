# Botanix — Plant Leaf Disease Classification
## Model Eğitim Notebook'ları

**Dataset:** [Plant Leaf Disease Classification Dataset](https://www.kaggle.com/datasets/mertcangelbal/plant-leaf-disease-classification-dataset)  
**Ana Dizin:** `/Users/mertcangelbal/Documents/botanix/`  
**GPU:** RTX 5090 (Vast.ai)

---

## Proje Dizin Yapısı

```
botanix/
├── Archive.zip                        # Ham dataset arşivi (Kaggle'dan indirilmiş)
├── data/                              # Hazırlanmış dataset
│   ├── train/                         # 273.155 görüntü, 105 sınıf
│   ├── val/                           # 50.729 görüntü, 105 sınıf
│   └── test/                          # 50.627 görüntü, 105 sınıf
└── notebooks/                         # Model eğitim notebook'ları
    ├── README.md                               # Bu dosya
    ├── model1_cnn_baseline.ipynb               # CNN Baseline (Transfer Learning)
    ├── model2_vit.ipynb                        # Vision Transformer (ViT-B/16)
    ├── model3_vit_segmentation.ipynb           # ViT + Segmentation Head
    ├── model4_swin_transformer.ipynb           # Swin Transformer (Swin-B)
    ├── model5_swin_segmentation.ipynb          # Swin Transformer + Segmentation Head
    ├── botanix_cnn_scratch.ipynb               # Botanix — Sıfırdan CNN (PlantCLEF)
    ├── botanix_cnn_scratch_segmentation.ipynb  # Botanix — Sıfırdan CNN + Seg (PlantCLEF)
    └── model_comparison.ipynb                  # Tüm modellerin karşılaştırması
```

---

## Dataset Bilgileri

| Özellik | Değer |
|---------|-------|
| Toplam sınıf | 105 |
| Bitki türü | 29 |
| Görüntü boyutu | 256×256 px (orijinal) → 224×224 (model girişi) |
| Renk modu | RGB |
| Format | JPEG |
| Train | 273.155 görüntü |
| Validation | 50.729 görüntü |
| Test | 50.627 görüntü |
| Toplam | ~374.511 görüntü |
| Disk kullanımı | ~10 GB |

---

## Model Listesi

| # | Dosya | Model | Pretrained | Segmentation | Açıklama |
|---|-------|-------|-----------|-------------|---------|
| 1 | `model1_cnn_baseline.ipynb` | CNN Baseline | ✅ ImageNet | ❌ | Transfer learning ile CNN — referans model |
| 2 | `model2_vit.ipynb` | ViT-B/16 | ✅ ImageNet-21k | ❌ | Vision Transformer, 196 patch (14×14) |
| 3 | `model3_vit_segmentation.ipynb` | ViT + Seg | ✅ ImageNet-21k | ✅ | ViT + patch-level saliency attention |
| 4 | `model4_swin_transformer.ipynb` | Swin-B | ✅ ImageNet-22k | ❌ | Shifted Window Transformer |
| 5 | `model5_swin_segmentation.ipynb` | Swin + Seg | ✅ ImageNet-22k | ✅ | Swin + spatial attention map |
| B1 | `botanix_cnn_scratch.ipynb` | Botanix CNN | ❌ | ❌ | PlantCLEF için sıfırdan CNN |
| B2 | `botanix_cnn_scratch_segmentation.ipynb` | Botanix CNN + Seg | ❌ | ✅ | PlantCLEF için sıfırdan CNN + seg |
| — | `model_comparison.ipynb` | Karşılaştırma | — | — | Tüm sonuçları karşılaştırır, LaTeX tablo üretir |

---

## Eğitim Parametreleri

| Parametre | Değer |
|-----------|-------|
| Image Size | 224 × 224 |
| Batch Size | 32 |
| Optimizer | AdamW |
| Learning Rate | 3e-4 |
| Weight Decay | 1e-4 |
| Epoch | 50 |
| Loss Function | CrossEntropyLoss (weighted) |
| LR Scheduler | CosineAnnealingLR (eta_min=1e-6) |
| Grad Clip | max_norm=1.0 |

> Model 6 ve 7 (sıfırdan eğitim) için ek olarak **Warmup Scheduler** (5 epoch) uygulanır.

---

## Veri Augmentation

### Train (tüm modeller)
| Yöntem | Parametre |
|--------|----------|
| RandomHorizontalFlip | p=0.5 |
| RandomVerticalFlip | p=0.2 |
| RandomRotation | ±15° |
| ColorJitter | brightness/contrast/saturation=0.3, hue=0.1 |
| RandomCrop | padding=8 |

### Train (Scratch modeller — Model 6 & 7)
Yukarıdakilere ek:
| Yöntem | Parametre |
|--------|----------|
| RandomRotation | ±20° (artırıldı) |
| ColorJitter | brightness/contrast/saturation=0.4, hue=0.15 (artırıldı) |
| RandomCrop | padding=12 (artırıldı) |
| RandomGrayscale | p=0.05 |

### Validation / Test
Sadece Resize + Normalize uygulanır.

### Normalizasyon (tüm modeller)
```
mean = [0.485, 0.456, 0.406]   # ImageNet ortalaması
std  = [0.229, 0.224, 0.225]
```

---

## Model Mimarileri

### Model 1 — CNN Baseline
```
Input (224×224×3)
→ ConvBlock(3→64)   + MaxPool  → 112×112
→ ConvBlock(64→128) + MaxPool  → 56×56
→ ConvBlock(128→256)+ MaxPool  → 28×28
→ ConvBlock(256→512)+ MaxPool  → 14×14
→ ConvBlock(512→512)+ MaxPool  → 7×7
→ Global Average Pooling
→ FC(512) → Dropout(0.5) → FC(105)
→ Softmax
```
Her ConvBlock: Conv3×3 → BN → ReLU → Conv3×3 → BN → ReLU

---

### Model 2 — Vision Transformer (ViT-B/16)
```
Input (224×224×3)
→ Patch Embedding  (16×16 patch → 196 token, dim=768)
→ Positional Encoding
→ 12× Transformer Encoder Block
   └─ Multi-Head Self-Attention (12 head)
   └─ MLP Block (768→3072→768)
→ CLS Token
→ Classification Head (768→105)
```
Kaynak: `timm.create_model("vit_base_patch16_224", pretrained=True)`

---

### Model 3 — ViT + Segmentation
```
Input → ViT Backbone (global_pool="" ile tüm token'lar alınır)
                ↓
        CLS Token [B,768]    Patch Tokens [B,196,768]
                              ↓
                    Segmentation Head
                    Linear(768→256)→GELU→Linear(256→1)
                              ↓
                    Softmax Attention [B,196]
                              ↓
                    Weighted Patch Feature [B,768]
                ↓
        CLS + Weighted Feature (toplama)
                ↓
        LayerNorm → Linear(768→105)
```
Entropy regularization loss ile seg head düzenlenir.

---

### Model 4 — Swin Transformer (Swin-B)
```
Input (224×224×3)
→ Patch Partition (4×4 → 56×56×96)
→ Stage 1: Window Attention (window=7) → 56×56×96
→ Stage 2: Shifted Window + Patch Merging → 28×28×192
→ Stage 3: Shifted Window + Patch Merging → 14×14×384
→ Stage 4: Shifted Window + Patch Merging → 7×7×768
→ Global Average Pooling
→ Linear(768→105)
```
Kaynak: `timm.create_model("swin_base_patch4_window7_224", pretrained=True)`

---

### Model 5 — Swin + Segmentation
```
Input → Swin Backbone (son stage: [B, 49, 1024])
                ↓
    Global Feature [B,1024]     Patch Features [B,49,1024]
                                      ↓
                           Segmentation Head
                           Linear(1024→256)→GELU→Linear(256→1)
                                      ↓
                           Softmax Attention [B,49]
                                      ↓
                           Weighted Feature [B,1024]
                ↓
    Global + Weighted Feature (toplama)
                ↓
    LayerNorm → Linear(1024→105)
```

---

### Botanix — CNN Sıfırdan (PlantCLEF)
```
Input (224×224×3)
→ Stem: Conv7×7(stride=2) → BN → ReLU → MaxPool  → 56×56×64
→ Layer1: Conv3×3(stride=2) + ResidualBlock(128)  → 28×28×128
→ Layer2: Conv3×3(stride=2) + ResidualBlock(256)  → 14×14×256
→ Layer3: Conv3×3(stride=2) + ResidualBlock(512)  → 7×7×512
→ Global Average Pooling
→ FC(512→1024) → ReLU → Dropout(0.5) → FC(1024→105)
```
Ağırlık başlatma: Kaiming Normal (Conv), Xavier (Linear)

---

### Botanix — CNN Sıfırdan + Segmentation (PlantCLEF)
```
Input → CNN Encoder (Model 6 ile aynı)
                ↓
    Feature Map [B, 512, 7, 7]
         ↙                    ↘
Global Avg Pool          Segmentation Head
[B, 512]                 Conv1×1(512→128)→BN→ReLU→Conv1×1(128→1)
                         Spatial Attention Map [B, 1, 7, 7]
                                ↓
                         Weighted Feature [B, 512]
         ↘                    ↙
    Global + Weighted Feature (toplama)
                ↓
    FC(512→1024) → ReLU → Dropout(0.5) → FC(1024→105)
```

---

## Segmentation Yaklaşımı Hakkında

Segmentation head, **label olmadan** (self-supervised) çalışır.  
Pixel/region maskesi gerekmez — yalnızca sınıflandırma etiketleri kullanılır.

**Çalışma prensibi:**
- Her patch/spatial konuma bir "önem skoru" atanır
- Softmax ile normalize edilerek attention map oluşturulur
- Bu map ile feature ağırlıklandırılır → modelin hastalıklı bölgeye odaklanması sağlanır
- **Entropy regularization loss** ile attention map'in yoğunlaşması (sparsity) teşvik edilir

**Loss fonksiyonu:**
```
Total Loss = CrossEntropyLoss + 0.3 × Seg Entropy Loss
```

---

## Değerlendirme Metrikleri

| Metrik | Açıklama |
|--------|---------|
| Accuracy | Doğru sınıflandırma oranı |
| Precision (weighted) | Sınıf ağırlıklı kesinlik |
| Recall (weighted) | Sınıf ağırlıklı duyarlılık |
| F1 Score (weighted) | Precision ve Recall harmonik ortalaması |
| Confusion Matrix | Sınıf bazlı hata analizi |
| Training Time | Toplam eğitim süresi (dakika) |
| Inference Time | Görüntü başına çıkarım süresi (ms) |

---

## Çıktı Dosyaları

Her notebook çalıştırıldıktan sonra şu dosyalar oluşur:

```
./checkpoints/
├── cnn_baseline_best.pth
├── vit_best.pth
├── vit_seg_best.pth
├── swin_best.pth
├── swin_seg_best.pth
├── botanix_cnn_scratch_best.pth
└── botanix_cnn_scratch_seg_best.pth

./results/
├── cnn_baseline_results.json
├── cnn_baseline_training.png
├── cnn_baseline_confusion_matrix.png
├── cnn_baseline_per_class_f1.png
├── cnn_baseline_top_errors.png
├── cnn_baseline_roc.png
├── vit_results.json
├── vit_training.png
├── vit_confusion_matrix.png
├── vit_per_class_f1.png
├── vit_top_errors.png
├── vit_roc.png
├── vit_seg_results.json
├── vit_seg_attention_maps.png
├── vit_seg_per_class_f1.png
├── vit_seg_top_errors.png
├── vit_seg_roc.png
├── swin_results.json
├── swin_training.png
├── swin_confusion_matrix.png
├── swin_per_class_f1.png
├── swin_top_errors.png
├── swin_roc.png
├── swin_seg_results.json
├── swin_seg_maps.png
├── swin_seg_per_class_f1.png
├── swin_seg_top_errors.png
├── swin_seg_roc.png
├── botanix_cnn_scratch_results.json
├── botanix_cnn_scratch_per_class_f1.png
├── botanix_cnn_scratch_top_errors.png
├── botanix_cnn_scratch_roc.png
├── botanix_cnn_scratch_seg_results.json
├── botanix_cnn_scratch_seg_maps.png
├── botanix_cnn_scratch_seg_per_class_f1.png
├── botanix_cnn_scratch_seg_top_errors.png
├── botanix_cnn_scratch_seg_roc.png
├── model_comparison_metrics.png       # Tüm modeller metrik grafiği
├── model_comparison_efficiency.png    # Accuracy vs Eğitim Süresi
└── model_comparison_radar.png         # Radar grafiği
```

---

## Vast.ai Kurulum Adımları

```bash
# 1. Kaggle API anahtarı
mkdir -p ~/.kaggle
echo '{"username":"YOUR_USERNAME","key":"YOUR_API_KEY"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# 2. Gerekli kütüphaneler (notebook içinde otomatik yüklenir)
pip install kaggle timm scikit-learn seaborn matplotlib

# 3. Dataset indirme (her notebook'un ilk hücresinde de mevcut)
kaggle datasets download -d mertcangelbal/plant-leaf-disease-classification-dataset --unzip -p ../data

# 4. Notebook'ları sırayla çalıştırın
jupyter notebook
```

> **Not:** Notebook'lar `botanix/notebooks/` dizininden çalıştırılmalıdır.  
> Dataset yolu `../data` olarak ayarlanmıştır — `botanix/data/` klasörüne işaret eder.

---

## Çalışmanın Amacı

Bu çalışma **7 farklı derin öğrenme mimarisini** karşılaştırmalı olarak değerlendirmektedir:

1. **CNN vs Transformer** karşılaştırması — bitki hastalık tespitinde hangi yaklaşım daha başarılı?
2. **Segmentation head etkisi** — sınıflandırma performansını artırıyor mu?
3. **Transfer learning etkisi** — pretrained ağırlıklar ne kadar katkı sağlıyor?
4. **Hesaplama verimliliği** — mobil uygulama için hangi model uygun?

---

## Botanix — PlantCLEF Pre-train → Fine-tune Planı

### PlantCLEF 2026 Hakkında

[PlantCLEF 2026](https://www.imageclef.org/PlantCLEF2026) bir **multi-label** yarışmasıdır.  
50×50 cm vejetasyon plot görüntülerinde birden fazla bitki türü tahmin edilmesi beklenir.

| Aşama | Dataset | Görev Tipi | Loss |
|-------|---------|-----------|------|
| **Model 1–5 Karşılaştırma** | Hastalık veri seti (105 sınıf) | Single-label | CrossEntropyLoss |
| **Botanix Faz 1 — Pre-train** | PlantCLEF ~1.4M görüntü, 7.806 tür | Single-label | CrossEntropyLoss |
| **Botanix Faz 2 — Fine-tune** | Hastalık veri seti (105 sınıf) | Single-label | CrossEntropyLoss |
| **Botanix Faz 3 — Plot test** | PlantCLEF vejetasyon plot'ları | **Multi-label** | BCEWithLogitsLoss |

### Çalışma Akışı

1. `model1–5` notebook'ları kendi veri setiyle eğitilir
2. `model_comparison.ipynb` çalıştırılarak en iyi mimari belirlenir
3. `botanix_cnn_scratch.ipynb` — PlantCLEF ile pre-train, ardından kendi veri setiyle fine-tune
4. `botanix_cnn_scratch_segmentation.ipynb` — Aynı akış, segmentation head ile

### PlantCLEF Dataset İndirme (Vast.ai Terminal)

```bash
# ~160GB versiyon (önerilen)
wget -c "https://lab.plantnet.org/LifeCLEF/PlantCLEF2024/single_plant_training_data/PlantCLEF2024singleplanttrainingdata_800_max_side_size.tar" -P ../plantclef/
tar -xf ../plantclef/PlantCLEF2024singleplanttrainingdata_800_max_side_size.tar -C ../plantclef/

# Metadata (sınıf etiketleri)
wget "https://lab.plantnet.org/LifeCLEF/PlantCLEF2024/single_plant_training_data/PlantCLEF2024singleplanttrainingdata.csv" -P ../plantclef/
```

### Multi-label Notu

Kendi hastalık veri setinle eğitim **single-label**dır (her görüntüde 1 hastalık = 1 sınıf).  
Multi-label yalnızca PlantCLEF'in vejetasyon plot test aşamasında gerekir.  
Bu durumda `CrossEntropyLoss` → `BCEWithLogitsLoss` ve `argmax` → `sigmoid + threshold` olarak değiştirilmelidir.
