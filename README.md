This repository hosts the MushR Mask R CNN model for maturity
detection of Oyster mushrooms.

# Dependencies

To run inference, please follow the official instructions to [install
detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).

# Annotations

1. Train: [MushR_Dataset_V1.1_2023_04_04-COCO.json](MushR_Dataset_V1.1_2023_04_04-COCO.json)
2. Test: [MushR_Dataset_2023_06_29_TEST-COCO.json](MushR_Dataset_2023_06_29_TEST-COCO.json)

# Training 

All training related configuration can be found in [train_2023_04_04/](train_2023_04_04/).

We used the [train.sh](train.sh) script to train the model.

To continue training from the last checkpoint, use: [continue_training.sh](continue_training.sh)

## NOTE

The model weights of our trained model can be found in <https://github.com/ETCE-LAB/mushr-mask-r-cnn/releases>


# Inference

The scripts, [inference_images.sh](inference_images.sh) and [inference_images-single.sh](inference_images-single.sh)
can be used to run inference.

Please check [inference_2023_04_20](inference_2023_04_20) for examples of expected results.

# Evaluation

To evaluate the model, please use the [eval.sh](eval.sh) script.

Our evaluation results of the trained model are given in [eval_2023_07_19](eval_2023_07_19).

## Folder-based classification

### Windowsでグループメンバーが実行する方法

このプロジェクトのフォルダーをメンバーのPCにコピーします。各PCで最初に
Python 3.10または3.11をインストールし、PowerShellでプロジェクトフォルダーを
開いて次を一度だけ実行してください。

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup_windows.ps1
```

以降は、プロジェクトフォルダーのターミナルで画像を1枚指定して実行します。
明るい画像か暗い画像かを自分で判断する必要はありません。`classify.bat` が
画像の明るさを自動判定し、暗い場合だけ補正してから、同じ方法で判定します。
実行方法は常に次の1種類です。

```powershell
.\classify.bat "C:\Users\name\Pictures\mushroom.jpg"
```

画像は必ず1枚だけ指定してください。`--enhance` や `--auto-enhance` を
メンバーが使い分ける必要はありません。

パスに空白が含まれる場合も、必ず `" "` で囲んでください。学習済みモデル
`classifier_output/best_model.pth` はメンバー間で共有してください。モデルを
再学習する必要はありません。

If images are organized into class folders rather than COCO instance annotations,
the classifier can be trained with `train_classifier.py`. The expected layout is:

```text
ANNOTATION_BATCHES/
  hiratake_dateset/
    成長済み/
    未成熟/
  test_datelist/
    成長済み/
    未成熟/
```

From the project directory, run:

```bash
python train_classifier.py --device cpu --epochs 10
python classify_image.py --input path/to/image.jpg
```

The best checkpoint and training history are written to `classifier_output/`.
The classification command prints both the predicted class and its confidence:

```text
path/to/image.jpg [明るさ補正なし]：成長済み（信頼度: 92.43%）
```

暗い画像の場合は、結果に `[明るさ補正あり]` と表示されます。

### Optional brightness correction

The classifier includes a built-in brightness correction for dark images. It
does not require another model, a download, or additional training.

Run classification without correction:

```bash
python classify_image.py --input path/to/image.jpg
```

Run classification with brightness correction:

```bash
python classify_image.py --input path/to/image.jpg --enhance
```

To correct only images detected as dark, use automatic mode. Images with an
average brightness below 40% are corrected; other images are left unchanged:

```bash
python classify_image.py --input path/to/image.jpg --auto-enhance
```

The threshold can be changed, for example to 30%:

```bash
python classify_image.py --input path/to/image.jpg --auto-enhance --dark-threshold 0.30
```

The same image can be classified both ways to compare the results. Correction
does not retrain the classifier; its effect should be evaluated using images
with known correct labels.

To compare both methods on a labeled test folder, run:

```bash
python evaluate_brightness.py --device cpu
```

By default, this evaluates `ANNOTATION_BATCHES/test_datelist/` and writes
per-image results to `classifier_output/brightness_comparison.csv`. The output
includes accuracy, average confidence, and the number of predictions changed by
brightness correction. Accuracy, not confidence alone, should be used to decide
whether correction improves the classifier.
