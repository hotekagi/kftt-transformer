# kftt-transformer

「Attention is all you need」での本家Transformerをゼロから実装し、京都フリー翻訳タスク(KFTT)を学習する。

## 環境
```
Python 3.9.8 (main, Aug  8 2022, 01:58:31)
[GCC 11.2.0] on linux

matplotlib==3.7.0
numpy==1.24.1
torch==1.13.1
torchtext==0.14.1
```

## コーパスの取得
```
cd corpus
wget http://www.phontron.com/kftt/download/kftt-data-1.0.tar.gz
tar -zxvf kftt-data-1.0.tar.gz
```

## 参考文献
- https://github.com/YadaYuki/en_ja_translator_pytorch
- https://zenn.dev/yukiyada/articles/59f3b820c52571
