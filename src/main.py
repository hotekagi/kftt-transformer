from os.path import join
from typing import List
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from model.Transformer import Transformer
from Trainer import Trainer
from util.bleu import BleuScore
from util.dataset import KfttDataset
from util.text import get_vocab, tensor_to_text, text_to_tensor

# debug
# torch.autograd.set_detect_anomaly(True)

TRAIN_SRC_CORPUS_PATH = "../corpus/kftt-data-1.0/data/tok/kyoto-train.en"
TRAIN_TGT_CORPUS_PATH = "../corpus/kftt-data-1.0/data/tok/kyoto-train.ja"
VAL_SRC_CORPUS_PATH = "../corpus/kftt-data-1.0/data/tok/kyoto-dev.en"
VAL_TGT_CORPUS_PATH = "../corpus/kftt-data-1.0/data/tok/kyoto-dev.ja"
TEST_SRC_CORPUS_PATH = "../corpus/kftt-data-1.0/data/tok/kyoto-test.en"
TEST_TGT_CORPUS_PATH = "../corpus/kftt-data-1.0/data/tok/kyoto-test.ja"


# create train data
src_vocab = get_vocab(TRAIN_SRC_CORPUS_PATH, vocab_size=20000)
tgt_vocab = get_vocab(TRAIN_TGT_CORPUS_PATH, vocab_size=20000)

src_vocab_size = len(src_vocab)
tgt_vocab_size = len(tgt_vocab)


max_len = 24
d_model = 128
heads_num = 4
d_ff = 256
N = 3
dropout_rate = 0.1
layer_norm_eps = 1e-8
pad_idx = 0
batch_size = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epoch = 3


net = Transformer(
    src_vocab_size=src_vocab_size,
    tgt_vocab_size=tgt_vocab_size,
    max_len=max_len,
    d_model=d_model,
    heads_num=heads_num,
    d_ff=d_ff,
    N=N,
    dropout_rate=dropout_rate,
    layer_norm_eps=layer_norm_eps,
    pad_idx=pad_idx,
    device=device,
)


def src_text_to_tensor(text: str, max_len: int) -> torch.Tensor:
    return text_to_tensor(text, src_vocab, max_len, eos=False, bos=False)


def src_tensor_to_text(tensor: torch.Tensor) -> str:
    return tensor_to_text(tensor, src_vocab)


def tgt_text_to_tensor(text: str, max_len: int) -> torch.Tensor:
    return text_to_tensor(text, tgt_vocab, max_len)


def tgt_tensor_to_text(tensor: torch.Tensor) -> str:
    return tensor_to_text(tensor, tgt_vocab)


train_dataset = KfttDataset(
    TRAIN_SRC_CORPUS_PATH,
    TRAIN_TGT_CORPUS_PATH,
    max_len,
    src_text_to_tensor,
    tgt_text_to_tensor,
)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = KfttDataset(
    VAL_SRC_CORPUS_PATH,
    VAL_TGT_CORPUS_PATH,
    max_len,
    src_text_to_tensor,
    tgt_text_to_tensor,
)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

test_dataset = KfttDataset(
    TEST_SRC_CORPUS_PATH,
    TEST_TGT_CORPUS_PATH,
    max_len,
    src_text_to_tensor,
    tgt_text_to_tensor,
)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


trainer = Trainer(
    net,
    optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), amsgrad=True),
    nn.CrossEntropyLoss(),
    BleuScore(tgt_vocab),
    device,
)
train_losses: List[float] = []
train_bleu_scores: List[float] = []
val_losses: List[float] = []

for i in range(epoch):
    print(f"epoch: {i + 1} \n")
    (
        train_losses_per_epoch,
        train_bleu_scores_per_epoch,
        val_losses_per_epoch,
        val_bleu_scores_per_epoch,
    ) = trainer.fit(train_loader, val_loader, print_log=True)

    train_losses.extend(train_losses_per_epoch)
    train_bleu_scores.extend(train_bleu_scores_per_epoch)
    val_losses.extend(val_losses_per_epoch)
    torch.save(trainer.net, join("../result", f"epoch_{i}.pt"))


test_losses, test_bleu_scores = trainer.test(test_loader)

fig = plt.figure(figsize=(24, 8))
train_loss_ax = fig.add_subplot(1, 2, 1)
val_loss_ax = fig.add_subplot(1, 2, 2)

train_loss_ax.plot(list(range(len(train_losses))), train_losses, label="train loss")
val_loss_ax.plot(
    list(range(len(train_bleu_scores))),
    train_bleu_scores,
    label="val loss",
)
train_loss_ax.legend()
val_loss_ax.legend()
plt.savefig(join("../result", f"loss{20230302}.png"))
