# -*- coding: utf-8 -*-
"""AudioWhisper_Train_v00001.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1P4ClLkPmfsaKn2tBbRp0nVjGMRKR-EWz

# [Whisper](https://github.com/openai/whisper) Fine Tuning demo

MIT License

# lib install
"""

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# ! pip install git+https://github.com/openai/whisper.git
# ! pip install jiwer

# Commented out IPython magic to ensure Python compatibility.
# # Lib for train
# %%capture
# ! pip install pyopenjtalk==0.3.0
# ! pip install pytorch-lightning==1.7.7
# ! pip install -qqq evaluate==0.2.2

"""# Import"""


from pathlib import Path

import os
import numpy as np

try:
    import tensorflow  # required in Colab to avoid protobuf compatibility issues
except ImportError:
    pass

import torch
from torch import nn
import pandas as pd
import whisper
import torchaudio
import torchaudio.transforms as at

from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from tqdm.notebook import tqdm
import evaluate

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup
)

DATASET_DIR = "./dataset"
SAMPLE_RATE = 16000
BATCH_SIZE = 2
TRAIN_RATE = 0.8

AUDIO_MAX_LENGTH = 480000
TEXT_MAX_LENGTH = 120
SEED = 3407
DEVICE = "gpu" if torch.cuda.is_available() else "cpu"
seed_everything(SEED, workers=True)


def run():
    dataset_dir = Path(DATASET_DIR)
    transcript_file_path = dataset_dir / "text.txt"
    with open(transcript_file_path, 'r') as file:
        transctipts = file.readlines()
    transctipts = [t.replace("\n", "").split("\t") for t in transctipts]

    # [(audio_id, str(audio_path), text)]
    train_data = [(t[0].replace(".wav", ""), str(dataset_dir / t[0]), t[1]) for t in transctipts]

    train_num = int(len(train_data) * TRAIN_RATE)
    train_audio_transcript_pair_list, eval_audio_transcript_pair_list = train_data[:train_num], train_data[train_num:]
    print("TRAIN AUDIO DATASET NUM: ", len(train_audio_transcript_pair_list))
    print("EVAL AUDIO DATASET NUM: ", len(eval_audio_transcript_pair_list))

    woptions = whisper.DecodingOptions(language="en", without_timestamps=True)
    wmodel = whisper.load_model("base")
    wtokenizer = whisper.tokenizer.get_tokenizer(True, language="en", task=woptions.task)


    """# Confirm Dataloading"""

    dataset = JvsSpeechDataset(eval_audio_transcript_pair_list, wtokenizer, SAMPLE_RATE)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=WhisperDataCollatorWhithPadding())

    for b in loader:
        print(b["labels"].shape)
        print(b["input_ids"].shape)
        print(b["dec_input_ids"].shape)

        for token, dec in zip(b["labels"], b["dec_input_ids"]):
            token[token == -100] = wtokenizer.eot
            text = wtokenizer.decode(token)
            print(text)

            dec[dec == -100] = wtokenizer.eot
            text = wtokenizer.decode(dec)
            print(text)
        break

    with torch.no_grad():
        input_ids = b["input_ids"]
        labels = b["labels"].long()
        dec_input_ids = b["dec_input_ids"].long()

        audio_features = wmodel.encoder(input_ids.cuda())
        print(dec_input_ids)
        print(input_ids.shape, dec_input_ids.shape, audio_features.shape)
        print(audio_features.shape)
        print()
    out = wmodel.decoder(dec_input_ids.cuda(), audio_features)

    print(out.shape)
    print(out.view(-1, out.size(-1)).shape)
    print(b["labels"].view(-1).shape)

    tokens = torch.argmax(out, dim=2)
    for token in tokens:
        token[token == -100] = wtokenizer.eot
        # text = wtokenizer.decode(token, skip_special_tokens=True)
        text = wtokenizer.decode(token)
        print(text)

    """# Trainer"""

    """# Observation


    """

    # Commented out IPython magic to ensure Python compatibility.
    # %load_ext tensorboard

    # Commented out IPython magic to ensure Python compatibility.
    # %tensorboard --logdir ./train/logs

    """# main code"""

    log_output_dir = "./train/logs"
    check_output_dir = "./train/artifacts"

    train_name = "whisper"
    train_id = "00001"

    model_name = "base"
    lang = "en"

    cfg = Config()

    Path(log_output_dir).mkdir(exist_ok=True, parents=True)
    Path(check_output_dir).mkdir(exist_ok=True, parents=True)

    tflogger = TensorBoardLogger(
        save_dir=log_output_dir,
        name=train_name,
        version=train_id
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{check_output_dir}/checkpoint",
        filename="checkpoint-{epoch:04d}",
        save_top_k=-1  # all model save
    )

    callback_list = [checkpoint_callback, LearningRateMonitor(logging_interval="epoch")]
    model = WhisperModelModule(cfg, model_name, lang, train_audio_transcript_pair_list, eval_audio_transcript_pair_list)

    trainer = Trainer(
        precision=16,
        accelerator=DEVICE,
        max_epochs=cfg.num_train_epochs,
        accumulate_grad_batches=cfg.gradient_accumulation_steps,
        logger=tflogger,
        callbacks=callback_list
    )

    trainer.fit(model)

    """# load weight and inference"""

    checkpoint_path = "./train/artifacts/checkpoint/checkpoint-epoch=0007.ckpt"

    state_dict = torch.load(checkpoint_path)
    print(state_dict.keys())
    state_dict = state_dict['state_dict']

    whisper_model = WhisperModelModule(cfg)
    whisper_model.load_state_dict(state_dict)

    woptions = whisper.DecodingOptions(language="en", without_timestamps=True)
    dataset = JvsSpeechDataset(eval_audio_transcript_pair_list, wtokenizer, SAMPLE_RATE)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=WhisperDataCollatorWhithPadding())

    refs = []
    res = []
    for b in tqdm(loader):
        input_ids = b["input_ids"].half().cuda()
        labels = b["labels"].long().cuda()
        with torch.no_grad():
            # audio_features = whisper_model.model.encoder(input_ids)
            # out = whisper_model.model.decoder(enc_input_ids, audio_features)
            results = whisper_model.model.decode(input_ids, woptions)
            for r in results:
                res.append(r.text)

            for l in labels:
                l[l == -100] = wtokenizer.eot
                # ref = wtokenizer.decode(l, skip_special_tokens=True)
                ref = wtokenizer.decode(l)
                refs.append(ref)

    cer_metrics = evaluate.load("cer")
    cer_metrics.compute(references=refs, predictions=res)

    for k, v in zip(refs, res):
        print("-" * 10)
        print(k)
        print(v)


def load_wave(wave_path, sample_rate: int=16000) -> torch.Tensor:
    waveform, sr = torchaudio.load(wave_path, normalize=True)
    if sample_rate != sr:
        waveform = at.Resample(sr, sample_rate)(waveform)
    return waveform


def get_audio_file_list(transcripts_path_list, text_max_length=120, audio_max_sample_length=480000, sample_rate=16000):
    audio_transcript_pair_list = []
    for transcripts_path in tqdm(transcripts_path_list): 
        audio_dir = transcripts_path.parent / "wav24kHz16bit"
        if not audio_dir.exists():
            print(f"{audio_dir}は存在しません。")
            continue

        # 翻訳テキストからAudioIdとテキストを取得
        with open(transcripts_path, "r") as f:
            text_list = f.readlines()
        for text in text_list:
            audio_id, text = text.replace("\n", "").split(":")
            #print(audio_id, text)

            audio_path = audio_dir / f"{audio_id}.wav"
            if audio_path.exists():
                # データのチェック
                audio = load_wave(audio_path, sample_rate=sample_rate)[0]
                if len(text) > text_max_length or len(audio) > audio_max_sample_length:
                    print(len(text), len(audio))
                    continue
                audio_transcript_pair_list.append((audio_id, str(audio_path), text))
    return audio_transcript_pair_list


class JvsSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, audio_info_list, tokenizer, sample_rate) -> None:
        super().__init__()

        self.audio_info_list = audio_info_list
        self.sample_rate = sample_rate
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.audio_info_list)
    
    def __getitem__(self, id):
        audio_id, audio_path, text = self.audio_info_list[id]

        # audio
        audio = load_wave(audio_path, sample_rate=self.sample_rate)
        audio = whisper.pad_or_trim(audio.flatten())
        mel = whisper.log_mel_spectrogram(audio)

        text = [*self.tokenizer.sot_sequence_including_notimestamps] + self.tokenizer.encode(text)
        labels = text[1:] + [self.tokenizer.eot]

        return {
            "input_ids": mel,
            "labels": labels,
            "dec_input_ids": text
        }


class WhisperDataCollatorWhithPadding:
    def __call__(sefl, features):
        input_ids, labels, dec_input_ids = [], [], []
        for f in features:
            input_ids.append(f["input_ids"])
            labels.append(f["labels"])
            dec_input_ids.append(f["dec_input_ids"])

        input_ids = torch.concat([input_id[None, :] for input_id in input_ids])
        
        label_lengths = [len(lab) for lab in labels]
        dec_input_ids_length = [len(e) for e in dec_input_ids]
        max_label_len = max(label_lengths+dec_input_ids_length)

        labels = [np.pad(lab, (0, max_label_len - lab_len), 'constant', constant_values=-100) for lab, lab_len in zip(labels, label_lengths)]
        dec_input_ids = [np.pad(e, (0, max_label_len - e_len), 'constant', constant_values=50257) for e, e_len in zip(dec_input_ids, dec_input_ids_length)] # 50257 is eot token id

        batch = {
            "labels": labels,
            "dec_input_ids": dec_input_ids
        }

        batch = {k: torch.tensor(np.array(v), requires_grad=False) for k, v in batch.items()}
        batch["input_ids"] = input_ids

        return batch


class Config:
    learning_rate = 0.0005
    weight_decay = 0.01
    adam_epsilon = 1e-8
    warmup_steps = 2
    batch_size = 8
    num_worker = 2
    num_train_epochs = 10
    gradient_accumulation_steps = 1
    sample_rate = SAMPLE_RATE


class WhisperModelModule(LightningModule):
    def __init__(self, cfg:Config, model_name="base", lang="en", train_dataset=[], eval_dataset=[]) -> None:
        super().__init__()
        self.options = whisper.DecodingOptions(language=lang, without_timestamps=True)
        self.model = whisper.load_model(model_name)
        self.tokenizer = whisper.tokenizer.get_tokenizer(True, language="en", task=self.options.task)

        # only decoder training
        for p in self.model.encoder.parameters():
            p.requires_grad = False
        
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.metrics_wer = evaluate.load("wer")
        self.metrics_cer = evaluate.load("cer")

        self.cfg = cfg
        self.__train_dataset = train_dataset
        self.__eval_dataset = eval_dataset
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_id):
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()

        with torch.no_grad():
            audio_features = self.model.encoder(input_ids)

        out = self.model.decoder(dec_input_ids, audio_features)
        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))
        self.log("train/loss", loss, on_step=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_id):
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()

        audio_features = self.model.encoder(input_ids)
        out = self.model.decoder(dec_input_ids, audio_features)

        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))

        out[out == -100] = self.tokenizer.eot
        labels[labels == -100] = self.tokenizer.eot

        o_list, l_list = [], []
        for o, l in zip(out, labels):
            o = torch.argmax(o, dim=1)
            # o_list.append(self.tokenizer.decode(o, skip_special_tokens=True))
            # l_list.append(self.tokenizer.decode(l, skip_special_tokens=True))
            o_list.append(self.tokenizer.decode(o))
            l_list.append(self.tokenizer.decode(l))
        cer = self.metrics_cer.compute(references=l_list, predictions=o_list)
        wer = self.metrics_wer.compute(references=l_list, predictions=o_list)

        self.log("val/loss", loss, on_step=True, prog_bar=True, logger=True)
        self.log("val/cer", cer, on_step=True, prog_bar=True, logger=True)
        self.log("val/wer", wer, on_step=True, prog_bar=True, logger=True)

        return {
            "cer": cer,
            "wer": wer,
            "loss": loss
        }

    def configure_optimizers(self):
        """オプティマイザーとスケジューラーを作成する"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() 
                            if not any(nd in n for nd in no_decay)],
                "weight_decay": self.cfg.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() 
                            if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, 
                          lr=self.cfg.learning_rate, 
                          eps=self.cfg.adam_epsilon)
        self.optimizer = optimizer

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.cfg.warmup_steps, 
            num_training_steps=self.t_total
        )
        self.scheduler = scheduler

        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]
    
    def setup(self, stage=None):
        """初期設定（データセットの読み込み）"""

        if stage == 'fit' or stage is None:
            self.t_total = (
                (len(self.__train_dataset) // (self.cfg.batch_size))
                // self.cfg.gradient_accumulation_steps
                * float(self.cfg.num_train_epochs)
            )
    
    def train_dataloader(self):
        """訓練データローダーを作成する"""
        dataset = JvsSpeechDataset(self.__train_dataset, self.tokenizer, self.cfg.sample_rate)
        return torch.utils.data.DataLoader(dataset, 
                          batch_size=self.cfg.batch_size, 
                          drop_last=True, shuffle=True, num_workers=self.cfg.num_worker,
                          collate_fn=WhisperDataCollatorWhithPadding()
                          )

    def val_dataloader(self):
        """バリデーションデータローダーを作成する"""
        dataset = JvsSpeechDataset(self.__eval_dataset, self.tokenizer, self.cfg.sample_rate)
        return torch.utils.data.DataLoader(dataset, 
                          batch_size=self.cfg.batch_size, 
                          num_workers=self.cfg.num_worker,
                          collate_fn=WhisperDataCollatorWhithPadding()
                          )



if __name__ == '__main__':
    run()