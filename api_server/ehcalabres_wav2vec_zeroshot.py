import os.path

from transformers import AutoModelForAudioClassification
import urllib
import librosa
from torch import nn
import torch
import json
from sklearn.metrics import f1_score, precision_recall_fscore_support
from argparse import ArgumentParser
from noise_pause_process import speech_only_audio

# MODEL_CHECKPOINT = "/home/siyanli/wav2vec2-negative/"
MODEL_CHECKPOINT = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"

WAV2VEC_MODEL = AutoModelForAudioClassification.from_pretrained(MODEL_CHECKPOINT)

# EHCALABRES MODEL SPECIFIC ================
WAV2VEC_MODEL.projector = nn.Linear(1024, 1024, bias=True)
WAV2VEC_MODEL.classifier = nn.Linear(1024, 8, bias=True)

torch_state_dict = torch.load('model_storage/pytorch_model.bin', map_location=torch.device('cpu'))

WAV2VEC_MODEL.projector.weight.data = torch_state_dict['classifier.dense.weight']
WAV2VEC_MODEL.projector.bias.data = torch_state_dict['classifier.dense.bias']

WAV2VEC_MODEL.classifier.weight.data = torch_state_dict['classifier.output.weight']
WAV2VEC_MODEL.classifier.bias.data = torch_state_dict['classifier.output.bias']
# ==============

from transformers import AutoFeatureExtractor

feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_CHECKPOINT)
max_duration = 15.0
WAV2VEC_MODEL.cuda()
WAV2VEC_MODEL.eval()

SETUP = 2


def call_frustration(audio_url):
    if audio_url is not None:
        urllib.request.urlretrieve(audio_url, "audio_cache/audio.wav")
        wav, silence_ratio = speech_only_audio()
        if silence_ratio >= 0.5:
            return 1, f"student is pausing more than usual => {silence_ratio}"
        inputs = feature_extractor(
            wav,
            sampling_rate=feature_extractor.sampling_rate,
            max_length=int(feature_extractor.sampling_rate * max_duration),
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).input_values
        inputs = inputs.cuda()
        # EHCALABRES MODEL SPECIFIC ================
        neg_emotion = WAV2VEC_MODEL(inputs)[0]
        neg_emotion = torch.softmax(neg_emotion, dim=1)
        return_str = " ".join(['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised', "===>", str(neg_emotion)])
        if SETUP == 0:
            neg_emotion = neg_emotion[0, 0] + neg_emotion[0, 2] + neg_emotion[0, 3] + neg_emotion[0, 6]
            neg_emotion = neg_emotion.detach().cpu().item()
        elif SETUP == 1:
            neg_emotion = neg_emotion[0, 0] + neg_emotion[0, 2] + neg_emotion[0, 3]
            neg_emotion = neg_emotion.detach().cpu().item()
        elif SETUP == 2:
            neg_emotion = neg_emotion[0, 0]
            neg_emotion = neg_emotion.detach().cpu().item()
        elif SETUP == 3:
            neg_emotion = neg_emotion[0, 2] + neg_emotion[0, 3]
            neg_emotion = neg_emotion.detach().cpu().item()
        elif SETUP == 4:
            neg_emotion = neg_emotion[0, 2] + neg_emotion[0, 0]
            neg_emotion = neg_emotion.detach().cpu().item()
        elif SETUP == 5:
            neg_emotion = neg_emotion[0, 3] + neg_emotion[0, 0]
            neg_emotion = neg_emotion.detach().cpu().item()
        else:
            neg_emotion = int(int(torch.argmax(neg_emotion, dim=1).detach().cpu().item()) in [0, 2, 3])
    else:
        # Default to not frustrated
        neg_emotion = 0
        return_str = ""
    return neg_emotion, return_str


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--frust_thresh", type=float, default=0.5)
    parser.add_argument("--setup", type=int, default=2)
    args = parser.parse_args()
    FRUST_THRESH = args.frust_thresh
    SETUP = args.setup
    agg_label_file = json.load(open("../audio_emotion_data/emotion_labeled_data.json"))
    preds = []
    labels = []

    if os.path.exists(f"{SETUP}_outputs.json"):
        data = json.load(open(f"{SETUP}_{int(FRUST_THRESH * 10)}.json"))
        preds = data["preds"]
        labels = data["labels"]
    else:
        neg_counts = 0

        for af in agg_label_file:
            if agg_label_file[af]["label"] == "Negative":
                try:
                    class_frust, ret_str = call_frustration(af["audio_url"])
                except RuntimeError:
                    continue
                preds.append(class_frust)
                labels.append(1)
        neg_counts = len(preds)
        for af in agg_label_file:
            if agg_label_file[af]["label"] == "Neutral":
                try:
                    class_frust, ret_str = call_frustration(af["audio_url"])
                except RuntimeError:
                    continue
                preds.append(class_frust)
                labels.append(0)
                neg_counts = neg_counts - 1
        json.dump({
            "preds": preds, "labels": labels
        }, open(f"{SETUP}_outputs.json", "w+"))
    preds = [int(x > FRUST_THRESH) for x in preds]

    print(f1_score(labels, preds, average="weighted"))
    print(precision_recall_fscore_support(labels, preds, average="weighted"))
