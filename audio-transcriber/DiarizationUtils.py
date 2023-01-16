import json
import logging
from pathlib import Path

from nemo.collections.asr.models.msdd_models import ClusteringDiarizer
from omegaconf import OmegaConf
import whisperx

from Configs import CACHE_DIRECTORY

logging.getLogger('nemo_logger').setLevel(logging.ERROR)

def get_words_speaker_mapping(wrd_ts, spk_ts, word_anchor_option="start"):
    def get_word_ts_anchor(s, e, option="start"):
        if option == "end":
            return e
        elif option == "mid":
            return (s + e) / 2
        return s

    s, e, sp = spk_ts[0]
    wrd_pos, turn_idx = 0, 0
    wrd_spk_mapping = []
    for wrd_dict in wrd_ts:
        ws, we, wrd = (int(wrd_dict["start"] * 1000), int(wrd_dict["end"] * 1000), wrd_dict["text"],)
        wrd_pos = get_word_ts_anchor(ws, we, word_anchor_option)
        while wrd_pos > float(e) and (turn_idx != len(spk_ts) - 1):
            turn_idx += 1
            turn_idx = min(turn_idx, len(spk_ts) - 1)
            s, e, sp = spk_ts[turn_idx]
        result = {"word": wrd, "start_time": ws, "end_time": we, "speaker": sp}
        wrd_spk_mapping.append(result)
    return wrd_spk_mapping


def get_realigned_ws_mapping_with_punctuation(word_speaker_mapping, max_words_in_sentence=50):

    sentence_ending_punctuations = ".?!"

    def get_first_word_idx_of_sentence(word_idx, word_list, speaker_list, max_words):
        is_word_sentence_end = (lambda x: x >= 0 and word_list[x][-1] in sentence_ending_punctuations)
        left_idx = word_idx
        while (
            left_idx > 0
            and word_idx - left_idx < max_words
            and speaker_list[left_idx - 1] == speaker_list[left_idx]
            and not is_word_sentence_end(left_idx - 1)
        ):
            left_idx -= 1

        return left_idx if left_idx == 0 or is_word_sentence_end(left_idx - 1) else -1

    def get_last_word_idx_of_sentence(word_idx, word_list, max_words):
        is_word_sentence_end = (lambda x: x >= 0 and word_list[x][-1] in sentence_ending_punctuations)
        right_idx = word_idx
        while (
            right_idx < len(word_list)
            and right_idx - word_idx < max_words
            and not is_word_sentence_end(right_idx)
        ):
            right_idx += 1

        return (
            right_idx
            if right_idx == len(word_list) - 1 or is_word_sentence_end(right_idx)
            else -1
        )

    is_word_sentence_end = (
        lambda x: x >= 0
        and word_speaker_mapping[x]["word"][-1] in sentence_ending_punctuations
    )
    wsp_len = len(word_speaker_mapping)

    words_list, speaker_list = [], []
    for k, line_dict in enumerate(word_speaker_mapping):
        word, speaker = line_dict["word"], line_dict["speaker"]
        words_list.append(word)
        speaker_list.append(speaker)

    k = 0
    while k < len(word_speaker_mapping):
        line_dict = word_speaker_mapping[k]
        if (
            k < wsp_len - 1
            and speaker_list[k] != speaker_list[k + 1]
            and not is_word_sentence_end(k)
        ):
            left_idx = get_first_word_idx_of_sentence(k, words_list, speaker_list, max_words_in_sentence)
            right_idx = (get_last_word_idx_of_sentence(k, words_list, max_words_in_sentence - k + left_idx - 1) if left_idx > -1 else -1)
            if min(left_idx, right_idx) == -1:
                k += 1
                continue

            spk_labels = speaker_list[left_idx : right_idx + 1]
            mod_speaker = max(set(spk_labels), key=spk_labels.count)
            if spk_labels.count(mod_speaker) < len(spk_labels) // 2:
                k += 1
                continue

            speaker_list[left_idx : right_idx + 1] = [mod_speaker] * (right_idx - left_idx + 1)
            k = right_idx

        k += 1

    k, realigned_list = 0, []
    while k < len(word_speaker_mapping):
        line_dict = word_speaker_mapping[k].copy()
        line_dict["speaker"] = speaker_list[k]
        realigned_list.append(line_dict)
        k += 1

    return realigned_list


def get_sentences_speaker_mapping(word_speaker_mapping, spk_ts):
    s, e, spk = spk_ts[0]
    prev_spk = spk

    snts = []
    snt = {"speaker": f"Speaker {spk}", "start_time": s, "end_time": e, "text": ""}

    for wrd_dict in word_speaker_mapping:
        wrd, spk = wrd_dict["word"], wrd_dict["speaker"]
        s, e = wrd_dict["start_time"], wrd_dict["end_time"]
        if spk != prev_spk:
            snts.append(snt)
            snt = {
                "speaker": f"Speaker {spk}",
                "start_time": s,
                "end_time": e,
                "text": "",
            }
        else:
            snt["end_time"] = e
        snt["text"] += wrd + " "
        prev_spk = spk

    snts.append(snt)
    return snts


def get_speaker_aware_transcript(sentences_speaker_mapping):
    speaker_aware_transcript = ""
    for sentence_dict in sentences_speaker_mapping:
        sp = sentence_dict["speaker"]
        text = sentence_dict["text"].lower()
        speaker_aware_transcript = speaker_aware_transcript + (f"{sp}: {text} ")
    return speaker_aware_transcript


class Diarizer:
    
    
    

    def __init__(self):
        MODEL_CONFIG = str(CACHE_DIRECTORY / "diar_infer_meeting.yaml")
        config = OmegaConf.load(MODEL_CONFIG)

        config.num_workers = 1
        config.batch_size = 32
        config.diarizer.manifest_filepath = str(CACHE_DIRECTORY / "manifest.json")
        config.diarizer.out_dir = str(CACHE_DIRECTORY / "diarized")
        config.diarizer.speaker_embeddings.model_path = "titanet_large"
        config.diarizer.speaker_embeddings.parameters.window_length_in_sec = [1.5,1.0,0.5,]
        config.diarizer.speaker_embeddings.parameters.shift_length_in_sec = [0.75,0.5,0.25,]
        config.diarizer.speaker_embeddings.parameters.multiscale_weights = [0.33,0.33,0.33,]
        config.diarizer.speaker_embeddings.parameters.save_embeddings = False
        config.diarizer.ignore_overlap = False
        config.diarizer.oracle_vad = False
        config.diarizer.collar = 0.25
        config.diarizer.vad.model_path = "vad_multilingual_marblenet"
        config.diarizer.oracle_vad = False

        self.model = ClusteringDiarizer(cfg=config)

    def diarize(self, transcription, audioPath):
        MODEL_NAME = "WAV2VEC2_ASR_LARGE_LV60K_960H"
        DEVICE = "cuda"
        MODEL, METADATA = whisperx.load_align_model(language_code="en", device=DEVICE, model_name=MODEL_NAME)
        resultAligned = whisperx.align(
            transcription["segments"], MODEL, METADATA, audioPath, DEVICE
        )

        # Storing words timestamps mapping in a file.
        with open(str(CACHE_DIRECTORY / "word_ts.text"), "w+") as f:
            for line in resultAligned["word_segments"]:
                line_temp = line.copy()
                line_temp["text"] = line_temp["text"].strip()
                f.write(f"{json.dumps(line_temp)}\n")

        # Creating the manifest
        diarize_manifest = {
            "audio_filepath": audioPath,
            "offset": 0,
            "duration": None,
            "label": "infer",
            "text": "-",
            "num_speakers": None,
            "rttm_filepath": str(CACHE_DIRECTORY / "diarized.rttm"),
            "uniq_id": "",
        }

        with open(CACHE_DIRECTORY / "manifest.json", "w") as f:
            f.write(json.dumps(diarize_manifest))

        # Running diarization
        self.model.diarize()

        speaker_ts = []
        sampleAudio = Path(audioPath)
        with open(str(CACHE_DIRECTORY/ "diarized"/ "pred_rttms"/ (str(sampleAudio.stem) + ".rttm")),"r",) as f:
            lines = f.readlines()
            for line in lines:
                line_list = line.split(" ")
                s = int(float(line_list[5]) * 1000)
                e = s + int(float(line_list[8]) * 1000)
                speaker_ts.append([s, e, int(line_list[11].split("_")[-1])])

        word_ts = []
        with open(CACHE_DIRECTORY / "word_ts.text", "r+") as f:
            for line in f:
                line_temp = json.loads(line)
                word_ts.append(line_temp)

        wsm = get_words_speaker_mapping(word_ts, speaker_ts, "start")
        wsm = get_realigned_ws_mapping_with_punctuation(wsm)
        ssm = get_sentences_speaker_mapping(wsm, speaker_ts)
        diarized = get_speaker_aware_transcript(ssm)
        return diarized


