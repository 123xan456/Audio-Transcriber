from nltk.tokenize import sent_tokenize, word_tokenize
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

DEFAULT_MODEL = "philschmid/bart-large-cnn-samsum"

'''
Summarization mostly uses a pretrained model from huggingface.
References:
https://huggingface.co/philschmid/bart-large-cnn-samsum
https://www.guru99.com/tokenize-words-sentences-nltk.html
'''

class Summarizer:
    def __init__(self, modelName=DEFAULT_MODEL) -> None:
        self.modelName = modelName
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.modelName)
        self.tokenizer = AutoTokenizer.from_pretrained(self.modelName)

    def splitSentences(self, paragraph, maxSentenceLength=1024):
        sentences = sent_tokenize(paragraph)
        chunks = []
        sentenceLenPair = []
        for sentence in sentences:
            sentenceLenPair.append((sentence, len(word_tokenize(sentence))))

        modelTokenLimit = maxSentenceLength
        chunks = []
        chunk = ""
        totalTokens = 0
        while sentenceLenPair:
            while totalTokens < modelTokenLimit:
                if len(sentenceLenPair) == 0:
                    break
                sentence, size = sentenceLenPair.pop(0)
                chunk = chunk + sentence + " "
                totalTokens = totalTokens + size
            chunks.append(chunk)
            totalTokens = 0
            chunk = ""
        return chunks

    def summarize(
        self, text, maxLen=150, minLen=40, lengthPenalty=2.0, repetitionPenalty=1.2
    ):
        summary = ""
        chunks = self.splitSentences(text)
        for chunk in chunks:
            if len(chunk) < minLen:
                continue
            inputs = self.tokenizer(
                "summarize: " + chunk, return_tensors="pt", truncation=True
            )
            outputs = self.model.generate(
                inputs["input_ids"],
                # max_length=maxLen,
                max_new_tokens=maxLen,
                min_length=minLen,
                length_penalty=2.0,
                repetition_penalty=1.2,
                # beam search
                num_beams=4,
                early_stopping=True,
                # top p sampling
                # do_sample=True,
                # top_p=0.80,
            )
            summaryPart = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            summary = summary + summaryPart + "\n"

        return summary
