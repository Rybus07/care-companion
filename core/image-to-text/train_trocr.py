import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    TrOCRProcessor, 
    VisionEncoderDecoderModel, 
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments, 
    default_data_collator
)
import evaluate
from PIL import Image

# 1. Load Processor and Pretrained ViT-Transformer Model
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# Set special tokens for generation
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size

# 2. Define Custom Clinical Dataset
class PrescriptionDataset(Dataset):
    def __init__(self, df, img_dir, processor, max_target_length=128):
        self.df = df
        self.img_dir = img_dir
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df.iloc[idx]['file_name']
        text = self.df.iloc[idx]['text']
        image = Image.open(f"{self.img_dir}/{file_name}").convert("RGB")
        
        # Pixel values for ViT Encoder
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()
        
        # Token labels for Transformer Decoder
        labels = self.processor.tokenizer(
            text, 
            padding="max_length", 
            max_length=self.max_target_length
        ).input_ids
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        return {"pixel_values": pixel_values, "labels": torch.tensor(labels)}

# 3. Setup Metrics (CER & WER)
cer_metric = evaluate.load("cer")
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"cer": cer, "wer": wer}

# 4. Training Arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./trocr_prescription_weights",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    num_train_epochs=5,
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="cer"
)