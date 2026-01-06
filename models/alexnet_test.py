import torch
from torchvision.models import alexnet
from torchvision import transforms
import torch.nn as nn
from support.LMDB import LMDBDataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd

torch.backends.cudnn.benchmark = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IDX_TO_CLASS = {
    0: "Normal",
    1: "Pneumonia",
    2: "COVID-19"
}

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

test_dataset = LMDBDataset(
    lmdb_path="./lmdbs/test.lmdb",
    transform=val_transform
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0,   # LMDB-safe
    pin_memory=True
)

def load_fold_model(weight_path):
    model = alexnet(weights=None)
    model.classifier[6] = nn.Linear(4096, 3)
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

models = [
    load_fold_model(f"./checkpoints/alexnet_fold_{i}.pth")
    for i in range(5)
]

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        logits_sum = None

        for model in models:
            with torch.amp.autocast("cuda"):
                logits = model(images)

            if logits_sum is None:
                logits_sum = logits
            else:
                logits_sum += logits

        avg_logits = logits_sum / len(models)
        preds = avg_logits.argmax(1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


print("Test Accuracy:", accuracy_score(all_labels, all_preds))

print(
    classification_report(
        all_labels,
        all_preds,
        target_names=[IDX_TO_CLASS[i] for i in range(3)]
    )
)

print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

df = pd.DataFrame({
    "gt_label": [IDX_TO_CLASS[i] for i in all_labels],
    "pred_label": [IDX_TO_CLASS[i] for i in all_preds]
})

df.to_csv("test_predictions_alexnet_ensemble.csv", index=False)

for i in range(10):
    print(
        "GT:", IDX_TO_CLASS[all_labels[i]],
        "PRED:", IDX_TO_CLASS[all_preds[i]]
    )