# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc/4.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import lightning as L
import torchmetrics
from torch.optim.lr_scheduler import _LRScheduler
import wandb
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC, ConfusionMatrix, PrecisionRecallCurve


class WarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super(WarmupScheduler, self).__init__(optimizer, last_epoch)
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [base_lr * (self.last_epoch + 1) / self.warmup_steps for base_lr in self.base_lrs]
        return [base_lr for base_lr in self.base_lrs]


class ReadingClassifier(L.LightningModule):
    def __init__(self, model, num_classes=4, lr=3e-3, use_loss_weight=False, loss_weight=None):
        super().__init__()
        self.model = model
        self.lr = lr
        self.task = "multiclass" if num_classes > 2 else "binary"
        self.loss_fn = nn.CrossEntropyLoss()
        self.num_classes = num_classes
        if use_loss_weight:
            self.loss_fn = nn.CrossEntropyLoss(weight=loss_weight)
        self.accuracy = Accuracy(num_classes=num_classes, average='micro', task=self.task)
        self.precision = Precision(num_classes=num_classes, average='macro', task=self.task)
        self.recall = Recall(num_classes=num_classes, average='macro', task=self.task)
        self.f1score = F1Score(num_classes=num_classes, average='macro', task=self.task)
        self.auroc = AUROC(num_classes=num_classes, average='macro', task=self.task)
        self.confmat = ConfusionMatrix(num_classes=num_classes, task=self.task)
        self.pr_curve = PrecisionRecallCurve(num_classes=num_classes, task=self.task)
        self.val_loss = torchmetrics.MeanMetric()
        self.val_pred = torchmetrics.CatMetric()
        self.val_label = torchmetrics.CatMetric()
        self.test_pred = torchmetrics.CatMetric()
        self.test_label = torchmetrics.CatMetric()
        self.test_per_type_metrics = {}
        

    def get_per_type_metric(self, type_str):
        if type_str not in self.test_per_type_metrics:
            self.test_per_type_metrics[type_str] = {
                'pred': torchmetrics.CatMetric(),
                'label': torchmetrics.CatMetric(),
                'auroc': AUROC(num_classes=self.num_classes, average='macro', task=self.task),
                'confmat': ConfusionMatrix(num_classes=self.num_classes, task=self.task),
            }
        return self.test_per_type_metrics[type_str]

    def log_pr_curve(self, precision, recall, title="Precision-Recall Curve", plot_id="pr_curve"):
        # Prepare data for WandB plot
        data = [[p, r] for p, r in zip(precision.tolist(), recall.tolist())]
        table = wandb.Table(data=data, columns=["Precision", "Recall"])
        # Log the plot to WandB
        self.logger.log({
            plot_id: self.logger.plot.line(table, "Recall", "Precision", title=title)
        })

    def on_validation_epoch_start(self):
        self.val_task_preds = {}
        self.val_task_targets = {}

    def on_test_epoch_start(self):
        self.test_task_preds = {}
        self.test_task_targets = {}

    def training_step(self, batch, batch_idx):
        et_batch = batch  # B, num_track, input_dim
        pred = self.model(et_batch)
        target = batch['label']  # B
        loss = self.loss_fn(pred, target)
        pred = torch.argmax(pred, -1)
        self.log("train_loss", loss)
        acc = self.accuracy(pred, target)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        et_batch = batch  # B, num_track, input_dim
        pred = self.model(et_batch)
        target = batch['label']  # B
        task = batch['task']  # B
        loss = self.loss_fn(pred, target)
        pred_ = pred[:,1] if self.task == "binary" else pred        
        self.auroc.update(preds=pred_, target=target)
        self.pr_curve.update(preds=pred_, target=target)
        pred = torch.argmax(pred, -1)
        self.val_loss.update(value=loss)
        self.val_pred.update(value=pred)
        self.val_label.update(value=target)
        
        # Store predictions and targets per task
        for t in torch.unique(task):
            task_mask = task == t
            task_preds = pred[task_mask]
            task_targets = target[task_mask]
            if t.item() not in self.val_task_preds:
                self.val_task_preds[t.item()] = []
                self.val_task_targets[t.item()] = []
            self.val_task_preds[t.item()].append(task_preds)
            self.val_task_targets[t.item()].append(task_targets)
        return loss

    def test_step(self, batch, batch_idx):
        et_batch = batch  # B, num_track, input_dim
        pred = self.model(et_batch)
        target = batch['label']  # B
        task = batch['task']  # B
        pred_ = pred[:,1] if self.task == "binary" else pred
        self.auroc.update(preds=pred_, target=target)
        self.pr_curve.update(preds=pred_, target=target)
        pred = torch.argmax(pred, -1)
        self.test_pred.update(value=pred)
        self.test_label.update(value=target)
        
        # Store predictions and targets per task
        for t in torch.unique(task):
            task_mask = task == t
            task_preds = pred[task_mask]
            task_targets = target[task_mask]
            if t.item() not in self.test_task_preds:
                self.test_task_preds[t.item()] = []
                self.test_task_targets[t.item()] = []
            self.test_task_preds[t.item()].append(task_preds)
            self.test_task_targets[t.item()].append(task_targets)

    def predict_step(self, batch):
        return F.softmax(self.model(batch), -1)[:,1]

    def log_confusion_matrix_as_table(self, confmat):
        confmat = confmat #/ torch.sum(confmat)
        data = []
        columns = ["-"]
        for i in range(len(confmat)):
            columns.append("p"+str(i))
            row = ["gt"+str(i)]
            for j in range(len(confmat[i])):
                row.append(confmat[i][j] / torch.sum(confmat[i]))
            data.append(row)
        self.logger.log_table(key="conf_mat", columns=columns, data=data)
        
    def on_validation_epoch_end(self):
        avg_loss = self.val_loss.compute()
        self.val_loss.reset()
        preds = self.val_pred.compute()
        targets = self.val_label.compute()
        val_acc = self.accuracy(preds, targets)
        val_auc = self.auroc.compute()
        val_recall = self.recall(preds, targets)
        val_precision = self.precision(preds, targets)
        val_f1score = self.f1score(preds, targets)
        confmat = self.confmat(preds, targets)
        if self.num_classes == 2:
            precision, recall, threshold = self.pr_curve.compute()
            num_points_to_log = 50  # Adjust this number as needed
            indices = torch.linspace(0, len(precision) - 1, steps=num_points_to_log).long()
            precision = precision[indices]
            recall = recall[indices]
            #threshold = threshold[indices]
            self.logger.log_table(key="val_pr_curve", columns=["Precision", "Recall"], data=list(zip(precision.tolist(), recall.tolist())))
            #self.log_pr_curve(precision, recall, title="Validation Precision-Recall Curve", plot_id="val_pr_curve")
            self.pr_curve.reset()
        values = {
            "val_loss": avg_loss,
            "val_acc": val_acc,
            "val_auc": val_auc.cpu().item(),
            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_f1_score": val_f1score,
        }
        self.log_dict(values, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log_confusion_matrix_as_table(confmat)

        # Prepare table data for validation metrics per task
        table_data = []
        for t in self.val_task_preds:
            task_preds = torch.cat(self.val_task_preds[t])
            task_targets = torch.cat(self.val_task_targets[t])

            task_acc = self.accuracy(task_preds, task_targets)
            task_precision = self.precision(task_preds, task_targets)
            task_recall = self.recall(task_preds, task_targets)
            task_f1score = self.f1score(task_preds, task_targets)

            # Append row for the current task
            table_data.append([f"Task {t}", task_acc.item(), task_precision.item(), task_recall.item(), task_f1score.item()])

        # Log the table
        if len(table_data) > 0:
            self.logger.log_table(
                key="val_metrics_per_task",
                columns=["Task", "Accuracy", "Precision", "Recall", "F1"],
                data=table_data
            )
        # Reset task-specific metrics
        self.val_task_preds.clear()
        self.val_task_targets.clear()
        self.val_pred.reset()
        self.val_label.reset()
        self.auroc.reset()

    def on_test_epoch_end(self):
        preds = self.test_pred.compute()
        targets = self.test_label.compute()
        test_auc = self.auroc.compute()
        test_recall = self.recall(preds, targets)
        test_precision = self.precision(preds, targets)
        test_f1score = self.f1score(preds, targets)
        test_acc = self.accuracy(preds, targets)
        confmat = self.confmat(preds, targets)
        if self.num_classes == 2:
            precision, recall, threshold = self.pr_curve.compute()
            num_points_to_log = 50  # Adjust this number as needed
            indices = torch.linspace(0, len(precision) - 1, steps=num_points_to_log).long()
            precision = precision[indices]
            recall = recall[indices]
            #threshold = threshold[indices]
            self.logger.log_table(key="test_pr_curve", columns=["Precision", "Recall"], data=list(zip(precision.tolist(), recall.tolist())))
            #self.log_pr_curve(precision, recall, title="Test Precision-Recall Curve", plot_id="test_pr_curve")

            self.pr_curve.reset()
        values = {
            "test_auc": test_auc.cpu().item(),
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f1_score": test_f1score,
            "test_accuracy": test_acc,
        }
        self.log_confusion_matrix_as_table(confmat)
        self.log_dict(values, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # Prepare table data for test metrics per task
        table_data = []

        for t in self.test_task_preds:
            task_preds = torch.cat(self.test_task_preds[t])
            task_targets = torch.cat(self.test_task_targets[t])
            task_acc = self.accuracy(task_preds, task_targets)
            task_precision = self.precision(task_preds, task_targets)
            task_recall = self.recall(task_preds, task_targets)
            task_f1score = self.f1score(task_preds, task_targets)
            # Append row for the current task
            table_data.append([f"Task {t}", task_acc.item(), task_precision.item(), task_recall.item(), task_f1score.item()])
        # Log the table

        if len(table_data) > 0:
            self.logger.log_table(
                key="test_metrics_per_task",
                columns=["Task", "Accuracy", "Precision", "Recall", "F1"],
                data=table_data
            )

        # Reset task-specific metrics
        self.test_task_preds.clear()
        self.test_task_targets.clear()
        self.test_pred.reset()
        self.test_label.reset()
        self.auroc.reset()
        self.recall.reset()
        self.precision.reset()
        self.f1score.reset()
        self.accuracy.reset()
        self.confmat.reset()
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = WarmupScheduler(optimizer, warmup_steps=100)
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]