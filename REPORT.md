# EEC 174AY Lab 2: Supervised Learning

**Student Name:** Joshua Kofoworola
**Course:** EEC 174AY  
**Assignment:** Lab 2 — Supervised Learning  


## 1. Data and Labels

### 1.1 Prediction Task

Each row of the dataset represents one breath from a patient receiving ventilator support. The dataset contains features extracted from ventilator flow and pressure waveforms, such as inspiratory time, expiratory volume, maximum flow, and airway pressure.

The model predicts one of three classes: Non-PVA, Breath Stacking Asynchrony (BSA), or Double-Trigger Asynchrony (DTA). Non-PVA represents breaths without either of these two asynchrony annotations. Breaths annotated only as BSA belong to the BSA class, while breaths annotated only as DTA belong to the DTA class.

### 1.2 Ambiguous Labels

Breaths annotated as both BSA and DTA are ambiguous because they match two classes simultaneously, even though the model is designed to predict one class per breath. Keeping these examples without a clear labeling rule could introduce confusion during training and evaluation.

One alternative to removing them would be to create a separate combined BSA-DTA class. Another possibility would be to use a multilabel classification approach that allows a breath to have both labels.

### 1.3 Patient Identifiers

The `patient` and `breathid` columns are excluded from model inputs because they identify the patient and individual breath rather than directly describing the physiological characteristics of the breath. Using these identifiers as predictive features could allow a model to learn patterns associated with particular patients or IDs instead of learning meaningful relationships between waveform features and asynchrony.

However, excluding these identifiers does not guarantee generalization to unseen patients. Breaths from the same patient may still share physiological characteristics. Patient-level validation, in which all breaths from a patient are assigned to either training or validation, is needed to estimate performance on previously unseen patients.

## 2. Features and Preprocessing

### 2.1 Feature Extraction

Feature extraction converts raw ventilator waveforms into numerical measurements that a machine learning model can use. Instead of directly processing every waveform sample, the model can learn from informative measurements describing each breath.

For example, inspiratory time (`i_time`) measures the duration of inhalation, while peak inspiratory pressure (`pip`) represents the maximum pressure during inhalation. These features capture different aspects of breathing: timing and pressure. Other features, such as respiratory rate (`rr`) and the tidal volume ratio (`tve_tvi_ratio`), describe breathing frequency and the relationship between exhaled and inhaled volume.

### 2.2 Feature Scaling

StandardScaler was used in the L1-regularized Logistic Regression pipeline to standardize numerical features. It transforms features using the training data's mean and standard deviation.

The scaler should be fitted only on the training data and then applied to the validation data. Fitting it on both datasets would allow information about the validation distribution to influence preprocessing, creating data leakage. Using the training statistics ensures that validation data remains separate during model development.

## 3. Model Performance

### 3.1 Random Forest

A Random Forest combines predictions from multiple decision trees. Each tree is trained using randomized samples and feature selections, and the classifier combines their predictions to determine the final class.

Random Forest is a reasonable choice for this dataset because it can model nonlinear relationships between physiological features and asynchrony classes. It can also capture interactions between features without requiring all relationships to be linear.

### 3.2 Class-Specific Errors

DTA had the lowest recall in the Random Forest classification report. The final report showed approximately 0.01 recall for DTA, compared with 0.96 for Non-PVA and 0.97 for BSA. The average DTA recall across 10 runs was 0.025.

This indicates that the model missed most actual DTA examples. Although DTA precision was 1.00 in the final report, this does not mean that DTA detection was effective: the model identified very few DTA examples in the first place.

Overall accuracy alone can hide this problem because Non-PVA and BSA account for more examples than DTA. A model can achieve high accuracy while failing to identify an important minority class. Class-specific recall and macro F1-score therefore provide useful additional information.

### 3.3 Precision and Recall

A false positive occurs when a model predicts a class for a breath that does not belong to that class. A false negative occurs when the model fails to identify a breath that actually belongs to the class.

Precision measures the proportion of predicted class examples that are correct, while recall measures the proportion of actual class examples that are identified correctly.

For DTA detection, recall may be prioritized when the goal is to avoid missing genuine asynchrony events. However, improving recall may increase false positives, so precision must also be monitored. The appropriate balance depends on the clinical application and the consequences of different errors.

### 3.4 Model Comparison

Random Forest was compared with L1-regularized Logistic Regression using the same reported validation set.

| Metric | Random Forest | Logistic Regression |
|---|---:|---:|
| Accuracy | 0.88 | 0.77 |
| Misclassification rate | 0.1171 | 0.2269 |
| Macro F1-score | 0.62 | 0.52 |
| Non-PVA F1-score | 0.93 | 0.86 |
| BSA F1-score | 0.90 | 0.66 |
| DTA F1-score | 0.02 | 0.04 |

Random Forest performed better overall, achieving higher accuracy and macro F1-score and a lower misclassification rate. It performed better for Non-PVA and BSA, while Logistic Regression had a slightly higher DTA F1-score. However, both models struggled to detect DTA effectively.

## 4. Model Selection and Validation

### 4.1 Feature Selection and Expert Rules

Mutual-information feature selection retained eight features: `e_time`, `i_e_ratio`, `rr`, `tve`, `tve_tvi_ratio`, `maw`, `ip_auc`, and `ep_auc`.

The full-feature Random Forest achieved approximately 88% accuracy and a macro F1-score of 0.62. The mutual-information Random Forest achieved approximately 89% accuracy and a macro F1-score of 0.6527 on the validation set. Thus, the selected features produced a small improvement in overall validation performance.

However, DTA recall remained low at 0.07 for the mutual-information model. The expert-rule classifier achieved approximately 92% accuracy and a macro F1-score of 0.8322, with DTA recall of 0.50. It therefore performed better than the selected-feature Random Forest on the reported validation set.

A feature that appears useful on training data may fail to improve validation performance because it may capture noise, overfit the training examples, or provide little additional information for unseen examples. Feature selection should therefore be evaluated using validation data that was not used to select the features.

### 4.2 Five-Fold Cross-Validation

Stratification preserves approximately the same class proportions across the folds as in the original dataset. This helps ensure that each fold contains a representative distribution of the target classes, including minority classes.

The five-fold cross-validation results were:

| Fold | Macro precision | Macro recall | Macro F1-score |
|---|---:|---:|---:|
| 1 | 0.8619 | 0.7045 | 0.7607 |
| 2 | 0.8681 | 0.7200 | 0.7750 |
| 3 | 0.8442 | 0.6668 | 0.7267 |
| 4 | 0.8681 | 0.6885 | 0.7506 |
| 5 | 0.8544 | 0.6652 | 0.7256 |
| **Average** | **0.8593** | **0.6890** | **0.7477** |

The average macro F1-score was 0.7477. Fold 2 achieved the highest macro F1-score of 0.7750, while Fold 5 had the lowest at 0.7256. The variation indicates that model performance depends somewhat on the data included in each fold. Substantial differences between folds could suggest sensitivity to the training samples, variation in class difficulty, or insufficient representation of particular patterns.

### 4.3 Patient-Level Validation

If breaths from the same patient appear in both the training and validation sets, the model may benefit from similarities between those breaths. This can produce an overly optimistic estimate of performance on genuinely unseen patients.

To evaluate generalization more reliably, the dataset should be divided by patient rather than by individual breath. All breaths from a particular patient should belong exclusively to one fold or split. Group-based methods such as GroupKFold or StratifiedGroupKFold can be used where appropriate. Feature selection, scaling, and model tuning should also be performed within the training portion of each split to prevent data leakage.

This approach would provide a more realistic estimate of how well the classifier performs on patients whose data were not available during training.

## Conclusion

The experiments show that Random Forest performed better overall than L1-regularized Logistic Regression on the reported validation set. Feature selection provided a small improvement in Random Forest performance, while the expert-rule classifier achieved the highest reported macro F1-score of 0.8322.

However, DTA detection remained a major challenge for both machine learning models. The cross-validation results also showed some variation between folds. Further evaluation using patient-level splits would help determine whether the observed performance generalizes to patients not seen during training.