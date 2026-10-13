# Heart disease evaluation example

`heart_disease_evaluation_example.csv` is generated from the actual labeled spectrogram files in:

`E:\Spectrogram-Based-Heart-Disease-Detection\processed_data\test`

Predictions were produced with the saved best checkpoint:

`E:\Spectrogram-Based-Heart-Disease-Detection\Traning\training_outputs16_18\mobilenetv2_heart_sound_best16_18.h5`

The companion training outputs in `E:\Spectrogram-Based-Heart-Disease-Detection\Traning\training_outputs16_18\content\training_outputs` were inspected for the model's evaluation history and setup. The model was run on all 346 labeled images in the separate `processed_data\test` split; images were resized to 224x224 and preprocessed with MobileNetV2's `preprocess_input`, matching the training script. These rows contain real predictions made by that checkpoint, not the reconstructed confusion-matrix rows from the earlier draft.

The CSV includes the relative spectrogram path as `sample_id`, the class-folder name as `true_label`, the model's top class as `predicted_label`, and one softmax probability column per class. It contains no image pixels or patient identifiers. This is a model evaluation example, not a clinical diagnosis.

## Use in ModelSleuth

1. Upload this CSV in **Analyze Data**.
2. Map `true_label` to true labels, `predicted_label` to predictions, and `sample_id` to sample IDs.
3. Optionally map each `prob_<class>` field to its matching class probability. The probabilities come from an independent test split; confirm the out-of-sample checkbox if you want the optional Cleanlab review.
4. Run analysis. Metrics are recalculated from these 346 per-image predictions.

The saved checkpoint achieved **94.51% accuracy** on this folder (327 of 346 correct). The original `evaluation_report.txt` reports a separate 160-row validation result and is not the source of this CSV's rows or metrics.

