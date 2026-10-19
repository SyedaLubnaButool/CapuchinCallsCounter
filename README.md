# Capuchin Bird Call Detection 🐦

A deep learning project that detects Capuchin bird calls in audio recordings using **TensorFlow/Keras** and **Librosa**.

I built this project to learn how audio can be converted into spectrograms and analysed using a 2D convolutional neural network (CNN).

## What the project does

- Classifies short audio clips as Capuchin bird present or absent.
- Converts audio into mel spectrograms.
- Processes long forest recordings in three-second windows.
- Estimates call counts by grouping consecutive positive windows.

## Dataset

The dataset comes from [Z by HP Unlocked Challenge 3 — Signal Processing](https://www.kaggle.com/datasets/kenjee/z-by-hp-unlocked-challenge-3-signal-processing), shared by Ken Jee on Kaggle.

It contains:

| Audio type | Number of files |
|------------|----------------:|
| Capuchin bird clips | 217 |
| Non-Capuchin clips | 593 |
| Forest recordings | 100 |

Training clips are WAV files, while forest recordings are MP3 files. Download the dataset separately from Kaggle.

## Tools Used

- Python
- TensorFlow / Keras
- Librosa
- NumPy
- Matplotlib
- Scikit-learn

## Audio Preprocessing

Each training clip is:

1. Loaded as mono audio at a sample rate of 16,000 Hz.
2. Cropped or padded with silence to three seconds.
3. Converted into a mel spectrogram using:
   - 64 frequency bands
   - FFT window size of 512 samples
   - Hop length of 128 samples
4. Converted from power to decibels relative to the clip's maximum.
5. Given a channel dimension, producing an input shape of `(64, 376, 1)`.

The notebook also includes waveform and spectrogram plots to explain the audio representation.

## CNN Architecture

| Layer | Configuration |
|-------|---------------|
| Input | `(64, 376, 1)` |
| Conv2D | 16 filters, 3 × 3 kernel, ReLU |
| MaxPooling2D | 2 × 2 |
| Conv2D | 32 filters, 3 × 3 kernel, ReLU |
| MaxPooling2D | 2 × 2 |
| Flatten | Converts feature maps into a vector |
| Dense | 32 units, ReLU |
| Output | 1 unit, sigmoid |

The sigmoid output gives a predicted probability of a Capuchin call being present.

## Training

- **Training set:** 648 clips — 80%
- **Test set:** 162 clips — 20%
- **Split:** Stratified to preserve class proportions
- **Optimizer:** Adam
- **Loss:** Binary cross-entropy
- **Batch size:** 16
- **Epochs:** 10

Since there are more negative than positive clips, class weights give positive examples greater weight during training. These weights are calculated using only the training labels.

## Results

The saved notebook run achieved:

| Metric | Result |
|--------|-------:|
| Test accuracy | **99.38%** |
| Correct test predictions | **161 / 162** |
| Test loss | **0.04293** |

For `recording_95.mp3`, the model estimated **5 calls**, matching my manual count after listening to the recording.

These results describe one training run. Results can vary when the model is retrained.

## Counting Calls in Forest Recordings

Long recordings are divided into consecutive three-second windows. The final window is padded if necessary.

Each window is classified using a probability threshold of **0.5**:

- Probability ≥ 0.5 → bird present
- Probability < 0.5 → bird absent

Consecutive positive windows are grouped together. For example:

`[0, 1, 1, 0, 1]` → **2 estimated calls**

This avoids automatically counting a call twice when it spans neighbouring windows.

## How to Run

1. Download the notebook and the Kaggle dataset.
2. Place these extracted folders beside the notebook:
   - `Parsed_Capuchinbird_Clips`
   - `Parsed_Not_Capuchinbird_Clips`
   - `Forest Recordings`
3. Install the dependencies:

   `pip install tensorflow librosa numpy matplotlib scikit-learn soundfile notebook`

4. Open the notebook in Jupyter or VS Code.
5. Select the Python environment containing the dependencies.
6. Run the cells from top to bottom.

If the dataset is stored elsewhere, update `pos_dir`, `neg_dir`, and `forest_dir`.

After training and defining `count_calls()`, a recording can be checked with:

    path = os.path.join(forest_dir, "recording_95.mp3")
    count = count_calls(path)
    print("Estimated calls:", count)

## Limitations

- Test accuracy measures classification of short clips, not call-counting accuracy across all forest recordings.
- Consecutive-window grouping is an estimate: nearby calls can be merged, and missed windows can split one call into multiple groups.
- The split is at clip level; clips from the same original recording may share similar background sounds.
- The test set was checked during experimentation, so a fresh final evaluation would strengthen the results.
- Only one forest recording's count was manually verified.
- File ordering and model initialisation are not fully controlled, so results may vary between runs.

## What I Learned

- Loading audio and understanding sample rates.
- Cropping and padding recordings.
- Creating and interpreting mel spectrograms.
- Preparing input dimensions for a CNN.
- Handling class imbalance with class weights.
- Training and evaluating a binary classifier.
- Applying a short-clip classifier to longer recordings.

## Future Improvements

- Add precision, recall, and a confusion matrix.
- Use separate training, validation, and test sets.
- Split clips by their original source recording where possible.
- Check estimated counts against more manually labelled recordings.
- Improve reproducibility with sorted file lists and random seeds.
