# Prompt Sensitivity in Zero-Shot CLIP on GTSRB

COMS4036A Computer Vision project, University of the Witwatersrand

Bridgette Samuels and Shivash Jurakan

* * *

## What this project is about

CLIP can classify images without any training. You give it one sentence per class (for example "a photo of a speed limit 30 km/h") and it picks the sentence that best matches the image. We wanted to know how much the wording of that sentence matters on its own.

So we kept the model, the images and the classes exactly the same, and only changed the prompt template. We tested 7 templates on 18 German traffic sign classes that look very similar to each other (speed limits and red warning triangles).

* * *

## Results in short

- Best template (T1, just the class name): 53.5% top-1 accuracy
- Baseline (T2, "a photo of a {class}"): 50.0%
- Worst template (T5, "a blurry photo of a {class} sign on a road"): 42.6%
- Gap between best and worst: 10.9 percentage points (95% CI 9.9 to 12.1)
- Random guessing: 5.6%, always guessing the most common class: 12.8%

Every template was significantly different from the baseline (McNemar test, p < 0.0001 for each). Adding "taken on a sunny day", which has nothing to do with the sign, lowered accuracy by 3.4 points. Per class the effect was much bigger, with half of the classes moving by more than 45 points depending on the template.

We also checked whether CLIP's confidence could be used to pick a template. It could not: the template it was most confident with (T7) was not the most accurate one, and confidence and accuracy were moderately negatively correlated across the seven templates (Spearman correlation -0.61, with only seven templates, so this is exploratory).

The full write-up is in `report/report.pdf`.

* * *

## Folder layout

```
prompt_sensitivity.py    the full experiment (data, CLIP scoring, stats, figures)
run_colab.ipynb          notebook to run everything on Google Colab
requirements.txt         Python packages needed
results/                 the outputs from the run we used in the report
report/                  report source (LaTeX, IEEE format), figures and PDF
```

* * *

## How to run it

### Option 1: Google Colab (easiest)

1. Open `run_colab.ipynb` in Google Colab (File > Upload notebook, or open it from GitHub).
2. Go to Runtime > Change runtime type and choose T4 GPU.
3. Go to Runtime > Run all.

It takes around 10 to 15 minutes the first time, mostly downloading the dataset (about 600 MB) and encoding the images. At the end it prints a summary of the results, shows the figures and downloads a `results.zip`.

### Option 2: command line (any machine with a GPU, or a CPU if you are patient)

```
pip install -r requirements.txt
python prompt_sensitivity.py --data-root data --out results
```

Use a new `--out` folder for a new run. The script stops with an error if the cached features in the folder were made with a different class list or model.

Other options you can pass:

```
--seed 0             random seed (default 0)
--batch-size 256     images per batch when encoding
--max-per-class 0    use at most N images per class, 0 means use all
                     (e.g. --max-per-class 50 for a quick test)
--n-boot 1000        number of bootstrap resamples
```

The image embeddings get saved to `results/image_features.npz`. If that file already exists, the script skips the encoding step, so a second run only takes about a minute.

* * *

## Data

We used the test split of the German Traffic Sign Recognition Benchmark (GTSRB), which has 12,630 images over 43 classes. The images were labelled by hand by the people who made the dataset (Stallkamp et al., 2012).

You do not need to download it yourself. The script tries these sources in order:

1. A Kaggle copy of the official files: `meowmeowmeowmeowmeow/gtsrb-german-traffic-sign` (this is the one we ended up using)
2. A Hugging Face copy: `tanganke/gtsrb`
3. The official GTSRB server through torchvision (this kept timing out for us on Colab)

Whichever one works, the script checks that it matches the official test split (12,630 images in total, plus the exact image counts for six classes) before it carries on. This checks the labels and counts, not that every pixel is identical to the official files. It writes the source it used to `results/data_source.txt`.

From the test split we kept 18 classes, which gives 5,880 images:

- 8 speed limits: 20, 30, 50, 60, 70, 80, 100 and 120 km/h
- 10 warning triangles: general caution, dangerous curve left, dangerous curve right, double curve, bumpy road, slippery road, road narrows on the right, road work, traffic signals and pedestrians

**Licence and provenance.** The Kaggle page for the copy we used lists the licence as CC0 1.0 (public domain) (checked on 26 October 2026). We tried to check the terms on the original GTSRB site (benchmark.ini.rub.de) but it was not reachable, so we could not confirm its terms. Third-party copies of GTSRB elsewhere list different licences, so we relied only on the Kaggle page for the copy we downloaded. We did not redistribute the images (they are not in this repository), and we used them only for this coursework project. The images show road signs, not people.

* * *

## Method

We used the pretrained CLIP ViT-B/32 model (`openai/clip-vit-base-patch32` from Hugging Face). We did not train or fine-tune anything.

1. Encode each test image once with CLIP and save the embeddings.
2. For each template, build the 18 class sentences and encode them.
3. Give each image the class whose sentence is closest (cosine similarity).
4. Work out accuracy, statistics and figures.

The 7 templates (shown here for the 30 km/h sign):

```
T1  bare name          speed limit 30 km/h
T2  baseline           a photo of a speed limit 30 km/h
T3  domain             a photo of a speed limit 30 km/h, a type of traffic sign
T4  functional         a road sign indicating speed limit 30 km/h
T5  blurry/context     a blurry photo of a speed limit 30 km/h sign on a road
T6  verbose            a photograph of a speed limit 30 km/h, showing its visual
                       characteristics and function as a traffic sign
T7  irrelevant detail  a photo of a speed limit 30 km/h, taken on a sunny day
```

T2 is the standard "a photo of a ..." prompt used in the original CLIP paper, so we used it as the baseline. All 7 templates were decided in our proposal before we ran anything, and each one was run once on the same images. The model, the images, the class names, the metric and the seed were the same for every template.

For the statistics we used:

- 95% bootstrap confidence intervals (1000 resamples, seed 0), including one for the gap itself
- McNemar tests comparing each template to T2 on the same images
- A confidence check: average confidence vs accuracy, expected calibration error, and the Spearman correlation between confidence and accuracy across templates

What we reused: the CLIP model weights, the GTSRB data, and the usual libraries (PyTorch, NumPy, pandas, SciPy, matplotlib, Hugging Face transformers).

Who did what: we chose the classes, their names, the templates and the baseline ourselves in our proposal. We each implemented a solution, compared them, picked the best one and improved it. Claude (see "Use of AI tools" below) helped write and debug the code, including the data loading and checks, the scoring loop, the statistics, the confidence analysis and the figures. We reviewed the code, ran it ourselves, and checked the numbers against the files in `results/`.

One small change: instead of using `get_image_features` and `get_text_features` from transformers, we call CLIP's encoders and projection layers directly. Those functions return different things in different versions of transformers, which broke our first run on Colab. The result is the same embeddings.

* * *

## Output files (in results/)

```
template_accuracy.csv           accuracy per template with CIs, difference from T2,
                                McNemar p values, confidence and calibration numbers
spread_summary.json             the gap between best and worst template and its CI,
                                plus baselines and the confidence summary
per_class_accuracy.csv          accuracy for every class under every template
confusion_best_template.csv     confusion matrix for T1
confusion_worst_template.csv    confusion matrix for T5
data_source.txt                 which copy of the dataset was used
fig1_template_accuracy.png      accuracy per template
fig2_per_class_heatmap.png      per class accuracy heatmap
fig3_confusion_best_worst.png   confusion matrices for best and worst template
fig4_confidence_vs_accuracy.png confidence vs accuracy
```

Running the script also writes `image_features.npz` and `text_features.npz` (caches of the embeddings) into the output folder. They are regenerated on every fresh run and are not committed.

* * *

## Reproducibility

The seed is fixed at 0. CLIP gives the same output for the same inputs, and the only random part is the bootstrap resampling, which uses the seed. A second full run into a new `results/` folder reproduced our first run to four decimal places. Running from a clean copy of the repo should give the same numbers as `results/` and the report.

We ran everything on Google Colab with Python 3.13 and a Tesla T4 GPU.

* * *

## Use of AI tools

We used Claude (by Anthropic) as our only AI tool. It helped us:

- give feedback on our proposal and the experiment design (confidence intervals, balanced accuracy next to overall accuracy, reporting the spread across templates);
- write and debug the experiment code, including fixes for problems we hit on Colab (a transformers version change and the dataset server timing out) and clearer figure labels;
- help interpret the results and figures, and help draft the README, the slides and the report.

We ran all the experiments ourselves and both of us checked the numbers against the files in this repository. We take responsibility for all code and claims and can each explain every part. The same declaration is in the report.

* * *

## Reference

Stallkamp, J., Schlipsing, M., Salmen, J., Igel, C. (2012). Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition. Neural Networks. (GTSRB.)

Radford, A. et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. (CLIP.)
