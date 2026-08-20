# Evaluation dataset sources

This project does not download data automatically. Obtain only media you are
authorised to use, retain the source terms alongside the local acquisition,
and put no media or private identifiers in Git.

## FaceForensics++

- Official source: <https://github.com/ondyari/FaceForensics> and its
  [dataset documentation](https://github.com/ondyari/FaceForensics/blob/master/dataset/README.md).
- Contains original source videos plus manipulated variants. The documented
  methods include Deepfakes and FaceSwap (face swap), plus Face2Face and
  NeuralTextures (reenactment-style manipulations).
- Access: accept the FaceForensics terms of use and use the project-provided
  download route; do not rely on repackaged copies with unclear provenance.
- Usefulness: gives the first V3 baseline a controlled mix of real and
  face-swap/reenactment fakes. Its subjects and manipulation families may
  overlap historical training data, so record source IDs and do not present it
  as an independent generalisation result unless that overlap is ruled out.

## DeepFake Detection Challenge (DFDC)

- Official source: [Meta DFDC dataset page](https://ai.meta.com/datasets/dfdc/)
  and [Kaggle competition data](https://www.kaggle.com/competitions/deepfake-detection-challenge/data).
- Contains real and fake videos; the supplied metadata identifies `label` and,
  for fake clips, the original source clip. The full collection contains
  face-modification algorithms, but the per-clip manipulation method is not
  generally exposed in the benchmark metadata.
- Access: the Meta route requires an AWS account/IAM credentials; the Kaggle
  route requires accepting the competition rules. The full set is large, so
  begin with an authorised, documented subset rather than downloading it by
  script.
- Usefulness: source-video metadata can establish identity/source linkage and
  provides real face-swap deepfakes. Limitation: clip-level method categories
  may remain `unknown` unless established by authoritative metadata.

## Celeb-DF v2

- Official source: <https://github.com/yuezunli/celeb-deepfakeforensics>.
- Contains Celeb-real originals, additional YouTube-real videos, and
  Celeb-synthesis DeepFake videos.
- Access: the official repository directs users to request access through its
  form; it also specifies its terms of use and citation requirement. Do not
  substitute a third-party mirror.
- Usefulness: a challenging face-swap family for a separate V3 subgroup.
  Limitation: the material is celebrity/web-video based, not a proxy for
  ordinary phone-camera images.

## Provenance-recorded real phone photos

Create this subset only from consented images that your organisation is
authorised to evaluate. Target 100--500 *images*, with different lighting,
indoor/outdoor scenes, cameras, resolutions, backgrounds, glasses, partial
faces, and both single- and multi-person images. Do not put these images,
names, device identifiers, or consent records in the repository.

Set `source_dataset` to `real_phone` in the metadata CSV. Use an opaque
`identity_id` only where consent and local policy allow it; otherwise leave it
blank and the pipeline will correctly report that identity-disjointness cannot
be verified. `provenance` should describe the controlled local collection, and
`usage_note` should record the relevant consent/usage restriction without
including personal data.

## Local layout and metadata

```text
dataset/                         # ignored by Git
  real/                          # benchmark-compatible organisation only
  fake/
    face_swap/
    face_reenactment/
    other_manipulation/
  metadata.csv                   # ignored with the data
```

`metadata.csv` is mandatory for a trustworthy run. The exact columns are:

```csv
relative_path,label,source_dataset,source_id,identity_id,manipulation_type,provenance,usage_note
real/phone_001.jpg,real,real_phone,local-001,,none,controlled-consented-collection,internal evaluation only
fake/face_swap/frame_001.jpg,fake,dfdc,clip-123,actor-42,face_swap,DFDC metadata.json,subject to DFDC terms
```

Labels come only from this file, never from filenames or directory names.
For data derived from videos, retain the source-video ID and source identity
when the official metadata supplies them. Do not include any identity found in
V3 training data; if a comparison training manifest is unavailable, the report
will say that identity-disjointness cannot be verified.

## Commands after authorised acquisition

```powershell
venv\Scripts\python.exe tools\dataset_manifest.py --dataset dataset --out dataset_manifest.csv
venv\Scripts\python.exe tools\check_dataset.py --dataset dataset --manifest dataset_manifest.csv
venv\Scripts\python.exe tools\benchmark_model.py --dataset dataset --out benchmark-results
venv\Scripts\python.exe tools\summarize_benchmark.py --manifest dataset_manifest.csv
venv\Scripts\python.exe tools\analyze_failures.py --manifest dataset_manifest.csv
```

The benchmark command is unchanged. The final two commands only join its
predictions to manifest metadata; they do not alter a V3 score or threshold.
