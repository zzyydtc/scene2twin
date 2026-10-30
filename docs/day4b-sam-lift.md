# Day 4b: lifting SAM instances to 3D (ETH3D `office`)

Code: `semantics/lift.py` (mask-id assignment, cross-view voting, clustering),
`semantics/run_sam.py` (runner and report). Tests: `tests/test_lift.py` (5 tests; 32 in total pass).
Model: SAM ViT-B (`sam_vit_b_01ec64.pth`, pretrained, not trained here), automatic mask generator,
images downscaled to a 1536 px long side. SAM runs at about 4 to 7 s per image on an RTX 3080 Laptop GPU.

## Method

SAM separates things but does not name them, so this stage yields **instances, not categories**.
1. For every image, keep masks smaller than 25% of the image (larger ones are room surfaces),
   paint them from large to small, and read the id under each SfM observation (the smallest
   containing mask wins).
2. Two points are linked when they share a mask in at least 2 views and in at least 60% of the
   views that see both. Connected components with at least 5 points become instances.
3. The mean link ratio inside an instance is its cross-view agreement.

Voting, linking and clustering are my own code; the masks come from SAM.

## Results (9,484 points, 26 images, one scene)

| Item | Result |
|---|---|
| Points assigned to an instance | 8,317 of 9,484 |
| Instances | 47 |
| Surface-like (under 20% of points labelled object by the plane stage) | 26 instances, 7,679 points, mostly ceiling panels |
| Object-like (50% or more labelled object) | 18 instances, 542 points; the largest has 118 points |
| Mixed | 3 instances, 96 points |
| Mean agreement, object-like / surface-like | 0.883 / 0.907 |
| Adjusted Rand index vs. DBSCAN object clusters | 0.84, on only 533 points present in both |

Visual check (overlays rendered from the output): on a desk view the keyboards, backpack,
whiteboard and monitor bases appear as separate instances; on a ceiling view the ceiling panels
and the light-fixture area appear as large coloured regions. The five largest instances are such
ceiling pieces: thin (0.1 to 0.15 units) and 7 to 15 units wide.

## Problems

- **Chaining.** Three instances (184, 42 and 8 points) span the room in every axis (for example
  6.8 x 9.3 x 9.8 units); single-linkage components merged unrelated objects. The agreement score
  does not reveal this (instance 7 scores 0.86), so it is not a usable quality signal on its own.
  An extent check against the room size would catch these three, but I have not added one.
- **Little object evidence.** Only 542 of 9,484 sparse points belong to object-like instances,
  because the reconstruction concentrates on the ceiling (about 7,000 points). The comparison with
  DBSCAN therefore rests on 533 points and the 0.84 should not be over-read.
- **No categories.** Nothing here says "keyboard" or "chair". That needs a classifier on top of
  the masks, which has not been downloaded or run.
- **No ground truth.** There are no instance annotations for this scene, so quality is judged by
  inspection and by agreement with the plane stage.

## Next

- Reject or split instances whose extent is comparable to the room.
- A dense reconstruction would give far more object points to lift onto.
