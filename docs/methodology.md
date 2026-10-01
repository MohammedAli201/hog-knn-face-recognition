# Methodology and limitations

The submitted pipeline detects faces with OpenCV's frontal-face Haar cascade and keeps the largest detection. It crops the face, converts it to grayscale, resizes to 64 × 128, computes Sobel gradients and applies scikit-image HOG to the gradient magnitude. This is the actual submitted descriptor; it should not be described as a conventional HOG-on-grayscale baseline without comparison.

Training uses a seeded 80/20 image split, StandardScaler fitted on the training portion and a five-neighbor brute-force classifier. The split does not establish separation by recording session, participant identity or camera conditions. Similar captures may therefore appear on both sides. The unknown-person rule remains a fixed distance heuristic; it is not a calibrated open-set recognition evaluation.

The source cleanup skips unreadable files, aligns labels with successfully detected faces, supports differently sized source images, uses platform-independent dataset paths and creates the attendance CSV if absent. It also removes the misleading accuracy calculation that scored predictions against themselves. The unchanged original source and full original ZIP are retained for comparison.

This is a historical demonstration, not a validated attendance or identity-verification product. Placeholder names are display labels. New attendance files are runtime outputs and are ignored by Git. Original ZIP reconstruction was verified byte-for-byte; Python syntax was checked. Missing OpenCV/scikit-image prevented runtime evaluation in this cleanup environment.
