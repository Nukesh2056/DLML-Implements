# Lubricant Image-Based Degradation Analysis – Executive Summary  
We review and recommend a comprehensive computer-vision and machine-learning pipeline for lubricant condition monitoring from images.  **Image preprocessing** steps include noise filtering (e.g. Gaussian, median, bilateral) to remove speckle or sensor noise; color‐space conversion (RGB→HSV/Lab) to separate chromatic content; contrast enhancement (e.g. histogram equalization or CLAHE) to reveal subtle color changes; intensity normalization and illumination correction (e.g. flat-field or homomorphic filtering) to compensate for uneven lighting.  **Segmentation** can then isolate oil regions or particles using methods ranging from simple thresholding (for uniform backgrounds) to edge-based (Canny/Sobel), region-growing, watershed (with markers), or learned approaches (U-Net or Mask R-CNN).  We present a flowchart of a typical segmentation pipeline (see section on **Image Segmentation**).  

Next, we extract both **visual features** (e.g. color histograms/moments in RGB, Lab, HSV【40†L91-L94】; texture descriptors like GLCM/Haralick, LBP, Gabor filters; shape/morphology of detected particles – e.g. area, perimeter, circularity【40†L181-L189】; translucency or turbidity metrics) and **sensor-derived features** if available (e.g. multi-spectral bands, IR/UV absorbance, fluorescence response, dielectric constant, refractive index, or viscosity proxies).  We map each feature to expected sensitivity: for example, color tends to darken with oxidation (increasing TAN) or soot; texture features spike with particulate contamination (wear debris, soot); specular highlights and translucency drop with water or emulsions; refractive/absorption shifts with fuel dilution or additive depletion【40†L181-L189】【44†L84-L90】.  A summary table maps features to degradation effects and extraction methods (see **Feature Extraction**).  

For modeling, we distinguish **classification** vs **regression**: classification predicts discrete condition states (e.g. Class 1/2/3 for “good–moderate–bad” oil) while regression predicts a continuous degradation index or remaining useful life.  As Rahman *et al.* note, “the supervised task can be formulated as classification or a regression problem. Classification provides discrete output… regression provides a continuous numerical value”【50†L271-L279】.  Classification is used when categorization is needed (and labeled classes exist) – e.g. determine if oil is “fresh”, “worn”, or “failed”.  Regression is used if a continuous measure (like an Oil Degradation Index) is desired.  Common algorithms include **SVMs, Random Forests (RF), XGBoost** or similar tree/ensemble methods for structured features, as well as **CNNs or transfer‐learned deep nets** for raw images.  Ensembles (e.g. voting among RF and SVM) can improve robustness.  We recommend experimenting with SVM (RBF/poly kernels), RF/XGBoost (varying tree count and depth), and CNNs (e.g. ResNet or EfficientNet pre-trained on ImageNet) fine-tuned on oil images.  

Training strategy should use appropriate **loss functions**: cross-entropy for classification (optionally with class weights if imbalanced) and MSE/MAE for regression.  **Evaluation metrics** include accuracy, precision/recall/F1 and confusion matrices for classification, plus Cohen’s κ for chance-corrected agreement; for regression, use RMSE, MAE and R².  ROC or PR curves can help set class thresholds on continuous scores.  

We briefly summarize **lubricant chemistry**: common degradation processes include *oxidation* (forming acids, water, carbonyls), *thermal cracking* (soot, viscosity rise), *additive depletion*, *contamination* (water, fuel, particles), etc【18†L170-L178】【20†L50-L57】.  Typical lab measurements are viscosity, TAN/TBN, FTIR (carbonyl and nitration peaks), UV-Vis (optical absorbance, colorimetry), fluorescence, and particle counts.  Many of these have visual proxies: for example, oxidation/darkening increases oil color index and FTIR carbonyl absorbance【44†L84-L90】, water or soot cause turbidity/milky appearance【40†L173-L178】, wear particles appear as dark specks (countable objects)【15†L47-L53】, and fluorescence under UV can indicate fuel or additive changes.  Table **X** maps chemical changes to measurable properties and potential camera-detectable cues.  

Finally, we outline a **practical pipeline**: collect oil samples under controlled lighting and standardized containers (uniform white/diffuse illumination to avoid reflections); calibrate the camera with color/gray cards; label samples via laboratory tests (assign classes or indices); augment images (rotations, flips, jitter) to enlarge data; use k-fold cross-validation; and calibrate camera features against reference sensors.  Sensor fusion is possible (e.g. combining camera data with a simple UV/IR or turbidity sensor).  We suggest experiments with dozens to hundreds of labeled samples per class to train robust models.  Deployment must consider real-time illumination consistency and periodic re-calibration against lab benchmarks.  

**Recommended Features & Models:** Use a combination of color (RGB/HSV histograms or CNN color channels) and texture (GLCM/Haralick or CNN features) descriptors, plus explicit particle counts or shape descriptors【40†L181-L189】.  A baseline might be a RF or SVM on hand-crafted features, while a CNN (e.g. fine-tuned ResNet) can be a powerful alternative if labeled images are sufficient.  Hyperparameters to try include SVM kernel γ and C, RF/XGBoost tree depth and learning rate, and CNN learning rate and batch size.  Evaluate with accuracy/F1 (for classification) or RMSE/R² (for regression), and set class-thresholds to balance precision/recall (e.g. using ROC analysis).  

**Table 1** compares common preprocessing and segmentation techniques. **Table 2** (below) maps key features to degradation sensitivity and extraction. A flowchart (Mermaid diagram) outlines a typical segmentation pipeline.  This report synthesizes both literature and practical guidelines for image-based lubricant monitoring, providing a detailed reference for researchers and engineers.

## 1. Image Processing Techniques  
**Denoising:**  Lubricant images (often of fluid surfaces or in cuvettes) can be noisy due to sensor or lighting.  Common filters include *Gaussian blur* (low-pass) which smooths noise but may blur edges (kernel σ tuned to noise level), *median filter* (good for salt‑and‑pepper noise, preserves edges), and *bilateral filter* (smoothes while preserving edges by combining spatial and intensity kernels).  More advanced denoisers (e.g. non-local means or BM3D) could be used if noise is severe.  In practice, start with a 3×3 or 5×5 median/Gaussian; adjust size until speckle is reduced without over-blurring oil boundaries or particulates.  

**Color Space Transforms:**  After denoising, convert images to alternative color spaces.  RGB is intuitive but can mix intensity/color.  *Lab* separates *L* (luminance) from *a,b* (chromatic), which helps for color analysis (e.g. detecting brownish oxidation) independent of lighting.  *HSV* (hue–saturation–value) isolates hue (useful if oil color hue shifts) and saturation (oil may desaturate with dilution).  For many tasks, compute features in multiple spaces (e.g. RGB histograms plus Lab moments)【40†L91-L94】.  Converting to grayscale is also useful for texture or shape segmentation (focus on intensity).  

**Contrast Enhancement:**  Oils can have low contrast or small subtle differences; adaptive methods help.  *Histogram Equalization (HE)* globally spreads intensities but may amplify noise.  *Contrast Limited Adaptive HE (CLAHE)* adjusts small blocks, enhancing local contrast without over-amplifying noise【12†L671-L679】.  For example, CLAHE is known to improve visibility of fine details in low-contrast imagery【12†L671-L679】.  Parameters like clip-limit (~2–4) and tile-size (8×8 or 16×16) control enhancement; tune so that background noise is not magnified.  In harsh cases, combined denoise+CLAHE (e.g. TV-CLAHE, which adds total-variation denoising) can sharpen features while suppressing noise【12†L687-L694】.  

**Normalization & Illumination Correction:**  Normalize each image (e.g. subtract mean, divide by std) when feeding into ML models.  For illumination gradients, use techniques like *rolling ball* or morphological opening to estimate background illumination and subtract it.  Alternatively, use flat-field correction: image a uniform reference (white sheet) and use it to flatten each sample image.  Ensuring uniform white balance across captures is critical; include a color chart in the scene if possible for calibration.  In summary, the preprocessing pipeline might be: **Raw image → Denoise (Gaussian/median) → Color convert (HSV/Lab) → Contrast (CLAHE) → Illumination correction → Normalization**.  

## 2. Image Segmentation Methods  
Segmenting the oil, droplets, or wear particles can isolate regions of interest for feature extraction. Key methods include:

- **Thresholding:**  Simple global or adaptive thresholding on intensity or a single channel (e.g. Hue) is fast.  Use Otsu’s method or fixed thresholds if illumination is controlled.  Pros: trivial implementation.  Cons: fails under uneven lighting or low contrast.  Use when oil background is uniform (e.g. oil on white paper) and color difference is clear.

- **Edge-based (Canny/Sobel):**  Detect edges of oil droplets or particles via gradient filters.  Pros: finds sharp boundaries; Cons: produces disconnected edges/noise if image is poor.  Best for well-lit samples with clear boundaries.

- **Region Growing / Flood Fill:**  Starting from seed points (manually or automatically detected), expand regions by similarity (color/intensity).  Pros: can follow complex shapes.  Cons: requires seed selection, sensitive to inhomogeneity.  Use for segmenting blobs with similar intensity (e.g. water droplets in oil).

- **Watershed (Marker-based):**  Treat intensity as topography; flood from markers to segment basins.  Pros: good for separating touching objects if proper markers given.  Cons: over-segmentation without markers; sensitive to noise.  Often combined with distance transform on a binary mask, then watershed to split blobs.

- **Active Contours (Snakes) / Level Sets:**  Evolve contours from initial guesses to object boundaries using image gradients.  Pros: flexible shapes, sub-pixel accuracy.  Cons: requires good initialization and parameter tuning (e.g. smoothness weight), can be slow.  Useful for delineating irregular oil–air interfaces or oil film edges.

- **Graph Cuts:**  Model segmentation as global energy minimization on a pixel graph.  Pros: optimal under certain cost functions, handles textures.  Cons: computationally expensive for large images.  Use in combination with user-supplied markers or EM-based intensities for complex scenes.

- **Superpixels (e.g. SLIC):**  Pre-segment image into homogeneous patches, then classify patches.  Pros: reduces complexity, preserves object boundaries.  Cons: adds another step of classification.  Use when objects have complex textures; great input to graph-based or CNN methods.

- **Deep Learning (U-Net, Mask R-CNN):**  When sufficient labeled data exists, convolutional neural nets excel at segmentation.  U-Net (or variants) produces pixel masks; Mask R-CNN gives instance masks plus bounding boxes.  Pros: handles complex patterns, learns features end-to-end.  Cons: data-hungry and compute-intensive.  Use for end-to-end segmentation of debris or oil spills, especially if many annotated examples are available or can be synthesized.

**Table 1: Segmentation Methods Comparison**

| Method             | Pros                                             | Cons                                           | Typical Use                                 |
|--------------------|--------------------------------------------------|------------------------------------------------|---------------------------------------------|
| Thresholding       | Very fast; few parameters                        | Sensitive to lighting; one-phase only         | Uniform background; coarse oil/non-oil mask |
| Edge-based (Canny) | Detects precise boundaries                        | Noisy edges; may need post-processing         | Sharp object boundaries, simple shapes      |
| Region Growing     | Grows arbitrarily shaped regions                  | Needs seeds; can leak if texture varies       | Isolating droplets with uniform interior    |
| Watershed          | Separates touching objects well (with markers)    | Oversegmentation; noise-sensitive             | Splitting clustered particles               |
| Active Contours    | Smooth, flexible boundaries                       | Initialization needed; iterative; slow        | Refining outlines of oil droplets or spills |
| Graph Cuts         | Globally optimal under energy model               | Compute-heavy; needs cost model               | Complex textured scenes with learned costs  |
| Superpixels (SLIC) | Reduces pixel count; preserves edges              | Requires feature classification afterward     | Preprocessing for object segmentation       |
| **Deep (U-Net/Mask R-CNN)**  | Learns complex features; handles textures     | Requires large labeled set; GPU compute       | High-accuracy oil/contaminant segmentation  |

```mermaid
flowchart LR
    A[Raw Image] --> B[Preprocess (denoise, color, contrast)]
    B --> C{Choose Segmentation Method}
    C --> Th[Thresholding/Clustering]
    C --> Ed[Edge-based (Canny/Sobel)]
    C --> Rg[Region Growing / Flood Fill]
    C --> W[Watershed (with markers)]
    C --> Ac[Active Contours / Level Set]
    C --> G[Graph Cuts]
    C --> Sp[Superpixels (SLIC)]
    C --> DL[Deep Learning (U-Net, Mask R-CNN)]
    Th & Ed & Rg & W & Ac & G & Sp & DL --> D[Segmentation Mask / Contours]
    D --> E[Post-processing (refine, remove small)]
    E --> F[Extracted Regions of Interest]
```

*(Figure: Example segmentation pipeline. Start with preprocessing; choose method(s) based on image; combine outputs; refine mask for ROI.)*

## 3. Feature Extraction  
After segmentation, extract features that correlate with oil condition. **Color features**: compute histograms or color moments (mean, variance, skewness) in RGB, HSV or Lab【40†L91-L94】.  For instance, color histograms of the R/G/B or H/S channels capture overall oil tint (oxidation often shifts oil toward brown/yellow)【40†L91-L94】.  **Texture features**: use Gray-Level Co-occurrence Matrices (GLCM) to derive Haralick metrics (contrast, entropy, homogeneity), or Local Binary Patterns (LBP) for micro-texture; Gabor or wavelet transforms can detect periodic or multi-scale patterns in oil (e.g. sediment or emulsions).  **Morphological features**: for any detected particles or droplets, compute their area, perimeter, circularity, aspect ratio, etc【40†L181-L189】.  (For example, Hepworth *et al.* extracted *Area, Perimeter, Circularity* to characterize cells in liquid【40†L181-L189】.)  **Particle counts/distribution**: count segmented wear debris or soot particles; features include count, mean size, and size histogram (particle counting is a standard wear analysis metric【15†L47-L53】).  

**Translucency/Turbidity metrics**: measure overall image sharpness or scattering; e.g. the variance of intensity or the “blurriness” can indicate emulsified water or soot.  Specular highlights (bright spots) can indicate surface reflectance differences (thick fresh oil vs thin/contaminated oil).  Measuring the drop of intensity in backlit setups can give turbidity (OD measurement).  Gordillo *et al.* showed that turbidity grade strongly alters oil color and appearance【40†L173-L178】, so texture measures (variance, wavelet energy) can quantify turbidity.  

**Sensor-derived features**: if multispectral or UV/IR imaging is available, record those values (e.g. narrow-band absorption).  Viscosity and dielectric: usually measured by separate sensors (e.g. viscometer or permittivity probe) but if a flow or rotational camera setup exists, one could estimate viscosity by motion blur or flow pattern.  Fluorescence: shine UV and measure emission intensity (changes with additives or fuel contamination).  In essence, any ancillary sensor provides extra features for fusion.  For example, fiber-optic IR absorption sensors target oil oxidation peaks【15†L47-L53】【18†L158-L162】.  

**Feature-to-Degradation Sensitivity:** Table 2 summarizes typical features and their expected sensitivity. (High/Medium/Low sensitivity indicates how strongly a degradation factor influences the feature.)  For instance, overall color histograms strongly track oxidation (oil darkens) and soot levels【28†L43-L52】, while Haralick texture or particle count are very sensitive to solid contamination.  Translucency metrics track water/emulsions.  Sensor features like IR carbonyl absorption or TAN from FTIR are direct chemical proxies【18†L170-L178】【44†L84-L90】.  

| **Feature**               | **Extraction**          | **Sensitive to Degradation Factors**                                 |
|---------------------------|-------------------------|---------------------------------------------------------------------|
| Color histogram/moments (RGB, Lab, HSV)【40†L91-L94】 | Compute channel histograms, mean/var. | High: oxidation (color darkens), fuel dilution (color lightens)【28†L43-L52】; medium: additive changes. |
| Texture (GLCM/Haralick, LBP) | GLCM features; LBP patterns. | High: particle contamination (wear debris, soot), turbidity (emulsions)【40†L173-L178】; medium: oil microstructure. |
| Gabor/Wavelet features    | Apply filters at various orientations/scales. | Medium: periodic textures, heavy contamination layers. |
| Morphology (area, shape)【40†L181-L189】 | Segment objects; measure shape stats. | High: discrete particles (wear debris, water droplets); low: bulk fluid (none). |
| Particle count/size       | Threshold and label particles. | High: metal wear particles, soot. Quantifies wear severity. |
| Translucency/Opacity      | Measure intensity variance or backlight transmission. | High: emulsified water or fine particulates (makes oil milky); medium: oxidation slightly darkens. |
| Specular highlights       | Detect bright spots (threshold/highlights). | Medium: oil film thickness and refractive index changes (thin, clear oil has stronger highlight). |
| Turbidity (scattering)    | Image blurriness or scattering metrics. | High: water contamination or micro-dispersions【40†L173-L178】; indicates oil cleanliness. |
| **Sensor IR/UV** (absorbance at specific λ) | Use IR/UV camera or spectrometer. | High: oxidation/carbonyl peaks (IR), fuel dilution (UV), additive presence (UV fluorescence). See ASTM D7214-20. |
| **Fluorescence (UV light)** | UV lamp + camera. | Medium: fuel dilution (fuels often fluoresce), anti-wear additive depletion (some fluoresce). |
| **Viscosity (proxy)**      | Flow rate or motion blur estimation. | Medium: thickening (oxidation, soot) slows flow; can calibrate against viscometer. |
| **Dielectric/Refraction**  | (Non-visual) sensors. | High: water increases dielectric constant; fuel/water change refractive index (affect specular angle). |

*Table 2: Key image and sensor features, extraction methods, and sensitivity to lubrication degradation.*


## 4. Classification vs Regression; Modeling Strategies  
In supervised modeling, use **classification** if the goal is to assign discrete oil-condition categories (e.g. *Class1=fresh, Class2=used, Class3=expired*).  Use **regression** if you want a continuous *degradation index* or predicted lab value (like TAN).  As noted in lubricant ML literature, “the supervised task can be formulated as a classification or regression problem. Classification provides discrete-valued output… regression provides a continuous numerical value”【50†L271-L279】.  For example, Rodrigues *et al.* built a neural model to *classify* engine oil into usage classes【24†L65-L73】, whereas others predict remaining useful life as a continuous metric.  

**Algorithms:**  For classification, common choices include Support Vector Machines (with RBF/poly kernels), Random Forests, gradient-boosted trees (XGBoost/LightGBM), and Neural Networks/CNNs. For regression, the same models can be used (with e.g. linear or tree regressors).  CNNs are especially suited if raw images are inputs; one can fine-tune a pre-trained network (ResNet, EfficientNet, etc.) by replacing the top layers. Transfer learning often yields good results with limited data.  Ensemble methods (e.g. stacking RF and SVM) can improve generalization.  

**Training and Loss:**  Use cross-entropy loss for classification (with appropriate class weights if classes are imbalanced).  For regression of a degradation index, use MSE or MAE loss.  Regularization (dropout, L2) is important for neural nets.  Consider techniques like k-fold cross-validation or bootstrapping to robustly estimate performance.  

**Evaluation Metrics:**  - *Classification:* report **accuracy**, **precision/recall/F1** per class, and the **confusion matrix**.  ROC-AUC or PR-AUC curves can help choose decision thresholds.  Cohen’s kappa can be used for agreement beyond chance.  - *Regression:* use **RMSE** (root mean square error), **MAE** (mean absolute error) and coefficient of determination **R²** to assess fit.  Use residual plots and possibly Bland–Altman analysis to check biases.  

Given 3 classes (C1/C2/C3), one can either directly predict class with softmax or predict a continuous index and threshold it.  In the latter case, choose thresholds based on maximizing F1 or kappa between predicted and true classes.  **Decision Thresholds:** e.g. if oil degradation index > *T1* then Class3; between *T1* and *T2* then Class2, else Class1.  Use validation data to tune *T1,T2* via grid search and measure confusion matrix.  

*Algorithms to Try:* SVM (kernel RBF with C∈[0.1,100], γ∈[1e-3,1]), RF (n_estimators 100–500, max_depth 5–20), XGBoost (learning_rate 0.01–0.3, depth 3–10), CNN (ResNet/ResNext with frozen convolutional base, tune LR =1e-4–1e-2).  Ensembles (e.g. RF+SVM voting) can be evaluated.  See **Table 3** for example hyperparameter ranges.  

## 5. Lubricant Degradation Chemistry & Visual Cues  
Oil degrades via **oxidation** (O₂ attack forms peroxides, acids, sludge), **thermal stress** (breaks bonds, forms soot or varnish), **additive depletion**, and **contamination** (water, fuel, dirt).  These processes alter chemical properties: viscosity tends to rise (from polymer/soot) or fall (fuel dilution), total acid number (TAN) increases (from oxidation)【18†L170-L178】【44†L84-L90】, additives vanish, water content rises, and wear debris accumulates.  Common tests detect these changes: viscosity measurements, TAN/TBN titrations, **FTIR** spectroscopy (carbonyl peak ~1700 cm⁻¹ for oxidation, nitration peaks)【44†L84-L90】, UV-Vis spectroscopy (darkening; oxidation products absorb UV-blue light), and fluorescence (fuel or anti-oxidant additives often fluoresce).  **Particle analysis** (ISO 4406 codes) counts metal/silica debris; water by Karl Fischer or dielectric.  

**Visual/Sensor Cues:**  Many chemical effects have optical signatures.  Oxidation and soot darken oil, increasing its ASTM color index【28†L43-L52】.  ASTM D1500 notes that “color is an important quality characteristic” and deviations often signal contamination or breakdown, although color alone isn’t definitive【28†L43-L52】.  Water contamination (especially emulsified) makes oil milky or opaque, raising turbidity【40†L173-L178】.  Fuel dilution can lighten and clarify the oil.  Iron/copper wear particles appear as dark specks on bright backgrounds (camera-based particle counters detect these【15†L47-L53】).  Acid increases are not directly visible, but often correlate with color or particle content.  Table 4 maps chemical indicators to visual measures.  

| **Degradation Process**  | **Measured Property**            | **Visual/Sensor Cue**                                           |
|--------------------------|---------------------------------|-----------------------------------------------------------------|
| Oxidation                | Carbonyl FTIR peak; TAN↑【44†L84-L90】  | Oil darkening (brown/yellow)【28†L43-L52】; increased UV-Vis absorbance; possible fluorescence quenching.  |
| Thermal breakdown (soot) | Viscosity↑; insolubles         | Black particles/turbidity in oil; opaque appearance.            |
| Additive depletion       | TBN↓; specific IR peaks        | Subtle color shift; loss of fluorescence (if additive was fluorescent). |
| Contamination – *water*  | Karl Fisher; dielectric↑      | Milky/emulsified look; swirl patterns (immiscible layers); reduced specular reflection. |
| Contamination – *fuel*   | GC analysis; viscosity↓       | Oil appears thinner/clearer; slight change in refractive index; possible strong fluorescence under UV. |
| Wear particles (metal)   | Particle count (ISO codes)    | Visible dark flecks; can be segmented by vision algorithms【15†L47-L53】.             |
| Soot                      | Filter Patch Test (Ruler); IR soot signatures  | Very dark, gritty oil; residue on filter.                        |
| Additive (e.g. antioxidants) depletion | IR band loss (e.g. phosphate peaks) | Little direct optical cue; often inferred from other changes.       |

*Table 4: Lubricant degradation modes with measurable properties and corresponding visual/sensor cues.*  

Taken together, these cues (color, opacity, particles) form the basis for camera-based condition assessment.  For example, the photonic oil sensor by López *et al.* directly measures oil oxidation index and acid number via optical absorbance【18†L158-L162】.  Similarly, a colorimetric sensor can detect oil “yellowing” as oxidation proceeds.  

## 6. Practical Data Pipeline and Experiment Design  
**Imaging setup:** Acquire images in a controlled environment.  Use a uniform background and consistent diffuse lighting (e.g. ring light or lightbox) to minimize shadows and specular glare.  Ideally use a light source covering visible spectrum and fix camera white-balance to a reference.  Capture oil samples in identical glass cuvettes or bottles to control path length.  Include a color calibration target in preliminary images to normalize for lighting variations.  

**Sample preparation:** Ensure samples are free of air bubbles (they cause bright spots) by pre-wetting containers or degassing.  Stir or homogenize emulsified samples for consistency.  Maintain a constant temperature, as viscosity and appearance can change with heat.  

**Labeling strategy:** For supervised learning, each image must be labeled.  We recommend obtaining **laboratory reference data** for each sample (e.g. TAN, viscosity, particle count, etc.) and/or expert classification.  If using discrete classes (e.g. C1–C3), have domain experts label each sample by condition or use lab thresholds (e.g. TAN>X = class3).  If possible, use an independent test set from a different batch or machine.  

**Data augmentation:** Artificially expand data via image transforms that preserve class: small rotations/flips, intensity scaling, Gaussian blur, and slight color jitter (to simulate lighting variation).  Avoid augmentations that alter oil appearance semantically (e.g. do not flip if text or orientation matters).  Augmenting with synthetic turbidity or particle overlays might help generalization.  

**Cross-validation & Calibration:** Use k-fold cross-validation (e.g. k=5) to estimate performance.  When setting class thresholds on a regression output, use a validation split to optimize F1/kappa.  Calibrate image features to lab readings via regression analysis – for example, perform a linear fit between mean image color and measured TAN.  Use statistical methods (Bland–Altman plots) to check for systematic biases.  

**Sensor fusion:** If additional simple sensors are available (e.g. an IR photodiode, tuning-fork water sensor), combine their readings with image features.  For instance, include the oil’s dielectric constant or fluorescence reading as extra input features to the ML model.  Multi-modal fusion often improves robustness (e.g. vision + smell/e-nose).  

**Recommended experiments:** Gather at least tens of samples per class for classification, or a wide range of degradation levels for regression.  Stratify samples by age, usage, or lab value to cover the spectrum.  Consider time-series: take images periodically as oil degrades to capture progressive changes.  For sample size: aim for ≥50–100 labeled instances per class for initial models; more if variability is high.  Balance classes to avoid bias, or use class-weighting if imbalanced.  

**Deployment considerations:** In-field use may require a sealed imaging chamber or fixed-mount camera on machinery.  Automate background and lighting shields to ensure repeatability.  Plan periodic recalibration (e.g. photograph a reference sample) to correct for camera drift or lighting changes.  The model should be tested in the actual deployment environment against fresh data.  

## 7. Recommended Features, Models, Hyperparameters & Evaluation Plan  
Based on the above, we recommend the following as a starting point:  

- **Feature set:** Combine global color descriptors (RGB/HSV histograms or CNN color-feature maps) with texture descriptors (GLCM contrast, entropy, LBP) and simple morphology (particle count, area).  If possible, use CNN features (e.g. intermediate activations from a fine-tuned ResNet) to capture complex patterns.  Include any sensor readings (UV absorbance, IR, dielectric) as additional features.  

- **Model architectures:** For hand-crafted features, try **Random Forest** and **SVM**.  For image-based, use a **CNN** (e.g. ResNet-50) with transfer learning.  An ensemble of RF and SVM, or voting between RF and CNN, can be robust.  

- **Hyperparameters:**  Example grid: SVM with RBF kernel (C ∈ {1,10,100}, γ∈{0.01,0.1}), RF with n_estimators {100,300,500}, max_depth {None,10,20}, XGBoost with learning_rate {0.01,0.1,0.2}, max_depth {3,6,9}.  For CNN fine-tuning, try learning rates ~1e-4–1e-3, dropout 0.5, batch size 16–32, and early stopping.  Use validation to tune these via grid or Bayesian search.  

- **Evaluation plan:** Use stratified k-fold CV for classification; compute mean accuracy/F1 and confusion matrix.  For regression, use CV to compute RMSE/MAE.  Plot ROC and PR curves (especially if class imbalance).  Define acceptable thresholds: e.g., require ≥90% accuracy on fresh/vs-bad classes and F1≥0.8 for each class.  Set degradation-index cutoffs so that, say, predicted class matches lab-defined class ≥85% of the time.  

In sum, a hybrid approach is advisable: extract multiple feature types, try several algorithms, and rigorously validate performance.  Align model outputs with lab metrics to ensure practical utility.  

**References:** We draw on peer-reviewed studies and standards. For example, color-based oil classification has used RGB/HSV histograms【40†L91-L94】, and sensor methods for oxidation and TAN have been demonstrated in the literature【18†L158-L162】【44†L84-L90】.  CLAHE-based contrast enhancement is a well-known preprocessing step【12†L671-L679】.  Lubricant sensor reviews detail optical and IR measurement techniques【15†L47-L53】.  Lubricant ML reviews confirm the use of SVM, RF, ANN models and classification/regression paradigms【50†L271-L279】【24†L65-L73】. Our guidelines synthesize these findings into a cohesive pipeline. 

