
================================================================================
 Deep Learning - Assignment 5 (Bonus): Image Generation Using Diffusion Models
 Name: Saad Ali
 Roll Number: MSDS25066
 GitHub: https://github.com/Hashimi321/diffusion-animal-generation
================================================================================

--------------------------------------------------------------------------------
1. PROJECT STRUCTURE
--------------------------------------------------------------------------------
Saad_MSDS25066_05/
    MSDS25066_05_train.py       - Data loader, forward diffusion process,
                                   U-Net model (denoising network), custom
                                   loss function, training loop, and the
                                   image-generation (reverse diffusion /
                                   sampling) function.
    MSDS25066_05_test.py        - Loads a saved trained model from disk and
                                   generates a sample image from pure noise.
    MSDS25066_05_allCode.py     - Combined contents of the two files above,
                                   in one file, as required by the submission
                                   instructions.
    test_single_sample.ipynb    - Jupyter notebook that loads the trained
                                   model and displays a generated sample
                                   image inline. Intended for live
                                   demonstration during viva.
    Report.pdf                  - Written report with findings, problems,
                                   solutions, loss graphs, forward-noise
                                   step images, and generated sample images.
    Readme.txt                  - This file.
    saved_models/                - Trained model weight files (.pth):
                                       diffusion_model-50epoch.pth
                                       diffusion_model-200epoch.pth
    Graphs_Visualization/        - Saved result images: loss curves, forward
                                   diffusion noise-step visualizations, and
                                   generated sample images.
    animal_data/                 - NOT included in this submission, per
                                   assignment instructions (do not resubmit
                                   the provided dataset). To run the code,
                                   place the dataset folder (with the same
                                   structure: one subfolder per animal class,
                                   e.g. animal_data/Bear/, animal_data/Cat/,
                                   etc.) in the project root, or pass its
                                   path using --data_path (see below).

--------------------------------------------------------------------------------
2. ENVIRONMENT SETUP
--------------------------------------------------------------------------------
Requires Python 3.x and the following packages (PyTorch only, per assignment
rules - no other deep learning libraries were used):

    pip install torch torchvision pillow matplotlib numpy

NOTE ON HARDWARE: The model was developed and smoke-tested on CPU, then
trained for real (50 and 200 epochs) on a Google Colab GPU (Tesla T4) for
practical training speed. Both MSDS25066_05_train.py and
MSDS25066_05_test.py automatically detect and use a GPU if one is available
(via torch.cuda.is_available()), and fall back to CPU otherwise - no code
changes are needed to switch between environments.

--------------------------------------------------------------------------------
3. HOW TO RUN - TRAINING
--------------------------------------------------------------------------------
From inside the Saad_MSDS25066_05/ folder, with the animal_data/ dataset
folder present (see Section 1):

    python MSDS25066_05_train.py

This runs with sensible defaults: dataset path "animal_data", 50 epochs,
batch size 8, learning rate 1e-4, 20 images per class, across 5 classes
(Bear, Cat, Dog, Lion, Tiger).

The script also accepts command-line arguments to override these defaults:

    python MSDS25066_05_train.py --data_path animal_data --epochs 100 --batch_size 8 --learning_rate 0.0001 --images_per_class 20

Available arguments:
    --data_path         Path to the dataset folder (default: animal_data)
    --epochs             Number of training epochs (default: 50)
    --batch_size         Training batch size (default: 8)
    --learning_rate      Learning rate (default: 0.0001)
    --images_per_class   Number of images to use per class (default: 20)

Running this script will:
    1. Load and filter the dataset (keeping only original images, not the
       cropped/augmented _1, _2, _3 duplicate copies).
    2. Run and save a visualization of the forward diffusion process
       (forward_diffusion_test.png) showing an image becoming pure noise
       across t = 0 to 999.
    3. Build the U-Net model and verify input/output tensor shapes match.
    4. Train the model, printing the average loss after every epoch.
    5. Save the trained model weights to saved_models/diffusion_model.pth.

--------------------------------------------------------------------------------
4. HOW TO RUN - GENERATING A SAMPLE IMAGE FROM A TRAINED MODEL
--------------------------------------------------------------------------------
From inside the Saad_MSDS25066_05/ folder:

    python MSDS25066_05_test.py

This loads the trained model from saved_models/diffusion_model-200epoch.pth
(edit the model_path variable inside MSDS25066_05_test.py to point to a
different saved model file if you want to test a different checkpoint, e.g.
diffusion_model-50epoch.pth), runs the full reverse diffusion / sampling
process starting from pure random noise, and saves the generated image to
Graphs_Visualization/test_output.png.

Note: sampling involves 1000 sequential steps through the model and is
noticeably slower on CPU (roughly 1-2 minutes) than on GPU.

--------------------------------------------------------------------------------
5. HOW TO RUN - test_single_sample.ipynb (FOR VIVA DEMONSTRATION)
--------------------------------------------------------------------------------
Open test_single_sample.ipynb in VS Code or Jupyter, and use "Run All" to
execute every cell in order. The notebook will:
    1. Load the trained model (diffusion_model-200epoch.pth) from
       saved_models/.
    2. Run the reverse diffusion sampling process.
    3. Display the generated image directly inline in the notebook.

NOTE (Windows + Anaconda users): if you encounter a kernel crash with an
"OMP: Error #15" message related to libiomp5md.dll, this is a known conflict
between Anaconda's and PyTorch's bundled OpenMP runtimes on Windows, not a
bug in this code. The notebook's first cell sets the environment variable
KMP_DUPLICATE_LIB_OK=TRUE as a standard workaround for this specific issue
before importing torch.

--------------------------------------------------------------------------------
6. TRAINING ON GOOGLE COLAB (GPU)
--------------------------------------------------------------------------------
The same MSDS25066_05_train.py file was used, unmodified, to train on a
Google Colab GPU runtime (Tesla T4), by:
    1. Cloning the project's GitHub repository into the Colab environment.
    2. Uploading the animal_data dataset to Google Drive, mounting Drive in
       Colab, and unzipping the dataset alongside the cloned code.
    3. Importing the script's functions/classes and calling train_model(...)
       directly with device="cuda", taking advantage of automatic GPU
       detection already built into the script.

--------------------------------------------------------------------------------
7. NOTES ON DESIGN DECISIONS (see Report.pdf for full details)
--------------------------------------------------------------------------------
- The forward diffusion process uses the closed-form formula
  x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise, jumping
  directly to any timestep t in a single calculation rather than looping
  and adding noise step by step.
- The denoising model is a 2-level U-Net (PyTorch only, as required) with
  skip connections between matching down/up stages, GroupNorm and SiLU
  activations, and a sinusoidal time embedding that is projected and added
  into each convolutional block so the network can condition its denoising
  behaviour on how noisy the current input is.
- The loss is a custom (from-scratch) Mean Squared Error between the
  model's predicted noise and the actual injected noise, matching the
  training objective given in the assignment (Algorithm 1).
- Two trained checkpoints are included (50 and 200 epochs) to allow
  comparison of training loss and generation quality at different training
  durations - discussed further in Report.pdf, including an honest
  discussion of the generation quality limitations given the small dataset
  size (100 images across 5 classes) and a compact 2-level architecture.
================================================================================