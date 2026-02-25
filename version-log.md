[v2.0](#v20-interactive-inference--workspace-optimization)

# v2.0 Interactive Inference & Workspace Optimization

This update transitions the prediction workflow into an interactive session and cleans up the repository environment for better version control.

### Key Features:

#### Interactive Prediction Shell:

* **predict.py**: Replaced the one-off command-line argument system with a persistent `while` loop.
* **Session Persistence**: The model and weights are loaded into memory once, allowing users to test multiple images sequentially without restarting the script.
* **Graceful Interaction**: Added support for 'exit' and 'quit' commands to terminate the session safely.

#### Workspace & Git Optimization:

* **Refined .gitignore**: Updated the project exclusion rules to maintain a clean repository.
    * **Weights & Models**: Automatically ignores large `.pth` and `.pt` files to prevent accidental uploads of heavy model parameters.
    * **Data Management**: Excludes the `/data` directory and common dataset formats (e.g., CIFAR-10).

#### Enhanced Error Handling:

* **Path Validation**: The system now checks for file existence before attempting to load an image, providing clear error messages instead of crashing.
* **Format Compatibility**: Added an automatic `.convert('RGB')` step during image loading to ensure PNGs with alpha channels or grayscale images are compatible with the model's 3-channel input requirement.

#### Model Consistency:

* **Standardized Preprocessing**: Ensured that the interactive `predict.py` uses the exact same normalization constants `(0.4914, 0.4822, 0.4465)` and resizing `(32x32)` used during the training phase for maximum accuracy.