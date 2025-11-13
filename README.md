# 🤖 Convolutional Autoencoder for MNIST

This project is a deep learning model built in **PyTorch** to learn a compressed, low-dimensional representation (latent space) of the **MNIST** handwritten digit dataset.

The model is a **Convolutional Autoencoder (CAE)** that takes a 28x28 digit image, compresses it into a 32-dimensional vector, and then reconstructs it back into an image. This demonstrates powerful unsupervised feature learning, dimensionality reduction, and generative capabilities.

* [cite_start]**Code:** `notebooks/autoencoder_mnist.ipynb` [cite: 35-45]
* [cite_start]**Report:** `report/autoencoder_report.docx` [cite: 1-117]

---

## ⚙️ Model Architecture

[cite_start]The model is a symmetric "hourglass" network with an encoder and a decoder[cite: 12].

1.  **Encoder:** Compresses the image using `nn.Conv2d` layers.
    * `Input: [1, 28, 28]`
    * [cite_start]`Layer 1: Conv2d(1, 16)` -> `[16, 14, 14]` [cite: 17-24]
    * [cite_start]`Layer 2: Conv2d(16, 32)` -> `[32, 7, 7]` [cite: 27-32]
    * [cite_start]`Flatten & Linear:` -> `[32]` (This is the latent vector) [cite: 34-38]

2.  **Decoder:** Reconstructs the image from the latent vector using `nn.ConvTranspose2d` layers.
    * `Input: [32]`
    * [cite_start]`Linear & Unflatten:` -> `[32, 7, 7]` [cite: 44-49]
    * [cite_start]`Layer 1: ConvTranspose2d(32, 16)` -> `[16, 14, 14]` [cite: 50-56]
    * [cite_start]`Layer 2: ConvTranspose2d(16, 1)` -> `[1, 28, 28]` [cite: 58-63]
    * [cite_start]`Tanh Output:` (To match normalized input data) [cite: 66-67]

---

## 📊 Training & Results

[cite_start]The model was trained for **20 epochs** using the **Adam** optimizer and **MSE Loss** [cite: 77-80]. [cite_start]The training loss converged to a low value of **0.0168**[cite: 80].

[cite_start] [cite: 91]

### 1. Image Reconstruction
[cite_start]The model successfully learned to reconstruct the digits from the test set with a very low average test loss of **0.0171**[cite: 98]. [cite_start]The reconstructed images are high-quality and retain the key features of the originals [cite: 101-102].

[cite_start] [cite: 103]

### 2. Latent Space Clustering (t-SNE)
To prove the latent space is meaningful, the 32-dimensional vectors from the test set were extracted and visualized using **t-SNE**. [cite_start]The resulting 2D plot shows that the network **instinctively learned to cluster similar digits together** without ever being given the labels [cite: 105-107].

[cite_start] [cite: 107]

### 3. Generative Capabilities
[cite_start]By sampling random vectors from the learned latent space, the decoder can be used as a **generative model** to create new, synthetic digit images that look like the originals [cite: 108-109].

[cite_start] [cite: 109]

---

## ▶️ How to Run This Project

1.  Clone this repository:
    ```bash
    git clone [https://github.com/](https://github.com/)[your-username]/pytorch-autoencoder-mnist.git
    cd pytorch-autoencoder-mnist
    ```

2.  (Optional but recommended) Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

4.  Open the Jupyter Notebook. [cite_start]The MNIST dataset will be downloaded automatically by the script.
    ```bash
    jupyter notebook notebooks/autoencoder_mnist.ipynb
    ```
