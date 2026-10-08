# Colab Training Guide

Google Colab is a free, web-based Python environment that provides access to powerful GPUs. It allows you to train your models much faster than on a regular laptop, though it has time limits and can disconnect if left idle.

## Steps to Train on Colab

1. **Prepare your Google Account**
   Make sure you are logged into a Google account and open [Google Colab](https://colab.research.google.com).

2. **Upload the Project Zip to Google Drive**
   - Open your Google Drive.
   - Create a new folder named `MargDrishti_Colab`.
   - Upload the `margdrishti_colab.zip` file (which you generated in the project root) into this folder. The first upload may take a few minutes depending on your internet connection.

3. **Upload the Notebook**
   - In Google Colab, go to **File -> Upload notebook**.
   - Select and upload `notebooks/colab_training.ipynb` from your laptop.

4. **Enable the GPU**
   - In Colab, go to the top menu: **Runtime -> Change runtime type**.
   - Under Hardware accelerator, select **T4 GPU** and click **Save**.

5. **Run the Notebook Cells**
   - Run each cell one by one by clicking the play button on the left of the cell.
   - You will see output below the cell indicating its progress.
   - When asked to allow Drive access, follow the pop-up instructions to connect your Google Drive.

6. **Keep the Session Alive**
   - Keep the browser tab open and ensure your computer does not go to sleep.
   - Free Colab sessions can disconnect after some hours or if idle. 
   - **If it disconnects**: Reconnect to Colab, run steps 1 to 5 again, and then run the training cell again. Finished runs will be skipped automatically, and an unfinished run will continue from `last.pt`.

7. **How to Know a Run Finished**
   - At the end of each run, a block starting with `===== NOTE THIS FOR PPT =====` will be printed in the output.
   - A `summary.json` file will appear in your Drive inside the corresponding run folder.

8. **Download Results**
   - The final cell of the notebook will zip the results and trigger a download of `margdrishti_results_custom.zip`.
   - Alternatively, you can manually download the `experiments` folder from `MyDrive/MargDrishti_Colab/experiments` on Google Drive.
   - Save the zip file in the root directory of your MargDrishti project on your laptop.

9. **Second Colab Trip (ResNet50)**
   - After completing the custom CNN ablation study locally, you will train ResNet50 on Colab.
   - Run `python -m scripts.pack_for_colab` locally to rebuild `margdrishti_colab.zip` with the new code.
   - Re-upload `margdrishti_colab.zip` and the updated `notebooks/colab_training.ipynb` to Google Drive/Colab.
   - Run the new ResNet50 cells at the bottom of the notebook. Note: The first run will download the ResNet50 weights from torchvision (about 100 MB).
   - Once finished, the final cell will download `margdrishti_results_resnet.zip` to your laptop. Save it in the project root.

## Fairness Note
*The training time measured on Colab is on a cloud GPU and is not directly comparable to training time on your laptop's CPU. Always record the device per run and show it next to the training time in reports. Inference speed (how fast the model predicts) will be measured separately on your laptop, ensuring that comparison stays fair.*

## Common Problems & Fixes

- **No GPU Available**: Colab's free GPUs are subject to availability. Try again later, use a different Google account, or use Kaggle notebooks (which also provide free GPUs) as an alternative.
- **Out of Memory (OOM)**: Reduce the batch size in the configuration file if the GPU runs out of memory.
- **Drive Quota Exceeded**: Make sure you have enough free space on your Google Drive.
- **Session Lost**: If your session drops, simply reconnect, run the setup cells, and rerun the training cell to resume.
- **Import Errors**: Ensure you have uploaded the correct and complete `margdrishti_colab.zip`.

- **Import Errors**: Ensure you have uploaded the correct and complete `margdrishti_colab.zip`.
