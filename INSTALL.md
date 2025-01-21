This repository is built in PyTorch 2.3.0, training and testing on Python3.9.15, CUDA12.4.

Follow these instructions:

1. Clone our repository

    ```
    git clone https://github.com/OpenImagingLab/DualDn.git
    cd DualDn
    ```

2. Create and activate a new Conda environment

    ```
    conda create -n pytorch_DualDn python=3.9.15
    conda activate pytorch_DualDn
    ```

3. Install [ExifTool](https://exiftool.org/) for extracting Exif metadata from raw images.
   
   You can install ExifTool by following the [official installation guide](https://exiftool.org/install.html) on how to install it in Windows, MacOS or Unix platforms.
   
   Below, I've also provided some helpful tips:

   - We test our code with ExifTool version of 12.84, which means the latest version bigger than 12.84 should work fine too.
   - You can verify if ExifTool is correctly installed by running the following command:
      ```
      exiftool -ver
      ```
      If successful, it should return the version number such as `12.84`, if not, the installation is failed.

   - Ensure that ExifTool is added to your system's **GLOBAL PATH** as well as the compiler's (e.g. in VSCode) **PATH**.
   - On Unix-based platforms, typically used as remote servers, after unzipping the `Image-ExifTool-12.84.tar.gz` file, you can add the following lines to '.bashrc' file for automated configuration:
      ```
      export PERL5LIB=/path/to/Image-ExifTool-12.84/install/share/perl5
      export PATH=/path/to/Image-ExifTool-12.84/install/bin:$PATH
      ```    

5. Install dependencies

    ```
    pip install -r requirements.txt
    ```
