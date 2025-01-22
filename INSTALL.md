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

    
3. Install dependencies

    ```
    pip install -r requirements.txt
    ```

    
4. Install [ExifTool](https://exiftool.org/) for extracting Exif metadata from raw images
   
   Install ExifTool by following the [official installation guide](https://exiftool.org/install.html) on how to install it on the Windows, MacOS or Unix platforms.
   
   Below, fyi, some helpful tips are provided:

   - DualDn is tested with ExifTool version 12.84, so any version newer than 12.84 should work as well.
   - Check if ExifTool is correctly installed by running the following command:
      ```
      exiftool -ver
      ```
      If successful, it should return the version number such as `12.84`, if not, the installation is failed.

   - On Windows platforms, ensure that ExifTool is added to system's **GLOBAL PATH** as well as the compiler's (e.g. in VSCode) **PATH**.
   - On Unix platforms (which are typically used as remote servers), after unzipping the `Image-ExifTool-12.84.tar.gz` file, you can add the following lines to '.bashrc' file for automated configuration:
      ```
      export PERL5LIB=/'the path to'/Image-ExifTool-12.84/install/share/perl5
      export PATH=/'the path to'/Image-ExifTool-12.84/install/bin:$PATH
      ```    
