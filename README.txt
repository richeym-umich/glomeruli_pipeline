Created: Mar 30, 2021
by: Greg Teichert

This Python code takes an image of a nephrectomy slice
and performs the following operations:

- Coarsens the full resolution image by 35% along each dimension
- Performs object detection to identify glomeruli,
  using the torchvision library
- Crops out the individual glomeruli into their own images
- Performs image segmentation on the cropped images, using the fast.ai library,
  and saves only the identified glomerular tuft

########################
# Installation
########################

Note the following modules loaded on greatlakes: 
1) intel/18.0.5   2) openmpi/3.1.6   3) python3.7-anaconda/2020.02   4) cuda/11.2.1   5) cudnn/11.2-v8.1.0

The necessary conda environment and Python packages can be created/installed as follows:

conda create -n glomeruli python=3.6
conda activate glomeruli
conda install -c pytorch -c fastai pytorch=1.7.0 torchvision=0.8.1 fastai=1.0.61
pip install opencv-python

The workflow also requires the pyvips library. On macOS or Linux, it can be installed with conda:

conda install --channel conda-forge pyvips

On Windows, there are a couple more steps. The libvips executable can be downloaded
from https://libvips.github.io/libvips/install.html and the pyvips package can be installed with pip:

pip install --user pyvips

Additionally for Windows, the location of the libvips must be added to the system PATH,
either through the system settings or within the python script. See the documentation
at https://pypi.org/project/pyvips/ for additional help.


########################
# Instructions for use:
########################

- The saved model files for the detection and segmentation networks
  should be placed in the 'models' directory.
  + The detection network model filename has the format 'model{model_number}_epoch{epoch_number}'
  + The segmentation network model filename is 'export.pkl'

- The full resolution nephrectomy slice image should be placed in the 'input' directory
  + The image filename should have the format '{image_name}.tif'
  + If the input image is not in the 'input' directory, you can specify the full path of the directory (see next point)

- The code is run through the following command:

  python predict.py {image_name}

  where {image_number} is replaced by the filename of the image, excluding the '.tif' extension

  or using the command

  python predict.py {image_name} {input_dir}

  where {input_dir} is replaced by the full path to the directory that contains the input image.
  (Note that the .npy file defining the reference color histogram should always be in the 'input' directory.)

- The coarsened image, file listing the bounding box of each detected glomerulus,
  folder containing the cropped glumerulus images, and the folder containing the
  segmented glomerular tuft images are save in the 'output' folder.

- The slurm submission script 'submit_predict.slrm' can be used to submit on an HPC cluster
  (with the appropriate accounts, queue, etc. This script is for greatlakes at U. of Michigan).

- If the prediction script is to be run on several images, the script 'predict_many.py' can be used
  to quickly submit separate jobs for each image to the queue. Within 'predict_many.py', modify the
  list of input image numbers/names. You can also specify the input directory (it is expected that
  all input images will be in the same directory). Running 'predict_many.py' on the command line will
  write and submit the job script for each image.

- The script toXML.py can be used to convert the detections_{img_num}.txt predictions to an .xml file
  that can be opened in Aperio Imagescope (modify the image number in toXML.py).
