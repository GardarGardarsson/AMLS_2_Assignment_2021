

# Single Image Super Resolution
## Applied Machine Learning Systems 2
### University College London
#### Department of Electronic and Electrical Engineering
**Student Number:** 20167036

<img src="./Notebooks/imgs/teaser.png" alt="teaser" style="zoom:20%;" />

```python
PSNR: 31.20 dB 
SSIM: 0.95
```

This repository contains the final assignment from student number 20167036 for the *ELEC0135 Applied Machine Learning Systems-II* module. <br>
The project focuses on generating *Super Resolution* (SR) images from *Low Resolution* (LR) ones using deep learning techniques. <br>
More specifically, the project aims at applying an improved version of the [FSRCNN model of Dong et al.](https://arxiv.org/abs/1608.00367) to solve the [NTIRE2017 challenge](https://data.vision.ee.ethz.ch/cvl/ntire17/#challenge).

The datasets used for training and evaluating the models may be accessed from the [DIV2K Dataset Website](https://data.vision.ee.ethz.ch/cvl/DIV2K/) 



## Virtual Environment

A virtual environment yaml file,  `.yml`, is included in the `/Environment` directory. 

To run the complete project, please install the environment using:

```
conda env create -f amls_2_env_sn20167036
```

This file includes all the `conda` and `pip` installed packages needed to run the project. <br> Once setup is complete you can do: 

```
conda env list
```

Or alternatively:

```
conda info --envs
```

... to verify it's installation - i.e. you should see `amls_2_env_sn20167036` in the list. 

Don't forget to activate the environment by running:

```
conda activate amls_2_env_sn20167036
```

Or starting from the `GUI`. If you encounter errors during installation, *before abandoning all hope*, please try opening the `.yml` with a text editor, and fiddle with the `*` wildcard symbol for version constraints. E.g. if you see `numpy 1.5.3` giving you problems try updating it to `numpy 1.5.*` or `numpy 1.*` as there can be issues regarding `os` and package support I won't be able to verify. I've already put pretty extravagant wildcards but mention this just in case things go south.

For more information on managing `conda` environments you can always [refer to the documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file). 



## Navigating the directory

The project directory is structured as followed:

```
Datasets/
  |-> DIV2K_train_HR						# Should be added
      |-> 0000.png
          ...
  |-> DIV2K_train_LR_bicubic		# Could be added to verify functionality, feel free to try any track and scale
      |-> X4
      		|-> 0000x4.png
      		...
  |-> DIV2K_valid_HR						# Should be added regardless of chosen track
  		|-> 0801.png
  				...
 	|-> DIV2K_valid_LR_bicubic 		# Should match chosen track and scale, in this case 'bicubic x4'
 			|-> X4
 					|-> 0801x4.png
 							...  
  |-> Evaluate									# Small sample from the datasest for visual inspection in main.ipybn
      |-> Bicubic
      |-> HR
      |-> Predicted
      |-> Unknown
      
Environment/
	|-> amls_2_env_sn20167036.yml  # The virtual environment needed to run the project, see prev. section

Models/
	|-> ResCNN			# Residual networks weren't heavily researched and not reported
	|-> SRCNN				# The FSRCNN model of Dong et al. however was heavily studies
			|-> Conv-7_Flt- ... -22_bicubic_X2  # Directories for model weights and training losses 
					...
	|-> VDSR				

Modules/
	|-> data_processing.py	# Datahandling etc.
	|-> metrics.py					# PSNR, SSIM for both images and tensors
	|-> model.py						# The SRCNN model class and a ResCNN (which was used to experiment other models)
	|-> user_interface.py   # UI for interacting with the program

Notebooks/
	|-> imgs															# Images for notebooks and this README markdown
	|-> Model Performance Analysis.ipynb	# Sketchbook to generate graphs
  |-> project_notebook_1.ipynb					# Development notebook - getting acquainted with the ISR datahandler
  |-> project_notebook_2.ipynb					# Some transition towards the final main.ipynb...
  |-> project_notebook_3.ipynb					# ... same ...
  
main.ipynb		# The holy-grail, the crown jewel, the alpha, the omega.	
```

This short summary gives the most necessary details, but below are more detailed instructions to get things up and running:



### `./Datasets`

To verify training and running the evaluation on the 100 validation images, you should [get the DIV2K dataset here](https://data.vision.ee.ethz.ch/cvl/DIV2K/), or a subset thereof (e.g. one problem track, `x4 bicubic`) making sure to download the `HR`, `train` and `validation` sets from the link above, e.g. if you go for the `x4 bicubic` you'd get the following `.zip` files from the link above:

 <img src="/Users/gardar/Documents/UCL/ELEC0135 MLS-II Applied Machine Learning Systems 2/Assignments/Final/AMLSII_20-21_SN20167036/Notebooks/imgs/div2k_info.png" alt="div2k_info" style="zoom:35%;" />

Then place them in `./Datasets` and un-zip directly there! <br>**Note!** Once un-zipped, don't place them in a `./DIV2K` folder under `./Datasets/` as the DIV2K website suggests.



## Running the project

If you've gotten this far, i.e. installed the virtual environment and placed some datasets in the appropriate way in the `./Dataset` directory, you should be good to go to run the project!

This would be the `main.ipynb` notebook, based on the root of the project directory !

Based on your chosen problem track, when evaluation on the 100 validation images of the DIV2K dataset, you should witness results similar to the ones reported for `i-FSRCNN` in the table below:

<img src="/Users/gardar/Documents/UCL/ELEC0135 MLS-II Applied Machine Learning Systems 2/Assignments/Final/AMLSII_20-21_SN20167036/Notebooks/imgs/results.png" alt="results" style="zoom:40%;" />



## Tensorboard

To initialise `TensorBoard`:

- Open a terminal from your virtual environment
- In the terminal, navigate to the project directory, i.e. `/AMLSII_20-21_SN20167036`
- In the terminal, type in: `tensorboard --logdir='logs/'`
- Open a web browser, e.g. `Chrome, Firefox` and type in `localhost:6006` in the address bar
- The `TensorBoard` should load

