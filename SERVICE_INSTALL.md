# Latent Vectors Service
## About
This Python script is a wrapper to the PySlowfast set of models that will output the latent vectors given a set of input videos and the name of a specific PySlowfast model. This repo is meant for those who don’t need the predictions of the PySlowfast models but want an easy way to extract rich latent vectors for large numbers of input videos.

## Installation
### Dependencies
- Follow the instructions on the PySlowfast INSTALL.md page to download the dependencies in the list marked “Requirements”.
- Download the following dependencies:
  - yaml
  ```
  pip install PyYAML
  ```
  - pathlib
  ```
  pip install pathlib
  ```
  
## Clone this repo
- Clone the forked PySlowfast repo:
```
git clone https://github.com/sherylm77/SlowFast.git
```

## Usage

### Setup
- Add this repository to $PYTHONPATH.
```
export PYTHONPATH=/path/to/SlowFast/slowfast:$PYTHONPATH
```
- Build PySlowFast by running
```
cd SlowFast
python setup.py build develop
```
- Download the desired model(s):
  - Go to MODEL_ZOO.md
  - Identify the row in the table corresponding to the model you want. Click the link under the model column to download the model and place the file in the same directory as Slowfast. Note the name of the config file listed under the config column.
- Put the directory containing input videos in the Slowfast folder.
- If using one of the SSV2 models:
  - In the directory containing input videos, add a file containing labels (don't have to be true labels) for each of the input videos. An example label file is given as examples/inputs/something-something-v2-validation.json. The label file must have this name.

### Commands
- Command template:
```
cd SlowFast
python get_latent_vectors.py [input_videos_directory_name] [model_config_name]
```
- For example, to run using the Kinetics Slowfast 8x8 model, run the following command after cd'ing to the SlowFast folder
```
python get_latent_vectors.py "input/siq_videos" "Kinetics\c2\SLOWFAST_8x8_R50"
```

## Examples
In the SlowFast/examples/ folder, you will find examples of input videos, labels, output vectors, etc.
- examples/inputs: directory containing 4 Social IQ videos
- examples/inputs/something-something-v2-validation.json: label file containing labels for each video in examples/inputs
- examples/sample_outputs: npy files containing latent vectors for each video in examples/inputs

## Output
After running a command (as in the Commands section), you can find a folder called ```output_vecs``` in the folder containing input videos. There should be one npy file for each input video.

For example, your inputs might look like:
```
SlowFast
|_ examples
|  |_ inputs
|  |  |_ vid1.mp4
|  |  |_ vid2.mp4
|  |  |_ vid3.mp4
|  |  |_ something-something-v2-validation.json
```

Then you could run the following command:
```
python get_latent_vectors.py "examples/inputs" "SSv2/SLOWFAST_16x8_R50_multigrid"
```

Your outputs would look like:
```
SlowFast
|_ output_vecs
|  |_ output_latent_vec_vid1.npy
|  |_ output_latent_vec_vid2.npy
|  |_ output_latent_vec_vid3.npy
```
