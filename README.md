

# SPGen: Speed Profile Generator

=====================================
### A Two-Stage Deep Learning Based Approach for Predicting Instantaneous Vehicle Speed Profiles on Road Networks
=====================================

Welcome to SPGen, a tool for generating speed profiles using deep learning techniques, specifically Long Short-Term Memory networks (LSTMs). This repository showcases how to use the deep learning model for generating instantaneous vehicle speed profiles on road networks.

## Overview
-----------

SPGen is a Python-based tool that utilizes a two-stage deep learning approach to generate vehicle speed profiles on road networks. The model is designed to take into account various factors that affect vehicle speed, such as road geometry and traffic conditions.

## Getting Started
---------------

To get started with SPGen, follow these steps:

1. Clone this repository to your local machine.
2. Open the `main.ipynb` notebook in your favorite Jupyter environment.
3. Follow the step-by-step instructions in the notebook to generate your first speed profiles.


### Example Usage

```python
import pandas as pd
from SPGen import SPGen

# Load the input data (trips, sequences of links, along with their attributes)
df = pd.read_csv('input_data.csv')

# Preprocess the data
df = ... 

# Initiate the first speed value 
in_velocity = ...

# Generate speed profiles
profiles_generator = SPGen.SPGen()
speed_profiles = profiles_generator.generate_speed_profiles_from_pandas(df, in_velocity = in_velocity)

# Visualize the results
plot(speed_profiles)
```

## Paper
------

This project is published in the following paper:

A. Dib, A. Sciarretta and M. Balac, "A Two-Stage Deep Learning Based Approach for Predicting Instantaneous Vehicle Speed Profiles on Road Networks," 2023 IEEE 26th International Conference on Intelligent Transportation Systems (ITSC), Bilbao, Spain, 2023, pp. 1636-1642, doi: 10.1109/ITSC57777.2023.10422212.

## Citation
----------

If you use SPGen in your research, please cite the above paper.

## Contributing
------------

We welcome contributions and collaborations to enhance SPGen's capabilities. Feel free to reach out for further discussions and explorations. If you want a custom model, just prepare the training dataset!

## License
-------

This repository is licensed under the MIT License. See the LICENSE file for details.
