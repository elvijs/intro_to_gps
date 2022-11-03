# intro_to_gps

Welcome! This repo contains some accompanying material for 
an introductory talk on Gaussian Processes.

See the accompanying whiteboard notes in `Whiteboard.pdf`.


## Setup

1. Clone this repo
2. Create and activate a virtual environment using your preferred method.
    - For example, `python3 -m venv myvenv && source myvenv/bin/activate`
3. `make install`
4. Profit!
     - Run `make test` to check the install has been successful. 

## Notebooks

Run `make run_jupyter_server` to get the Jupyter NB up and running.
Note that the files are in the `notebooks` folder.

## Code

* `src/data` contains data loader convenience functions for a few toy datasets. 
    - Data in `<repo_root>/data` folder.
* `src/models` contains a linear regression and a GP model. They're not very good. 
