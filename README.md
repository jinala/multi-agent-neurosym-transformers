## Neurosymbolic Transformers for Multi-Agent Communication

# Dependencies 

Python 3

OpenAI Gym 0.11.0

PyTorch

Deep graph library (https://www.dgl.ai/)


# Video illustration for the various tasks:

Please find a short video at tasks_video.mp4 to get a better understanding of the tasks used in this  paper.  

# Pre-trained models:

You can find the pre-trained models for all the tasks in main/results/ folder.

random_cross task -> main/results/model_random_cross/

random_grid task -> main/results/model_random_grid/

unlabeled_goals task -> main/results/model_unlabeled/

two_groups_cross in Figure 1 -> main/results/model_2groups_cross/

### Generating plots:

```Shell
  #To generate plots for Figures 2 and 6
  # Random cross
  python3 -m main.plot_results --result_dir=model_random_cross --env=FormationTorch-v0
  # Random grid
  python3 -m main.plot_results --result_dir=model_random_grid --env=FormationTorch-v0
  # Random cross
  python3 -m main.plot_results --result_dir=model_unlabeled --env=UnlabeledGoals-v0
  ```

The above cmds should create plots at main/results/RESULT_DIR/plots/


### Simulating pre-trained models:

```Shell
   python3 -m main.replay  --result_dir=RESULT_DIR --baseline=BASELINE
  ```

  RESULT_DIR can be model_random_cross, model_random_grid, model_unlabeled, model_2groups_cross. 

  BASELINE can be tf-full, hard, dist, prog, prog-retrained, dt, dt-retrained, det, det-retrained. 

  Note that prog-retrained is the version corresponding to our full approach. 


# Training models from scratch 

To train models from scratch, run the following scripts. (Note that this part takes several hours and requires GPUs)

```Shell
  # Random cross
  bash run_random_cross.sh
  # Random grid
  bash run_random_grid.sh
  # Random cross
  bash run_unlabeled_goals.sh
  ```
