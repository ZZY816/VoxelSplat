# Train, Test and Evaluation

Train VoxelSplat using 8 GPUs. You can modify `shell/train.sh` if you want to use larger backbone configurations:
```
sh shell/train.sh
```

Test VoxelSplat with 8 GPUs. Predictions will be saved to `preds.gz`.
If you want to output the occupancy prediction for each sample, set `save_sample` to True in the config file:
```
sh shell/test.sh
```

Evaluate VoxelSplat’s performance. Make sure the paths to your predictions and ground-truths are correctly specified:
```
sh shell/evaluate.sh 
```