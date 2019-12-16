# representation-learning-project
Supervising disentangled variational auto encoder (SDVAE)

To run the SDVAE on the dsprites dataset, first download dsprites from [here](https://github.com/deepmind/dsprites-dataset). Pleae modify the dataset classs accordingly to be able to run on your machine.
After choosing the supervised target variable (e.g., 'size' or 'shape' or etc) in the "script_supevision.py", please run the following command make sure you set up the save directories in your script):
```
python script_supevision.py
```
the script will run and save the results. You can then run the script in the 'restore' format and evaluate your results. You should be able to get nice reconstructions and also be able to supervise latent factors like the image below:
![](images/figure_dsprites.png)
