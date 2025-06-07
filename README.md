## BotTrans Official Code Repo

### Running settings

#### Env Config

```python
torch==1.8.1+cu102
torch-cluster==1.5.9
torch-geometric==2.0.2
torch-scatter==2.0.9
torch-sparse==0.6.12
```

We have also tested under other settings, like torch 1.11.0+cu113&pyg2.4.0. The right combination of these two key packages can run the code normally

#### Structure

`/data`:   data of 10 domains . In each domain i, 'i/raw' is raw data (node feature, edge, graphlet), and 'i/processed' is graph object

`/log`: running logs, the running results will be written into a .log file at this dir.

`/graphlet`: an independent package where we offer our implement for graphlet kernels

`dataloader.py`: organize the data training batches into dataloader

`model.py`: model file

`train.py`: main entry, starts the training and testing

`utils.py`: help functions



#### Commander

```bash
python train.py --src_name A,B,C  --tgt_name D
```

Multisource adaptation from A,B,C to D, for example:

```
python train.py  --src_name 2,3,4,5  --tgt_name 6
```

Other settings are kept as default (e.g.: --cuda 0 --epoch 1000  --repeat 5).