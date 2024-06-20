
#!/usr/bin/env python3

import torch
from torch.utils.data import Dataset as AbstractDataset
from torch.utils.data import DataLoader
from unstruct_navigation.train import SampleGenerator
from torch.optim import Adam, SGD
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.distributed import init_process_group, destroy_process_group
import os

from tensorboardX import SummaryWriter
import time
import datetime
import pickle
import os
from unstruct_navigation.train import TrainDir
from unstruct_navigation.models import DynPredModel
from unstruct_navigation.models import EvidentialLossSumOfSquares, EvidentialRegression
# def ddp_setup():
#     init_process_group(backend="nccl")
#     torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

class Trainer:
    def __init__(
        self,
        dir: TrainDir,
        model: torch.nn.Module,
        train_data: DataLoader,                
        save_every: int,
        snapshot_path: str,        
    ) -> None:
        if hasattr(os.environ,"LOCAL_RANK"):        
            self.gpu_id = int(os.environ["LOCAL_RANK"])
        else:
            self.gpu_id = 0
        self.dir = dir
     
        self.optimizer = Adam(model.parameters(), lr=0.005)
        # self.loss = EvidentialRegression()
        
        # for p in self.loss.parameters():
        #     p.requires_grad = False

        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_dir = f"{self.dir.train_log_dir}/single_process_{current_time}"
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.model = model.to(self.gpu_id)   
        self.train_data = train_data                             
        
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = os.path.join(self.dir.snapshot_dir, snapshot_path)     
     
        if os.path.exists(self.snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(self.snapshot_path)
            
        if hasattr(os.environ,"LOCAL_RANK"):        
            self.model = DDP(self.model, device_ids=[self.gpu_id])
     

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        # self.model.load_state_dict(snapshot["MODEL_STATE"])
        # self.likelihood = snapshot["Liklihood"]
        # self.model = snapshot["Norm_stat"]        
        # self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")


    def _plot_model_performance_to_tensorboard(self,epoch, mu, sigma, ground_truth_pred_pose_residuals):        
        # writer.add_histogram('Normal Distribution', data, global_step=0)
        ground_truth_pred_pose_residuals = ground_truth_pred_pose_residuals.cpu()
        mu = mu.cpu()
        mean_error = (mu- ground_truth_pred_pose_residuals)        
        unpack_temporal_mean_error = mean_error.view(-1,9,mean_error.shape[1])        
        # unpack_temporal_mean_error = mean_error.view(mean_error.shape[0],9,-1)
        # writer.add_histogram('Normal Distribution', data, global_step=0)                    
        for sequence_idx in range(unpack_temporal_mean_error.shape[1]):
            for feature_idx in range(unpack_temporal_mean_error.shape[2]):
                sequence_feature_data = unpack_temporal_mean_error[:, sequence_idx, feature_idx]
                self.writer.add_histogram(f'Feature_{feature_idx}/Sequence_{sequence_idx}', sequence_feature_data, global_step=epoch)
        if sigma is not None:
            unpack_temporal_sigma = sigma.view(-1,9,sigma.shape[1])
            for sequence_idx in range(unpack_temporal_sigma.shape[1]):
                for feature_idx in range(unpack_temporal_sigma.shape[2]):
                    sequence_feature_data = unpack_temporal_sigma[:, sequence_idx, feature_idx]
                    self.writer.add_histogram(f'Sigma_{feature_idx}/Sequence_{sequence_idx}', sequence_feature_data, global_step=epoch)

 
    def _run_batch(self, source, targets,epoch):
        

        (init_del_xy, n_state, n_actions, rgb, grid, grid_center, valid_corresp, uv_corresp, img_features_grid)   = source                      
      
        self.model.train()       
        self.optimizer.zero_grad()
    
        outs = self.model(init_del_xy.cuda(), n_state.cuda(), n_actions.cuda(), grid.cuda(), grid_center.cuda(), img_features_grid.cuda(), valid_corresp.cuda()) 
        # outs = torch.stack(outs).reshape(4,-1)
        # outs = outs.permute(1,0)
        outs = torch.stack(outs).permute(1,0,2)
        gt_targets = targets[:,:outs.shape[1],:].cuda()     
               

        mse_loss = torch.nn.MSELoss()

        # Calculate the MSE loss
        loss = mse_loss(outs, gt_targets)

        
        # loss = EvidentialRegression(outs, gt_targets)   
        loss.backward()
        self.optimizer.step()                        
        self.writer.add_scalar("loss_total", float(loss), epoch)               
          
        for name, param in self.model.named_parameters():            
            self.writer.add_histogram(name + '/grad', param.grad, global_step=epoch)         
            if 'weight' in name:
                self.writer.add_histogram(name + '/weight', param, epoch)
            if 'bias' in name:
                self.writer.add_histogram(name + '/bias', param, epoch)
     
        # with torch.no_grad():                    
            # mx = gpoutput.mean.cpu()
            # std = gpoutput.stddev.detach()                        
            # self._plot_model_performance_to_tensorboard(epoch, mx, std, gt)
        return loss.item()

    def _run_epoch(self, epoch):
        b_sz = next(iter(self.train_data))[0][0].shape[0]
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        if hasattr(os.environ,"LOCAL_RANK"):
            self.train_data.sampler.set_epoch(epoch)
        total_loss = 0.0
        count = 0
        for source, targets in self.train_data:                            
            loss_bath = self._run_batch(source, targets,epoch)
            total_loss+=loss_bath
            count+=1
            print(f" Current batch count = {count} | Epoch {epoch} | Batchsize: {b_sz} | BATCH LOSS: {loss_bath:.6f}")


        avg_loss_non_torch = total_loss / (count+1)        
        self.writer.add_scalar('LTATT Loss/Train', avg_loss_non_torch, epoch + 1)        
        print(f" Epoch {epoch} | Batchsize: {b_sz} | AVG_LOSS: {avg_loss_non_torch:.6f}")
        

 

    def _save_snapshot(self, epoch):        
    
        if hasattr(os.environ,"LOCAL_RANK"):
            snapshot = {
                "MODEL_STATE": self.model.module.state_dict(),
                "EPOCHS_RUN": epoch,
            }
        else:
            snapshot = {
                "MODEL_STATE": self.model.state_dict(),
                "EPOCHS_RUN": epoch
            }
        
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:     
                self._save_snapshot(epoch)
                


def load_train_objs(args, dir):
    
    
    from unstruct_navigation.cfg import ExperimentParams, RosFeatureExtractorNodeParams
    from omegaconf import OmegaConf, read_write
    params = OmegaConf.structured(ExperimentParams)
    dirs = [dir.train_dir] 
    preprocessed_dataset_load = False
    if preprocessed_dataset_load:
        preprocessed_data_path = os.path.join(dir.preprocessed_dir, f'preprocessed_data_669.pth')                    
        data_path = preprocessed_data_path         
    else:
        data_path = None
    sampGen = SampleGenerator(dir, dirs, data_path = data_path, args = args)
    train_set = sampGen.get_dataset()

    
    with open(dir.normalizing_const_file, 'rb') as pickle_file:            
            norm_dict = pickle.load(pickle_file)

    model = DynPredModel(cfg= params.model.dyn_pred_cfg)
    model.set_normalizing_constatnt(norm_dict)
    

    return train_set, model
    
    # return train_set, model, optimizer, liklihood


def prepare_dataloader(dataset: AbstractDataset, batch_size: int):
    if hasattr(os.environ,"LOCAL_RANK"):
        dataloader = DataLoader(dataset,batch_size=batch_size,pin_memory=True,shuffle=False,sampler=DistributedSampler(dataset))
    else:
        dataloader = DataLoader(dataset,batch_size=batch_size,pin_memory=True,shuffle=True)
    return dataloader


def main(save_every: int, total_epochs: int, batch_size: int):
    # ddp_setup()
    args = {'input_grid_width':153,
            'input_grid_height':128,
            'n_time_step':10, 
            'lstm_hidden_size': 20,              
            'input_state_dim':5, # [vx, vy, wz, roll, pitch] 
            'input_action_dim':2, # [ax, delta]  
            'conv_to_feedforwad_dim': 20,               
            'batch_size':2,
            'num_epochs': 2,
            'output_residual_dim': 5, # [delx, dely, del_vx, del_vy, del_wz]
            'distributed_train': False,
            'arnp_train': True,
            'auclstm_output_dim': 5,
            'auclstm_out_fc_hidden_size': 28,
            'param_yaml_path': '/home/offroad/stego_ws/src/unstruct_navigation/unstruct_navigation_ros/config/default.yaml'
            }
    
    snapshot_path = 'unstruc_snapshot.pth'
    dir = TrainDir()
    
    dataset, model= load_train_objs(args, dir)
    train_data = prepare_dataloader(dataset, batch_size)    
    trainer = Trainer(dir, model, train_data, save_every, snapshot_path)
    trainer.train(total_epochs)
    

if __name__ == "__main__":
    save_every = 50
    total_epochs = 10000
    batch_size = 160
    main(save_every, total_epochs, batch_size)