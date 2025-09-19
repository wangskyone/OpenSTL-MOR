import torch

# from openstl.models import SwinLSTM_D_Model, SwinLSTM_B_Model
from openstl.models.morlstm_model import MorLSTM_B_Model
from .base_method import Base_method


class MorLSTM_B(Base_method):
    
    def __init__(self, **args):
       super().__init__(**args)
    
    def _build_model(self, **args):
        return MorLSTM_B_Model(self.hparams)   
    
    def forward(self, batch_x, batch_y, **kwargs):
        """Forward the model"""
        # preprocess
        test_ims = torch.cat([batch_x, batch_y], dim=1).permute(0, 1, 3, 4, 2).contiguous()

        img_gen, _ = self.model(test_ims, return_loss=False)
        pred_y = img_gen[:, -self.hparams.aft_seq_length:].permute(0, 1, 4, 2, 3).contiguous()

        return pred_y      
    
    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        ims = torch.cat([batch_x, batch_y], dim=1).permute(0, 1, 3, 4, 2).contiguous()

        img_gen, loss = self.model(ims)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

# class SwinLSTM_B(MorLSTM_D):
#     def __init__(self, **args):
#        MorLSTM_D.__init__(self, **args) 
    
#     def _build_model(self, **args):
#         return SwinLSTM_B_Model(self.hparams)