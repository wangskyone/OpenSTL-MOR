import torch
import cv2
# from openstl.models import Mod_PredFormer_Model
# from openstl.models.PredFormer_FacTS import Mod_PredFormer_Model
# from openstl.models.PredFormer_FullAttention import Patch_Predformer_Model
from openstl.models.PredFormer_Quadruplet_TSST import Mod_PredFormer_Model
from .base_method import Base_method
import torch.nn.functional as F
import numpy as np

class PredFormer_method(Base_method):


    def __init__(self, **args):
        super().__init__(**args)

    def _build_model(self, **args):
        return Mod_PredFormer_Model(self.hparams.model_config)
    
    def diff_div_reg(self, pred_y, batch_y, tau=0.1, eps=1e-12):
        B, T, C = pred_y.shape[:3]
        if T <= 2:  return 0
        gap_pred_y = (pred_y[:, 1:] - pred_y[:, :-1]).reshape(B, T-1, -1)
        gap_batch_y = (batch_y[:, 1:] - batch_y[:, :-1]).reshape(B, T-1, -1)
        softmax_gap_p = F.softmax(gap_pred_y / tau, -1)
        softmax_gap_b = F.softmax(gap_batch_y / tau, -1)
        loss_gap = softmax_gap_p * \
            torch.log(softmax_gap_p / (softmax_gap_b + eps) + eps)
        return loss_gap.mean()
    
    def router_loss(self, batch_x, router_logits, patch_size,eps=1e-12):
        ndepth = len(router_logits)
        taus = np.linspace(1.0, 0.2, ndepth) # 分层温度控制（从1.0降到0.1）
        B, T, C = batch_x.shape[:3]
        if T <= 2:  return 0
        gap_batch = batch_x[:, 1:] - batch_x[:, :-1]
       
        # === 2. 通道聚合 (运动强度) ===
        motion = torch.norm(gap_batch, dim=2)         # [B, T-1, H, W]
        # === 3. 时间平均（获得整体运动显著性） ===
        motion = motion.mean(1)                       # [B, H, W]
        # 保证H,W能整除patch_size
        ph, pw = patch_size, patch_size
        motion_patch = motion.unfold(1, ph, ph).unfold(2, pw, pw)  # [B, H/ph, W/pw, ph, pw]
        motion_patch = motion_patch.contiguous().view(B, -1, ph*pw)  # [B, N, ph*pw]
        motion_patch = motion_patch.mean(-1)                        # [B, N]
        # === 4. 展平为 token 维度 ===
        motion = motion_patch.reshape(B, 1, -1)                # [B, N]
        # === 5. softmax 得到运动先验分布 ===
        motion_prior = F.softmax(motion / 0.2, dim=-1).detach()   # [B, N]
        total_loss = 0.0
        for i in range(ndepth):
            logits_i = router_logits[i]   # [B, T, N]
            N = logits_i.shape[1]
            # 移除最后那一维（通常是 1）
            logits_i = logits_i.squeeze(-1)   # [B*T, N]
            # 恢复成 [B, T, N]
            logits_i = logits_i.view(B, T, N)

            tau = taus[i]
            # === 6. router 分布 ===
            router_p = F.softmax(logits_i / 0.2, dim=-1)   # [B*T, N]
            # === 7. KL 散度 
            # print('motion_prior:', motion_prior.shape, 'router_p:', router_p.shape)
            loss_i = router_p * torch.log((router_p + eps) / (motion_prior + eps) + eps)

            total_loss += loss_i.mean()

        # === 8. 平均所有层 ===
        total_loss = total_loss / ndepth
        return total_loss
    
    def flow_magnitude_batch_cuda(self, batch_x, device_id=0, method="farneback"):
        """
        使用 OpenCV CUDA 计算光流模长（全流程 GPU）
        Args:
            batch_x: torch.Tensor [B, T, C, H, W], 值范围 [0,1] 或 [0,255]
            device_id: int, 指定 GPU 设备 (默认 0)
            method: "tvl1" 或 "farneback"
        Returns:
            flow_mags: torch.Tensor [B, T-1, H, W]
        """
        assert batch_x.ndim == 5, "Expect [B, T, C, H, W]"
        B, T, C, H, W = batch_x.shape

        # 设置 OpenCV CUDA 设备
        cv2.cuda.setDevice(device_id)
        # print(f"[INFO] Using OpenCV CUDA device: {device_id}")

        # 选择算法
        if method.lower() == "tvl1":
            flow_estimator = cv2.cuda_OpticalFlowDual_TVL1.create()
        elif method.lower() == "farneback":
            flow_estimator = cv2.cuda_FarnebackOpticalFlow.create(
                numLevels=5, pyrScale=0.5, fastPyramids=False,
                winSize=15, numIters=3, polyN=5, polySigma=1.2, flags=0)
        else:
            raise ValueError("method 必须是 'tvl1' 或 'farneback'")

        flow_mags = []

        # 将 Tensor 转为 GPU numpy（避免 CPU 拷贝）
        batch_x = (batch_x * 255).clamp(0, 255).byte().permute(0, 1, 3, 4, 2).contiguous()  # [B,T,H,W,C]
        batch_x = batch_x.detach().to("cpu").numpy()  # ⚠️ OpenCV 目前不支持 torch.cuda.Tensor

        for b in range(B):
            mags_b = []
            for t in range(T - 1):
                frame1 = batch_x[b, t]
                frame2 = batch_x[b, t + 1]

                # 灰度化
                if C > 1:
                    frame1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
                    frame2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
                else:
                    frame1 = frame1.squeeze(-1)
                    frame2 = frame2.squeeze(-1)

                # 上传到 GPU（指定设备）
                gpu_prev = cv2.cuda_GpuMat()
                gpu_next = cv2.cuda_GpuMat()
                gpu_prev.upload(frame1)
                gpu_next.upload(frame2)

                # 计算光流（GPU）
                gpu_flow = flow_estimator.calc(gpu_prev, gpu_next, None)

                # 直接在 GPU 上计算模长
                gpu_flow_x = cv2.cuda.split(gpu_flow)[0]
                gpu_flow_y = cv2.cuda.split(gpu_flow)[1]
                gpu_mag, _ = cv2.cuda.cartToPolar(gpu_flow_x, gpu_flow_y, angleInDegrees=False)

                # 下载结果到 CPU (最后一步)
                mag = gpu_mag.download()
                mags_b.append(mag)

            flow_mags.append(np.stack(mags_b, axis=0))  # [T-1,H,W]

        flow_mags = torch.from_numpy(np.stack(flow_mags, axis=0)).float().to(f"cuda:{device_id}")
        return flow_mags

    
    def flow_router_loss(self, batch_x, router_logits, patch_size,eps=1e-12):
        ndepth = len(router_logits)
        taus = np.linspace(1.0, 0.2, ndepth) # 分层温度控制（从1.0降到0.1）
        B, T, C = batch_x.shape[:3]
        if T <= 2:  return 0
        # motion = self.flow_magnitude_batch(batch_x)  # [B, T-1, H, W]
        # motion = self.torch_flow_mag_raft(batch_x, fp16=False)  # [B, T-1, H, W]
        motion = self.flow_magnitude_batch_cuda(batch_x, device_id=batch_x.device.index if batch_x.device.type=='cuda' else 0, method="farneback")  # [B, T-1, H, W]
        # === 3. 时间平均（获得整体运动显著性） ===
        motion = motion.mean(1)                       # [B, H, W]
        # 保证H,W能整除patch_size
        ph, pw = patch_size, patch_size
        motion_patch = motion.unfold(1, ph, ph).unfold(2, pw, pw)  # [B, H/ph, W/pw, ph, pw]
        motion_patch = motion_patch.contiguous().view(B, -1, ph*pw)  # [B, N, ph*pw]
        motion_patch = motion_patch.mean(-1)                        # [B, N]
        # === 4. 展平为 token 维度 ===
        motion = motion_patch.reshape(B, 1, -1)                # [B, N]
        # === 5. softmax 得到运动先验分布 ===
        motion_prior = F.softmax(motion, dim=-1).detach()   # [B, N]
        total_loss = 0.0
        for i in range(ndepth):
            logits_i = router_logits[i]   # [B, T, N]
            N = logits_i.shape[1]
            # 移除最后那一维（通常是 1）
            logits_i = logits_i.squeeze(-1)   # [B*T, N]
            # 恢复成 [B, T, N]
            logits_i = logits_i.view(B, T, N)

            tau = taus[i]
            # === 6. router 分布 ===
            router_p = F.softmax(logits_i / 0.2, dim=-1)   # [B*T, N]
            # === 7. KL 散度 
            # print('motion_prior:', motion_prior.shape, 'router_p:', router_p.shape)
            loss_i = router_p * torch.log((router_p + eps) / (motion_prior + eps) + eps)

            total_loss += loss_i.mean()

        # === 8. 平均所有层 ===
        total_loss = total_loss / ndepth
        return total_loss


    def forward(self, batch_x, batch_y=None, **kwargs):
        pre_seq_length, aft_seq_length = self.hparams.pre_seq_length, self.hparams.aft_seq_length
        if aft_seq_length == pre_seq_length:
            pred_y, router_logits = self.model(batch_x)
        elif aft_seq_length < pre_seq_length:
            pred_y, router_logits = self.model(batch_x)
            pred_y = pred_y[:, :aft_seq_length]
        elif aft_seq_length > pre_seq_length:
            pred_y = []
            router_logits = None
            d = aft_seq_length // pre_seq_length
            m = aft_seq_length % pre_seq_length
            
            cur_seq = batch_x.clone()
            for _ in range(d):
                cur_seq, logits = self.model(cur_seq)
                if router_logits is None:
                    router_logits = logits
                pred_y.append(cur_seq)


            if m != 0:
                cur_seq, logits = self.model(cur_seq)
                pred_y.append(cur_seq[:, :m])

            
            pred_y = torch.cat(pred_y, dim=1)

        return pred_y, router_logits
    
    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        pred_y, router_logits = self(batch_x)
        loss_mse = self.criterion(pred_y, batch_y)
        loss_flow = self.flow_router_loss(batch_x, router_logits, self.hparams.model_config['patch_size'])
        loss_router = self.router_loss(batch_x, router_logits, self.hparams.model_config['patch_size'])

        loss = loss_mse + 2 * loss_flow + loss_router
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('loss_mse', loss_mse, on_step=True, on_epoch=True, prog_bar=True)
        self.log('loss_flow', loss_flow, on_step=True, on_epoch=True, prog_bar=True)
        self.log('loss_router', loss_router, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        pred_y,_ = self(batch_x, batch_y)
        loss = self.criterion(pred_y, batch_y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=False)
        return loss
    
    def test_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        pred_y,_ = self(batch_x, batch_y)
        outputs = {'inputs': batch_x.cpu().numpy(), 'preds': pred_y.cpu().numpy(), 'trues': batch_y.cpu().numpy()}
        self.test_outputs.append(outputs)
        return outputs