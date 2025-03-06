import math

class OneEuroFilter:
    def __init__(self, dx0=0.0, min_cutoff=0.15, beta=0.3, d_cutoff=1.0):
        """
        初始化 One Euro Filter.

        参数:
            dx0 (float): 初始差分值.
            min_cutoff (float): 最小截止频率.
            beta (float): 截止频率的调节系数.
            d_cutoff (float): 差分的截止频率.
        """
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.dx_prev = float(dx0)

    def __call__(self, x, x_prev):
        """
        执行滤波操作.

        参数:
            x (float): 当前值.
            x_prev (float): 前一个值.

        返回:
            x_hat (float): 滤波后的值.
        """
        if x_prev is None:
            return x

        # 假设时间步长为1秒，根据实际情况可以调整
        t_e = 1

        # 计算差分的平滑因子
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)

        # 计算动态截止频率
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)

        # 计算值的平滑因子
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, x_prev)

        # 更新前一个差分值
        self.dx_prev = dx_hat

        return x_hat

    def smoothing_factor(self, t_e, cutoff):
        """计算平滑因子"""
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        """执行指数平滑"""
        return a * x + (1 - a) * x_prev

import torch
import math

def smoothing_factor(t_e: torch.Tensor, cutoff: torch.Tensor) -> torch.Tensor:
    """計算平滑因子"""
    tau = 1.0 / (2 * torch.pi * cutoff)
    return 1.0 / (1.0 + tau / t_e)


def exponential_smoothing(alpha: torch.Tensor, x: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
    """指數平滑"""
    return alpha * x + (1.0 - alpha) * prev


class OneEuroFilterTorch:
    def __init__(self, dx0=0.0, min_cutoff=0.1, beta=0.3, d_cutoff=0.5, device='cuda'):
        """
        初始化 One Euro Filter（PyTorch 版本）

        Args:
            dx0 (float): 初始差分值
            min_cutoff (float): 最小截止頻率
            beta (float): 截止頻率的調節係數
            d_cutoff (float): 差分的截止頻率
            device (str): 運算設備（'cuda' 或 'cpu'）
        """
        self.min_cutoff = torch.tensor(min_cutoff, dtype=torch.float32, device=device)
        self.beta = torch.tensor(beta, dtype=torch.float32, device=device)
        self.d_cutoff = torch.tensor(d_cutoff, dtype=torch.float32, device=device)
        self.dx_prev = torch.tensor(dx0, dtype=torch.float32, device=device)
        self.device = device
        self.prev_timestamp = None

    def __call__(self, x: torch.Tensor, x_prev: torch.Tensor, timestamp: float = None) -> torch.Tensor:
        """
        執行濾波操作

        Args:
            x (torch.Tensor): 當前值
            x_prev (torch.Tensor): 前一個值
            timestamp (float): 當前時間戳 (可選)

        Returns:
            torch.Tensor: 濾波後的值
        """
        if x_prev is None:
            return x

        # 計算時間步長 t_e
        if self.prev_timestamp is None or timestamp is None:
            t_e = torch.tensor(1.0, dtype=torch.float32, device=self.device)  # 預設為1.0
        else:
            t_e = torch.tensor(timestamp - self.prev_timestamp, dtype=torch.float32, device=self.device)

        self.prev_timestamp = timestamp if timestamp is not None else self.prev_timestamp

        # 差分平滑因子
        a_d = smoothing_factor(t_e, self.d_cutoff)
        dx = (x - x_prev) / t_e
        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)

        # 動態截止頻率
        cutoff = self.min_cutoff + self.beta * torch.abs(dx_hat)

        # 值的平滑因子
        a = smoothing_factor(t_e, cutoff)
        x_hat = exponential_smoothing(a, x, x_prev)

        # 更新前一個差分值
        self.dx_prev = dx_hat

        return x_hat