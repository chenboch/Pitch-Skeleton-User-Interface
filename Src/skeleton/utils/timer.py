import time


# class FPSTimer:
#     """A simple timer."""
#     def __init__(self):
#         self.total_time = 0.
#         self.calls = 0
#         self.start_time = 0.
#         self.diff = 0.

#     def tic(self):
#         # using time.time instead of time.clock because time time.clock
#         # does not normalize for multithreading
#         self.start_time = time.time()

#     def toc(self, average=True):
#         self.calls += 1
#         self.diff = time.time() - self.start_time
#         self.total_time += self.diff
#         if average:
#             return self.total_time / self.calls
#         else:
#             return self.diff

#     def clear(self):
#         self.total_time = 0.
#         self.calls = 0
#         self.start_time = 0.
#         self.diff = 0.


import time

class Timer:
    def __init__(self, duration):
        self.duration = duration
        self.start_time = None
    
    def start(self):
        self.start_time = time.time()
    
    def is_time_up(self):
        if self.start_time is None:
            return False
        diff_time = time.time() - self.start_time
        if (diff_time) < self.duration:
            return False
        return True
    
    def get_remaining_time(self):
        """返回剩餘的倒數時間"""
        if self.start_time is None:
            return self.duration
        elapsed_time = time.time() - self.start_time
        remaining_time = self.duration - elapsed_time
        return max(0, int(remaining_time)) 
    
    def reset(self):
        self.start_time = None

import time
import torch


def time_synchronized():
    """
    獲取同步時間（適用於 GPU 運算）。
    Returns:
        float: 當前時間（秒）。
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


class FPSTimer:
    """
    用於計算程式碼執行時間和 FPS（每秒幀數）的計時器類別。
    """
    def __init__(self):
        self.start_time = 0.0
        self.end_time = 0.0

    def tic(self):
        """
        計時開始。
        """
        self.start_time = time_synchronized()

    def toc(self):
        """
        計時結束。
        """
        self.end_time = time_synchronized()

    @property
    def time_interval(self):
        """
        獲取兩次計時之間的時間間隔（秒）。
        Returns:
            float: 執行時間（秒）。
        """
        return self.end_time - self.start_time

    @property
    def fps(self):
        """
        計算每秒幀數 (FPS)。
        Returns:
            float: FPS 值。
        """
        return round(1.0 / max(self.time_interval, 1e-10), 2)
