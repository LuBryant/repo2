import torch
import time

def gpu_warmup():
    """预热GPU，确保CUDA初始化完成"""
    if torch.cuda.is_available():
        # 创建一个小矩阵并进行一次矩阵乘法来预热GPU
        a = torch.rand(100, 100, device='cuda')
        b = torch.rand(100, 100, device='cuda')
        torch.matmul(a, b)
        # 同步GPU以确保计算完成
        torch.cuda.synchronize()
        print("GPU warmed up")

def matrix_multiply_with_transfer(a, b, device):
    """包含数据传输时间的矩阵乘法"""
    start_total = time.time()
    
    # 将矩阵移动到指定设备（CPU或GPU）
    a = a.to(device)
    b = b.to(device)
    
    # 同步设备以确保传输完成
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # 记录计算开始时间
    start_compute = time.time()
    
    # 执行矩阵乘法
    result = torch.matmul(a, b)
    
    # 同步设备以确保计算完成
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # 记录计算结束时间
    end_compute = time.time()
    
    # 将结果移回CPU（如果需要）
    if device == 'cuda':
        result_cpu = result.cpu()
    else:
        result_cpu = result
    
    # 记录总结束时间
    end_total = time.time()
    
    # 计算执行时间
    compute_time = end_compute - start_compute
    total_time = end_total - start_total
    
    return result_cpu, compute_time, total_time

# 设置矩阵大小（使用更大的矩阵以充分发挥GPU优势）
matrix_size = 5000

print(f"Creating {matrix_size}x{matrix_size} matrices...")

# 创建两个随机矩阵
matrix_a = torch.rand(matrix_size, matrix_size)
matrix_b = torch.rand(matrix_size, matrix_size)

# 预热GPU
if torch.cuda.is_available():
    gpu_warmup()

print("Running matrix multiplication on CPU...")
# 在CPU上执行矩阵乘法
cpu_result, cpu_compute_time, cpu_total_time = matrix_multiply_with_transfer(matrix_a, matrix_b, 'cpu')

# 检查是否有可用的GPU
if torch.cuda.is_available():
    print("Running matrix multiplication on GPU...")
    # 在GPU上执行矩阵乘法
    gpu_result, gpu_compute_time, gpu_total_time = matrix_multiply_with_transfer(matrix_a, matrix_b, 'cuda')
    
    # 验证结果是否相同
    is_correct = torch.allclose(cpu_result, gpu_result, rtol=1e-3, atol=1e-3)
    
    print(f"\nMatrix size: {matrix_size}x{matrix_size}")
    print(f"CPU computation time: {cpu_compute_time:.4f} seconds")
    print(f"CPU total time (including memory operations): {cpu_total_time:.4f} seconds")
    print(f"GPU computation time: {gpu_compute_time:.4f} seconds")
    print(f"GPU total time (including data transfer): {gpu_total_time:.4f} seconds")
    print(f"Computation speedup: {cpu_compute_time / gpu_compute_time:.2f}x")
    print(f"Total speedup: {cpu_total_time / gpu_total_time:.2f}x")
    print(f"Results match: {is_correct}")
else:
    print(f"\nMatrix size: {matrix_size}x{matrix_size}")
    print(f"CPU computation time: {cpu_compute_time:.4f} seconds")
    print(f"CPU total time: {cpu_total_time:.4f} seconds")
    print("GPU not available")

# 打印结果矩阵的一小部分以进行验证
print("\nResult matrix (top-left 5x5):")
print(cpu_result[:5, :5])