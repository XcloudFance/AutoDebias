"""
Memory management utilities
"""
import gc
import torch
import logging

logger = logging.getLogger(__name__)

def clean_memory():
    """
    Clean up GPU memory
    
    This function:
    1. Calls Python garbage collector
    2. Clears PyTorch CUDA cache
    3. Logs memory usage before and after cleanup
    """
    # Get initial memory usage
    if torch.cuda.is_available():
        initial_memory = torch.cuda.memory_allocated() / (1024 ** 3)
        initial_max_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
        
        logger.info(f"Memory before cleanup: {initial_memory:.2f} GB (max: {initial_max_memory:.2f} GB)")
        
        # Run garbage collector
        gc.collect()
        
        # Empty CUDA cache
        torch.cuda.empty_cache()
        
        # Get memory usage after cleanup
        current_memory = torch.cuda.memory_allocated() / (1024 ** 3)
        current_max_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
        
        logger.info(f"Memory after cleanup: {current_memory:.2f} GB (max: {current_max_memory:.2f} GB)")
        logger.info(f"Memory freed: {initial_memory - current_memory:.2f} GB")
        
        return {
            "initial_memory": initial_memory,
            "initial_max_memory": initial_max_memory,
            "current_memory": current_memory,
            "current_max_memory": current_max_memory,
            "freed_memory": initial_memory - current_memory
        }
    else:
        logger.info("CUDA not available, running garbage collection only")
        gc.collect()
        return {
            "cuda_available": False
        }

def get_memory_info():
    """获取当前GPU内存使用情况"""
    if not torch.cuda.is_available():
        return {"error": "CUDA不可用"}
    
    try:
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)    # GB
        max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB
        
        return {
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "max_allocated_gb": max_allocated,
            "utilization": allocated / reserved if reserved > 0 else 0
        }
    except Exception as e:
        logger.error(f"获取内存信息时出错: {e}")
        return {"error": str(e)}

def print_memory_stats():
    """打印内存统计信息"""
    if not torch.cuda.is_available():
        logger.info("CUDA不可用，无法获取内存统计信息")
        return
    
    memory_info = get_memory_info()
    
    if "error" in memory_info:
        logger.error(f"获取内存统计信息时出错: {memory_info['error']}")
        return
    
    logger.info(f"内存统计信息:")
    logger.info(f"  已分配: {memory_info['allocated_gb']:.2f} GB")
    logger.info(f"  已保留: {memory_info['reserved_gb']:.2f} GB")
    logger.info(f"  最大分配: {memory_info['max_allocated_gb']:.2f} GB")
    logger.info(f"  利用率: {memory_info['utilization']:.2f}")