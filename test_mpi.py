# test_mpi.py
from mpi4py import MPI
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    data = ["text" + str(i) for i in range(10)]  # Mimic batch_texts
    logger.info(f"Rank 0: Broadcasting {len(data)} items")
else:
    data = None

data = comm.bcast(data, root=0)
logger.info(f"Rank {rank}: Received {len(data)} items")