
def _head_text_cpu(log_file, mem):
    """
    """
    qsub_text = '''#!/bin/bash
#PBS -S /bin/bash
#PBS -o %s '''%log_file + '''
#PBS -j oe
#PBS -l nodes=1:ppn=1,mem=%dmb,walltime=720:00:00 \n'''%mem
    return qsub_text


def _head_text_gpu(log_file, mem):
    """
    """
    qsub_text = '''#!/bin/bash
#PBS -S /bin/bash
#PBS -o %s '''%log_file + '''
#PBS -j oe
#PBS -q cuda
#PBS -W x=GRES:gpus@1
#PBS -l nodes=1:gpus,mem=%dmb,walltime=720:00:00
hostname
echo $PBS_JOBID
module load cuda/7.5
source /home/tnguye46/gchmc/scripts/job_scripts/gpu_free.sh \n'''%mem
    return qsub_text


def head_text_4_qsub_script(cpu_or_gpu, log_file, mem=2048):
    if cpu_or_gpu == "cpu":
        return _head_text_cpu(log_file, mem)
    elif cpu_or_gpu == "gpu":
        return _head_text_gpu(log_file, mem)
    else:
        raise Exception("Unknown "+cpu_or_gpu)
