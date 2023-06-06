 This implements a naive data parallelism per chapter 3 of Jacob Hilton's Deep Learning Curriculum at https://github.com/jacobhilton/deep_learning_curriculum/blob/master/3-Training-at-Scale.md.

 Future plans:
 - implement async gradient updates, model parallelism, tensor parallelism, and pipeline parallelism
 - implement proper data sharding using memmapping 
 - hardcode some different communication strategies like ring-allreduce 


 Notes:
 - getting "Triton Error [CUDA]: an illegal memory access was encountered" when I try to use my custom FusedAdam kernel 