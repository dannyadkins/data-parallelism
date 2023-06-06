 This implements a naive data parallelism per chapter 3 of Jacob Hilton's Deep Learning Curriculum at https://github.com/jacobhilton/deep_learning_curriculum/blob/master/3-Training-at-Scale.md.

 Future plans:
 - implement async gradient updates, model parallelism, tensor parallelism, and pipeline parallelism
 - implement proper data sharding using memmapping 
 - hardcode some different communication strategies like ring-allreduce 

Interesting findings 
- for some reason, if I loop with to test many different hyperparams, the first loop is always the fastest, regardless of what the training hyperparams are. Could this be some kind of MPI cold-start? 
- accuracy vs. steps_per_update in grad accumulation is super spiky 

 Notes:
 - getting "Triton Error [CUDA]: an illegal memory access was encountered" when I try to use my custom FusedAdam kernel 