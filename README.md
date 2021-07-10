# DeepPath: A Reinforcement Learning Method for Knowledge Graph Reasoning

PyTorch implementation of the algorithm DeepPath. Paper available here: https://arxiv.org/abs/1707.06690

The code has been adapted from the original DeepPath code in TensorFlow available here: https://github.com/xwhan/DeepPath

## Access the dataset
For dawnloading NELL-995 dataset run upload.sh script; FB15k-237 is avaliable [here](https://drive.google.com/file/d/1klWL11nW3ZS6b2MtLW0MHnXu-XlJqDyA/view?usp=sharing)
## How to run our code 
1. unzip the data, put the data folder in the code directory
2. run the following scripts within `scripts/`
    *   `./pathfinder.sh ${relation_name}`  # find the reasoning paths, this is RL training, it might take sometime
    *   `./fact_prediction_eval.py ${relation_name}` # calculate & print the fact prediction results
    *   `./link_prediction_eval.sh ${relation_name}` # calculate & print the link prediction results

    Examples (the relation_name can be found in `NELL-995/tasks/`):
    * `./pathfinder.sh concept_athletehomestadium` 
    * `./fact_prediction_eval.py concept_athletehomestadium`
    * `./link_prediction_eval.sh concept_athletehomestadium`
3. Reasoning path is arleady put in the dataset, you can directly run fact_prediction_eval.py or link_prediction_eval.sh to get the final results for each reasoning task

## Format of the dataset
1. `raw.kb`: the raw kb data from NELL system
2. `kb_env_rl.txt`: we add inverse triples of all triples in `raw.kb`, this file is used as the KG for reasoning
3. `entity2vec.bern/relation2vec.bern`: transE embeddings to represent out RL states, can be trained using [TransX implementations by thunlp](https://github.com/thunlp/Fast-TransX)
4. `tasks/`: each task is a particular reasoning relation
    * `tasks/${relation}/*.vec`: trained TransH Embeddings
    * `tasks/${relation}/*.vec_D`: trained TransD Embeddings
    * `tasks/${relation}/*.bern`: trained TransR Embedding trained
    * `tasks/${relation}/*.unif`: trained TransE Embeddings
    * `tasks/${relation}/transX`: triples used to train the KB embeddings
    * `tasks/${relation}/train.pairs`: train triples in the PRA format
    * `tasks/${relation}/test.pairs`: test triples in the PRA format
    * `tasks/${relation}/path_to_use.txt`: reasoning paths found the RL agent
    * `tasks/${relation}/path_stats.txt`: path frequency of randomised BFS
